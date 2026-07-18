import os
import random
import re
import numpy as np
import torch
import torch.nn as nn
from models import build_model, get_loss
from utils import get_clip_normalization, prepare_model_inputs
# [新增] 引入调度器组件
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# [新增] 引入 timm 的 EMA 模块
try:
    from timm.utils import ModelEmaV2
except ImportError:
    print("[Warning] timm library not found! EMA will not be available.")
    print("Please install it using: pip install timm")
    ModelEmaV2 = None


def _strip_wrapper_prefixes(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        name = key
        while name.startswith("module.") or name.startswith("_orig_mod."):
            name = name[7:] if name.startswith("module.") else name[10:]
        if name in cleaned:
            raise RuntimeError(f"Checkpoint key collision after prefix cleanup: {name}")
        cleaned[name] = value
    return cleaned


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint_epoch(checkpoint, path):
    if isinstance(checkpoint, dict) and checkpoint.get("epoch") is not None:
        return int(checkpoint["epoch"])
    match = re.search(r"model_epoch_(\d+)\.pth$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _rng_state():
    np_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (np_state[0], torch.from_numpy(np_state[1].copy()), np_state[2], np_state[3], np_state[4]),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.cpu().numpy().astype(np.uint32, copy=False), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count differs from resume checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])

class Trainer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.total_steps = 0
        self.update_steps = 0
        self.skipped_update_steps = 0
        self.start_epoch = 0
        self.step_bias = 0
        self.resume_mode = bool(getattr(opt, "resume", False))
        self.resume_best_metrics = {
            "best_acc": 0.0, "best_ap": 0.0, "best_auc": 0.0,
            "best_f1": 0.0, "best_epoch": 0,
        }
        if self.resume_mode and opt.fine_tune:
            raise ValueError("--resume and --fine-tune are mutually exclusive")
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = (
            torch.device("cuda:{}".format(opt.gpu_ids[0]))
            if opt.gpu_ids
            else torch.device("cpu")
        )
        self.model = build_model(opt.arch)
        requested_chunk = int(getattr(opt, "region_checkpoint_chunk_size", -1))
        self.region_checkpoint_chunk_size = (
            60 if requested_chunk < 0 and 15 * opt.batch_size > 120
            else (0 if requested_chunk < 0 else requested_chunk)
        )
        if hasattr(self.model, "backbone"):
            self.model.backbone.checkpoint_chunk_size = self.region_checkpoint_chunk_size
        print(
            "[Info] Effective model switches: "
            f"LipFD.no_innov={getattr(self.model, 'no_innov', 'N/A')}, "
            f"use_modality_bias={getattr(self.model, 'use_modality_bias', 'N/A')}, "
            f"use_attn_bias={getattr(self.model, 'use_attn_bias', 'N/A')}, "
            f"use_se_fusion={getattr(self.model, 'use_se_fusion', 'N/A')}, "
            f"use_residual_cls={getattr(self.model, 'use_residual_cls', 'N/A')}, "
            f"Region.with_pe={getattr(getattr(self.model, 'backbone', None), 'with_pe', 'N/A')}, "
            f"Region.with_se={getattr(getattr(self.model, 'backbone', None), 'with_se', 'N/A')}, "
            f"Region.checkpoint_chunk={self.region_checkpoint_chunk_size}"
        )

        self._resume_checkpoint = None
        if opt.fine_tune or self.resume_mode:
            checkpoint = _load_checkpoint(opt.pretrained_model)
            model_state = checkpoint.get("model", checkpoint)
            if not isinstance(model_state, dict):
                raise RuntimeError("Checkpoint does not contain a model state_dict")
            self._load_model_state(
                model_state,
                allow_partial=bool(opt.fine_tune and opt.allow_partial_load),
            )
            saved_epoch = _checkpoint_epoch(checkpoint, opt.pretrained_model)
            if self.resume_mode:
                if saved_epoch is None:
                    raise RuntimeError(
                        "Checkpoint has no epoch metadata; use --fine-tune for weight-only loading"
                    )
                self.start_epoch = saved_epoch + 1
                saved_index = checkpoint.get("checkpoint_index")
                saved_index = saved_epoch if saved_index is None else int(saved_index)
                self.step_bias = saved_index - saved_epoch
                self._resume_checkpoint = checkpoint
            elif saved_epoch is not None:
                saved_index = checkpoint.get("checkpoint_index")
                self.step_bias = (saved_epoch if saved_index is None else int(saved_index)) + 1
            print(
                f"[LOAD] mode={'resume' if self.resume_mode else 'fine-tune'} | "
                f"file={os.path.basename(opt.pretrained_model)} | epoch={saved_epoch}"
            )

        if opt.fix_encoder:
            for name, p in self.model.named_parameters():
                if name.split(".")[0] in ["encoder"]:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        if opt.fix_backbone:
            raise NotImplementedError(
                "--fix_backbone was previously ignored. It is now rejected instead of silently changing nothing."
            )
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        params = self.trainable_params

        if opt.optim == "adamw":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("optim should be [sgd, adam, adamw]")

        # ============================
        # LR Scheduler: Warmup + CosineAnnealingLR (NO restart)
        #
        # 关键点（配合本项目“epoch 末尾 step 一次”的用法）：
        # 1) lr 会在“下一轮 epoch 开始时”生效（因为 step() 放在 epoch 末尾）。
        # 2) SequentialLR 在 last_epoch 命中 milestone 时会切换到下一个 scheduler，并调用 next_scheduler.step(0)，
        #    因此命中 milestone 的那一次不会再推进 warmup_scheduler。
        # 3) 约定 warmup_epochs=N 表示 warmup 覆盖前 N 个 epoch。
        #    为了让第 N 个 warmup epoch 能到达 base lr（而不是延后到第 N+1 个），
        #    LinearLR.total_iters 应该设置为 N-1（当 N>=2）。当 N==1 时用 total_iters=1。
        # ============================
        self.scheduler = None
        if opt.cosine_annealing:
            warmup_epochs = max(int(getattr(opt, "warmup_epochs", 0)), 0)
            warmup_epochs = min(warmup_epochs, int(getattr(opt, "epoch", warmup_epochs)))  # ✅ 防止 warmup > 总epoch
            eta_min = float(getattr(opt, "eta_min", 1e-6))

            cosine_epochs = max(opt.epoch - warmup_epochs, 1)
            T_max = max(cosine_epochs - 1, 1)

            main_scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)

            if warmup_epochs > 0:
                # ✅ 关键修正：让 warmup 在切换到 cosine 之前就达到 base_lr，避免切换瞬间上跳
                warmup_total_iters = max(warmup_epochs - 1, 1)
                warmup_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=0.001,
                    end_factor=1.0,
                    total_iters=warmup_total_iters
                )
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_epochs]
                )
            else:
                self.scheduler = main_scheduler
        else:
            self.scheduler = None

        self.criterion = get_loss().to(self.device)
        self.criterion1 = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 确保模型先移动到 GPU
        self.model.to(self.device)
        
        # -----------------------------------------------------------
        # [新增] EMA 初始化
        # -----------------------------------------------------------
        self.model_ema = None
        if self.opt.use_ema and ModelEmaV2 is None:
            raise RuntimeError("--use_ema requires timm")
        if self.opt.use_ema:
            # decay: 衰减率。0.999 是通用值。
            # 如果你的 Batch Size 很小(8)，这个值非常重要，它能平滑掉梯度的剧烈抖动。
            self.model_ema = ModelEmaV2(self.model, decay=self.opt.ema_decay, device=None)
            self.model_ema.eval()
            print(f"[Info] Model EMA initialized with decay {self.opt.ema_decay}")
        
        # 梯度累积相关参数
        self.accumulation_steps = opt.accumulation_steps
        self.accumulation_count = 0
        
        # -------------------------------------------------------
        # [AMP 修改 1] 初始化混合精度组件
        # -------------------------------------------------------
        self.device_type = self.device.type
        self.use_amp = opt.use_amp and self.device_type == 'cuda'
        self.compile_enabled = False
        self.compile_mode = None
        if self.use_amp:
            # 初始化梯度缩放器，用于防止 FP16 下梯度的下溢出
            self.scaler = torch.amp.GradScaler("cuda")

        if self._resume_checkpoint is not None:
            self._restore_training_state(self._resume_checkpoint)
            self._resume_checkpoint = None

    def _load_model_state(self, source_state, allow_partial=False):
        source_state = _strip_wrapper_prefixes(source_state)
        target_state = self.model.state_dict()
        unexpected = sorted(set(source_state) - set(target_state))
        missing = sorted(set(target_state) - set(source_state))
        mismatched = sorted(
            key for key in set(source_state).intersection(target_state)
            if tuple(source_state[key].shape) != tuple(target_state[key].shape)
        )
        matched = [
            key for key in set(source_state).intersection(target_state)
            if key not in mismatched
        ]
        matched_numel = sum(target_state[key].numel() for key in matched)
        total_numel = sum(value.numel() for value in target_state.values())
        coverage = matched_numel / max(total_numel, 1)
        print(
            f"[LOAD] coverage={coverage:.6%} | missing={len(missing)} | "
            f"unexpected={len(unexpected)} | shape_mismatch={len(mismatched)}"
        )
        if (missing or unexpected or mismatched) and not allow_partial:
            raise RuntimeError(
                "Refusing partial checkpoint load. "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
                f"shape_mismatch={mismatched[:5]}. Use --allow_partial_load "
                "only for an intentional cross-structure fine-tune."
            )
        load_state = {
            key: value for key, value in source_state.items()
            if key not in mismatched
        }
        self.model.load_state_dict(load_state, strict=not allow_partial)

    def _training_config(self):
        return {
            "arch": self.opt.arch,
            "optim": self.opt.optim,
            "batch_size": int(self.opt.batch_size),
            "accumulation_steps": int(self.accumulation_steps),
            "fix_encoder": bool(self.opt.fix_encoder),
            "use_amp": bool(self.use_amp),
            "use_ema": bool(self.model_ema is not None),
            "ema_decay": float(self.opt.ema_decay),
            "cosine_annealing": bool(self.opt.cosine_annealing),
            "warmup_epochs": int(self.opt.warmup_epochs),
            "eta_min": float(self.opt.eta_min),
            "planned_epochs": int(self.opt.epoch),
            "region_checkpoint_chunk_size": int(self.region_checkpoint_chunk_size),
            "compile": bool(self.opt.compile and not self.opt.no_compile),
            "compile_mode": self.opt.compile_mode,
            "real_list_path": os.path.abspath(self.opt.real_list_path),
            "fake_list_path": os.path.abspath(self.opt.fake_list_path),
        }

    def _restore_training_state(self, checkpoint):
        required = {"checkpoint_version", "epoch", "optimizer", "best_metrics", "rng_state", "training_config"}
        if self.scheduler is not None:
            required.add("scheduler")
        if self.use_amp:
            required.add("scaler")
        if self.model_ema is not None:
            required.add("model_ema")
        missing = sorted(required - set(checkpoint))
        if missing:
            raise RuntimeError(
                f"Incomplete resume checkpoint; missing={missing}. Use --fine-tune for weight-only loading."
            )
        if int(checkpoint["checkpoint_version"]) < 2:
            raise RuntimeError("Checkpoint predates complete resume format; use --fine-tune")
        current_config = self._training_config()
        mismatches = {
            key: (checkpoint["training_config"].get(key), value)
            for key, value in current_config.items()
            if checkpoint["training_config"].get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Resume configuration mismatch: {mismatches}")
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.use_amp:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if self.model_ema is not None:
            self.model_ema.module.load_state_dict(
                _strip_wrapper_prefixes(checkpoint["model_ema"]), strict=True
            )
            self.model_ema.eval()
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.update_steps = int(checkpoint.get("update_steps", 0))
        self.skipped_update_steps = int(checkpoint.get("skipped_update_steps", 0))
        self.resume_best_metrics.update(checkpoint["best_metrics"])
        _restore_rng_state(checkpoint["rng_state"])
        self.optimizer.zero_grad(set_to_none=True)
        print(
            f"[RESUME] start_epoch={self.start_epoch} | updates={self.update_steps} | "
            f"lr={self.optimizer.param_groups[0]['lr']:.3e}"
        )

    def set_input(self, input):
        self.label = input[2].to(self.device, non_blocking=True).long()

        if not hasattr(self, 'mean_tensor'):
            self.mean_tensor, self.std_tensor = get_clip_normalization(self.device)

        self.input, self.crops = prepare_model_inputs(
            input[0], input[1], self.device, self.mean_tensor, self.std_tensor
        )
        # Preserve the historical attribute for external/debug callers. The old
        # in-place normalization meant it referenced this normalized tensor too.
        self.input_raw = self.input

    def forward(self):
        # -------------------------------------------------------
        # [AMP 修改 2] 前向传播上下文管理
        # -------------------------------------------------------
        # 使用 autocast 自动将部分算子转为 FP16 运行
        if (
            self.compile_enabled
            and self.compile_mode == "reduce-overhead"
            and self.device_type == "cuda"
            and hasattr(torch.compiler, "cudagraph_mark_step_begin")
        ):
            torch.compiler.cudagraph_mark_step_begin()
        with torch.amp.autocast(self.device_type, enabled=self.use_amp):
            self._forward_impl()

    def _forward_impl(self):
        self.get_features()
        self.output, self.weights_max, self.weights_org = self.model(
            self.crops, self.features
        )
        
        # -------------------------------------------------------
        # [AMP 修改 3] 稳定性保障：计算 Loss 前强制转回 FP32
        # -------------------------------------------------------
        # 防止 CrossEntropyLoss 中的 exp() 计算在 FP16 下溢出导致 NaN
        if self.use_amp:
            self.output = self.output.float()
            self.weights_max = self.weights_max.float()
            self.weights_org = self.weights_org.float()
        
        self.loss_ral = self.criterion(self.weights_max, self.weights_org)
        self.loss_ce = self.criterion1(self.output, self.label)
        self.loss = 0.01 * self.loss_ral + 1.0 * self.loss_ce

    def get_loss(self):
        loss = self.loss.data.tolist()
        return loss[0] if isinstance(loss, type(list())) else loss

    def get_individual_losses(self):
        loss_ral = self.loss_ral.data.tolist()
        loss_ral = loss_ral[0] if isinstance(loss_ral, type(list())) else loss_ral
        loss_ce = self.loss_ce.data.tolist()
        loss_ce = loss_ce[0] if isinstance(loss_ce, type(list())) else loss_ce
        return loss_ral, loss_ce

    def _unwrapped_model(self):
        return getattr(self.model, "_orig_mod", self.model)

    def _step_optimizer(self, gradient_multiplier=1.0):
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        if gradient_multiplier != 1.0:
            for param in self.trainable_params:
                if param.grad is not None:
                    param.grad.mul_(gradient_multiplier)
        torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
        if not self.use_amp:
            self.optimizer.step()
            return True
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return self.scaler.get_scale() >= scale_before

    def _finish_optimizer_step(self, stepped):
        if stepped:
            if self.model_ema is not None:
                self.model_ema.update(self._unwrapped_model())
            self.update_steps += 1
        else:
            self.skipped_update_steps += 1
        self.accumulation_count = 0
        self.optimizer.zero_grad(set_to_none=True)

    def optimize_parameters(self):
        if self.accumulation_count == 0:
            self.optimizer.zero_grad(set_to_none=True)
            
        # -------------------------------------------------------
        # [AMP 修改 4] 反向传播 (Backward)
        # -------------------------------------------------------
        loss_scaled = self.loss / self.accumulation_steps
        
        if self.use_amp:
            # 使用 scaler 缩放 loss，防止梯度下溢
            self.scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()
        
        self.accumulation_count += 1
        
        # -------------------------------------------------------
        # [AMP 修改 5] 参数更新 (Step)
        # -------------------------------------------------------
        if self.accumulation_count >= self.accumulation_steps:
            self._finish_optimizer_step(self._step_optimizer())

    def get_features(self):
        self.features = self.model.get_features(self.input)

    def train(self, mode=True):
        super().train(mode)
        if self.model_ema is not None:
            self.model_ema.eval()
        return self

    def eval(self):
        return self.train(False)
        # 注意：EMA模型不需要手动eval，它始终处于评估模式

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_networks(
        self, save_filename, save_optimizer=False, training_epoch=None,
        checkpoint_index=None, best_metrics=None,
    ):
        """
        保存模型权重。
        Args:
            save_filename: 文件名
            save_optimizer: 是否保存优化器状态（默认False，以节省空间）
        """
        save_path = os.path.join(self.save_dir, save_filename)
        os.makedirs(self.save_dir, exist_ok=True)

        # 1. 基础部分：始终保存模型参数
        state_dict = {
            "checkpoint_version": 2,
            "model": self._unwrapped_model().state_dict(),
            "epoch": training_epoch,
            "checkpoint_index": checkpoint_index,
            "total_steps": self.total_steps,
            "update_steps": self.update_steps,
            "skipped_update_steps": self.skipped_update_steps,
            "training_config": self._training_config(),
        }
        
        # 2. EMA 部分：这是最宝贵的推理权重，必须保存
        if hasattr(self, 'model_ema') and self.model_ema is not None:
            state_dict["model_ema"] = self.model_ema.module.state_dict()

        # 3. 优化器部分：体积巨大(3GB+)，仅在需要断点续训时才保存
        if save_optimizer:
            if training_epoch is None or checkpoint_index is None or best_metrics is None:
                raise ValueError("Complete resume checkpoint requires epoch/index/best_metrics")
            if self.accumulation_count != 0:
                raise RuntimeError("Cannot save full resume state with incomplete gradients")
            state_dict["optimizer"] = self.optimizer.state_dict()
            state_dict["scheduler"] = self.scheduler.state_dict() if self.scheduler is not None else None
            if self.use_amp:
                state_dict["scaler"] = self.scaler.state_dict()
            state_dict["best_metrics"] = dict(best_metrics)
            state_dict["rng_state"] = _rng_state()

        torch.save(state_dict, save_path)
        # print(f"[Info] Saved model to {save_path} (Optimizer: {'Yes' if save_optimizer else 'No'})")

    def step_remainder_gradients(self):
        """处理 Epoch 结束时未满足累积步数的剩余梯度"""
        if self.accumulation_count > 0:
            scale_factor = self.accumulation_steps / self.accumulation_count
            stepped = self._step_optimizer(gradient_multiplier=scale_factor)
            self._finish_optimizer_step(stepped)
            if not stepped:
                print("[WARN] Remainder optimizer step skipped due to non-finite gradients")
