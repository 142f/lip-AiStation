import os
import re

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from models import build_model, get_loss

try:
    from timm.utils import ModelEmaV2
except ImportError:
    print("[Warning] timm library not found! EMA will not be available.")
    print("Please install it using: pip install timm")
    ModelEmaV2 = None


def _strip_state_dict_prefixes(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        name = key
        while name.startswith("module.") or name.startswith("_orig_mod."):
            if name.startswith("module."):
                name = name[7:]
            elif name.startswith("_orig_mod."):
                name = name[10:]
        cleaned[name] = value
    return cleaned


def _extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Unsupported checkpoint format for model weights.")


def _parse_epoch_from_path(path):
    filename = os.path.basename(path)
    # Supports names like model_epoch_29.pth or epoch29.pth
    m = re.search(r"epoch[_-]?(\d+)", filename)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


class Trainer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.total_steps = 0
        self.update_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.step_bias = 0

        self.device = (
            torch.device(f"cuda:{opt.gpu_ids[0]}") if opt.gpu_ids else torch.device("cpu")
        )

        self.model = build_model(opt.arch)

        if opt.fine_tune:
            self._load_pretrained(opt.pretrained_model)

        self._apply_freeze_policy()
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters left after freeze policy.")

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
                params,
                lr=opt.lr,
                momentum=0.9,
                weight_decay=opt.weight_decay,
            )
        else:
            raise ValueError("optim should be [sgd, adam, adamw]")

        self.scheduler = None
        if opt.cosine_annealing:
            warmup_epochs = max(int(getattr(opt, "warmup_epochs", 0)), 0)
            warmup_epochs = min(warmup_epochs, int(getattr(opt, "epoch", warmup_epochs)))
            eta_min = float(getattr(opt, "eta_min", 1e-6))

            cosine_epochs = max(opt.epoch - warmup_epochs, 1)
            t_max = max(cosine_epochs - 1, 1)
            main_scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)

            if warmup_epochs > 0:
                warmup_total_iters = max(warmup_epochs - 1, 1)
                warmup_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=0.001,
                    end_factor=1.0,
                    total_iters=warmup_total_iters,
                )
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                self.scheduler = main_scheduler

        self.criterion = get_loss().to(self.device)
        self.criterion1 = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.model.to(self.device)

        self.model_ema = None
        if self.opt.use_ema and ModelEmaV2 is not None:
            self.model_ema = ModelEmaV2(self.model, decay=self.opt.ema_decay, device=self.device)
            print(f"[Info] Model EMA initialized with decay {self.opt.ema_decay}")

        self.accumulation_steps = max(1, int(opt.accumulation_steps))
        self.accumulation_count = 0

        self.use_amp = bool(opt.use_amp and torch.cuda.is_available())
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.region_consistency_weight = float(getattr(opt, "region_consistency_weight", 0.02))

    def _apply_freeze_policy(self):
        if self.opt.fix_encoder:
            for name, p in self.model.named_parameters():
                p.requires_grad = not name.startswith("encoder.")

        if getattr(self.opt, "fix_backbone", False):
            backbone = getattr(self.model, "backbone", None)
            if backbone is None:
                print("[Warning] --fix_backbone is set but model has no `backbone` attribute.")
            else:
                for p in backbone.parameters():
                    p.requires_grad = False

    def _load_pretrained(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model_state = _extract_model_state(checkpoint)
        model_state = _strip_state_dict_prefixes(model_state)

        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        if missing or unexpected:
            details = (
                f"missing={len(missing)} unexpected={len(unexpected)}. "
                f"missing_head={missing[:8]} unexpected_head={unexpected[:8]}"
            )
            if getattr(self.opt, "allow_partial_load", False):
                print(f"[Warning] Partial checkpoint load allowed: {details}")
            else:
                raise RuntimeError(
                    "Checkpoint/model mismatch detected. "
                    "Use matching --arch/ablation config, or pass --allow_partial_load to override. "
                    + details
                )

        if isinstance(checkpoint, dict):
            self.total_steps = int(checkpoint.get("total_steps", 0) or 0)
            self.update_steps = int(checkpoint.get("update_steps", 0) or 0)

        parsed_epoch = _parse_epoch_from_path(ckpt_path)
        self.step_bias = (parsed_epoch + 1) if parsed_epoch is not None else 0

        print(f"Model loaded @ {os.path.basename(ckpt_path)}")

    @staticmethod
    def _unwrap_model(model):
        return model._orig_mod if hasattr(model, "_orig_mod") else model

    def _state_dict_for_save(self, model):
        base = self._unwrap_model(model)
        return _strip_state_dict_prefixes(base.state_dict())

    def set_input(self, input_data):
        x = input_data[0].to(self.device, non_blocking=True)
        self.label = input_data[2].to(self.device, non_blocking=True).long()

        if not hasattr(self, "mean_tensor"):
            self.mean_tensor = torch.tensor(
                [0.48145466, 0.4578275, 0.40821073], device=self.device
            ).view(1, 3, 1, 1)
            self.std_tensor = torch.tensor(
                [0.26862954, 0.26130258, 0.27577711], device=self.device
            ).view(1, 3, 1, 1)

        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)

        self.input_raw = x
        self.input = x.sub_(self.mean_tensor).div_(self.std_tensor)

        self.crops = []
        raw_crops = input_data[1]
        for scale_list in raw_crops:
            processed_scale = []
            for crop_batch in scale_list:
                processed_scale.append(crop_batch.to(self.device, non_blocking=True))
            self.crops.append(processed_scale)

    def forward(self):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            self._forward_impl()

    def _forward_impl(self):
        self.get_features()
        forward_out = self.model.forward(self.crops, self.features)
        aux_losses = {}
        if not isinstance(forward_out, (tuple, list)) or len(forward_out) < 3:
            raise RuntimeError("Model forward must return at least (logits, weights_max, weights_org).")

        self.output, self.weights_max, self.weights_org = forward_out[:3]
        if len(forward_out) >= 4 and isinstance(forward_out[3], dict):
            aux_losses = forward_out[3]

        if self.use_amp:
            self.output = self.output.float()
            self.weights_max = self.weights_max.float()
            self.weights_org = self.weights_org.float()

        self.loss_ral = self.criterion(self.weights_max, self.weights_org)
        self.loss_ce = self.criterion1(self.output, self.label)
        self.loss_region_consistency = torch.zeros((), device=self.device)
        if "region_consistency_loss" in aux_losses:
            self.loss_region_consistency = aux_losses["region_consistency_loss"]
            if self.use_amp:
                self.loss_region_consistency = self.loss_region_consistency.float()

        self.loss = (
            0.01 * self.loss_ral
            + 1.0 * self.loss_ce
            + self.region_consistency_weight * self.loss_region_consistency
        )

    def get_loss(self):
        loss = self.loss.data.tolist()
        return loss[0] if isinstance(loss, list) else loss

    def get_individual_losses(self):
        loss_ral = self.loss_ral.data.tolist()
        loss_ral = loss_ral[0] if isinstance(loss_ral, list) else loss_ral
        loss_ce = self.loss_ce.data.tolist()
        loss_ce = loss_ce[0] if isinstance(loss_ce, list) else loss_ce
        return loss_ral, loss_ce

    def _optimizer_step(self, grads_unscaled=False):
        if self.use_amp:
            if not grads_unscaled:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        if self.model_ema is not None:
            self.model_ema.update(self.model)

        self.accumulation_count = 0
        self.optimizer.zero_grad(set_to_none=True)
        self.update_steps += 1

    def optimize_parameters(self):
        if self.accumulation_count == 0:
            self.optimizer.zero_grad(set_to_none=True)

        loss_scaled = self.loss / self.accumulation_steps
        if self.use_amp:
            self.scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        self.accumulation_count += 1
        if self.accumulation_count >= self.accumulation_steps:
            self._optimizer_step()

    def get_features(self):
        self.features = self.model.get_features(self.input)

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_networks(self, save_filename, save_optimizer=False):
        save_path = os.path.join(self.save_dir, save_filename)
        os.makedirs(self.save_dir, exist_ok=True)

        state_dict = {
            "model": self._state_dict_for_save(self.model),
            "total_steps": self.total_steps,
            "update_steps": self.update_steps,
        }

        if self.model_ema is not None:
            state_dict["model_ema"] = self._state_dict_for_save(self.model_ema.module)

        if save_optimizer:
            state_dict["optimizer"] = self.optimizer.state_dict()
            if self.use_amp:
                state_dict["scaler"] = self.scaler.state_dict()

        torch.save(state_dict, save_path)

    def step_remainder_gradients(self):
        """Flush remainder grads at epoch end when accumulation is incomplete."""
        if self.accumulation_count <= 0:
            return

        scale_factor = self.accumulation_steps / self.accumulation_count

        if self.use_amp:
            self.scaler.unscale_(self.optimizer)

        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.mul_(scale_factor)

        self._optimizer_step(grads_unscaled=self.use_amp)
        print(f"Debug: Updates remaining gradients at step {self.update_steps}")
