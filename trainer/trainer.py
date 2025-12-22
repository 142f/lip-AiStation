import os
import torch
import torch.nn as nn
from models import build_model, get_loss

# [新增] 引入 timm 的 EMA 模块
try:
    from timm.utils import ModelEmaV2
except ImportError:
    print("[Warning] timm library not found! EMA will not be available.")
    print("Please install it using: pip install timm")
    ModelEmaV2 = None

class Trainer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = (
            torch.device("cuda:{}".format(opt.gpu_ids[0]))
            if opt.gpu_ids
            else torch.device("cpu")
        )
        self.model = build_model(opt.arch)

        self.step_bias = (
            0
            if not opt.fine_tune
            else int(opt.pretrained_model.split("_")[-1].split(".")[0]) + 1
        )
        if opt.fine_tune:
            state_dict = torch.load(opt.pretrained_model, map_location="cpu")
            self.model.load_state_dict(state_dict["model"], strict=False)
            self.total_steps = state_dict.get("total_steps", 0)
            print(f"Model loaded @ {opt.pretrained_model.split('/')[-1]}")

        if opt.fix_encoder:
            for name, p in self.model.named_parameters():
                if name.split(".")[0] in ["encoder"]:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            params = filter(lambda p: p.requires_grad, self.model.parameters())
        else:
            params = self.model.parameters()

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

        if opt.cosine_annealing:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=20, T_mult=1, eta_min=1e-8
            )
            self.scheduler_epoch = 0
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
        if self.opt.use_ema and ModelEmaV2 is not None:
            # decay: 衰减率。0.999 是通用值。
            # 如果你的 Batch Size 很小(8)，这个值非常重要，它能平滑掉梯度的剧烈抖动。
            self.model_ema = ModelEmaV2(self.model, decay=self.opt.ema_decay, device=self.device)
            print(f"[Info] Model EMA initialized with decay {self.opt.ema_decay}")
        
        # 梯度累积相关参数
        self.accumulation_steps = opt.accumulation_steps
        self.accumulation_count = 0
        self.update_steps = 0
        
        # -------------------------------------------------------
        # [AMP 修改 1] 初始化混合精度组件
        # -------------------------------------------------------
        self.use_amp = opt.use_amp and torch.cuda.is_available()
        if self.use_amp:
            # 初始化梯度缩放器，用于防止 FP16 下梯度的下溢出
            self.scaler = torch.cuda.amp.GradScaler()
            self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    def set_input(self, input):
        # input[0]: 全局图 (B, 3, 1120, 1120)
        # input[1]: 局部 Crops (B, 3, 5, 3, 224, 224) - 现在是 Tensor 了！
        # input[2]: 标签 (B,)
        
        self.input = input[0].to(self.device, non_blocking=True)
        
        # [修改核心] 以前是 List[List[Tensor]]，需要双重循环。
        # 现在是 5D Tensor，直接整块搬运，速度飞快。
        self.crops = input[1].to(self.device, non_blocking=True)
        
        self.label = input[2].to(self.device).long()

    def forward(self):
        # -------------------------------------------------------
        # [AMP 修改 2] 前向传播上下文管理
        # -------------------------------------------------------
        # 使用 autocast 自动将部分算子转为 FP16 运行
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            self._forward_impl()

    def _forward_impl(self):
        self.get_features()
        self.output, self.weights_max, self.weights_org = self.model.forward(
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
        self.loss = 1 * self.loss_ral + 1.0 * self.loss_ce

    def get_loss(self):
        loss = self.loss.data.tolist()
        return loss[0] if isinstance(loss, type(list())) else loss

    def get_individual_losses(self):
        loss_ral = self.loss_ral.data.tolist()
        loss_ral = loss_ral[0] if isinstance(loss_ral, type(list())) else loss_ral
        loss_ce = self.loss_ce.data.tolist()
        loss_ce = loss_ce[0] if isinstance(loss_ce, type(list())) else loss_ce
        return loss_ral, loss_ce

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
            if self.use_amp:
                # A. 先 Unscale (将梯度还原回正常数值，以便进行裁剪)
                self.scaler.unscale_(self.optimizer)
                
                # B. 再 Clip (对正常数值的梯度进行裁剪，防止梯度爆炸)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # C. 最后 Step (如果梯度中有 Inf/NaN，scaler 会自动跳过 step)
                self.scaler.step(self.optimizer)
                
                # D. 更新 Scaler 因子 (准备下一次迭代)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            # [新增] EMA 更新：只有在参数真正 update 后才更新 EMA
            # -------------------------------------------------------
            if self.model_ema is not None:
                self.model_ema.update(self.model)

            self.accumulation_count = 0
            self.optimizer.zero_grad(set_to_none=True)
            self.update_steps += 1
            
            if self.scheduler is not None:
                self.scheduler.step()

    def get_features(self):
        self.features = self.model.get_features(self.input).to(self.device)

    def eval(self):
        self.model.eval()
        # 注意：EMA模型不需要手动eval，它始终处于评估模式

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_networks(self, save_filename, save_optimizer=False):
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
            "model": self.model.state_dict(),
            "total_steps": self.total_steps,
            "update_steps": self.update_steps,
        }
        
        # 2. EMA 部分：这是最宝贵的推理权重，必须保存
        if hasattr(self, 'model_ema') and self.model_ema is not None:
            state_dict["model_ema"] = self.model_ema.module.state_dict()

        # 3. 优化器部分：体积巨大(3GB+)，仅在需要断点续训时才保存
        if save_optimizer:
            state_dict["optimizer"] = self.optimizer.state_dict()
            if self.use_amp:
                state_dict["scaler"] = self.scaler.state_dict()

        torch.save(state_dict, save_path)
        # print(f"[Info] Saved model to {save_path} (Optimizer: {'Yes' if save_optimizer else 'No'})")

    def step_remainder_gradients(self):
        """处理 Epoch 结束时未满足累积步数的剩余梯度"""
        if self.accumulation_count > 0:
            # [修正] 调整梯度比例：因为之前是按 accumulation_steps 平均的，现在只有 accumulation_count 个样本
            # 需要乘以 accumulation_steps / accumulation_count 恢复正确的平均值
            scale_factor = self.accumulation_steps / self.accumulation_count

            if self.use_amp:
    # -------------------------------------------------------
    # [AMP 修改 6] 处理剩余梯度的更新逻辑
    # -------------------------------------------------------
                self.scaler.unscale_(self.optimizer)
                
                # [新增] 修正梯度幅度
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(scale_factor)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # [新增] 修正梯度幅度 (非 AMP 模式也需要)
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(scale_factor)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # [新增] 记得这里也要更新 EMA
            if self.model_ema is not None:
                self.model_ema.update(self.model)
            
            self.optimizer.zero_grad(set_to_none=True)
            self.accumulation_count = 0
            self.update_steps += 1
            print(f"Debug: Updates remaining gradients at step {self.update_steps}")