import os
import torch
import torch.nn as nn
import math
from models import build_model, get_loss


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
        self.opt = opt
        self.model = build_model(opt.arch)

        self.step_bias = (
            0
            if not opt.fine_tune
            else int(opt.pretrained_model.split("_")[-1].split(".")[0]) + 1
        )
        if opt.fine_tune:
            state_dict = torch.load(opt.pretrained_model, map_location="cpu")
            self.model.load_state_dict(state_dict["model"])
            self.total_steps = state_dict["total_steps"]
            print(f"Model loaded @ {opt.pretrained_model.split('/')[-1]}")


        if opt.fix_encoder:
            # `params` 变量在这里实际上没有被使用，可以简化
            for name, p in self.model.named_parameters():
                if name.split(".")[0] in ["encoder"]:
                    p.requires_grad = False
                else:
                    p.requires_grad = True # 正确
            # 只将需要训练的参数传递给优化器
            params = filter(lambda p: p.requires_grad, self.model.parameters())
        else:
            # 如果不固定任何部分，所有参数都应该可训练
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

        # 初始化学习率调度器（如果启用）
        if opt.cosine_annealing:
            # 根据项目信息，使用CosineAnnealingWarmRestarts调度器
            # 重启周期设置为20个epoch，最小学习率设置为1e-7
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=20, T_mult=1, eta_min=1e-8
            )
            # 添加一个计数器来跟踪epoch
            self.scheduler_epoch = 0
        else:
            self.scheduler = None

        self.criterion = get_loss().to(self.device)
        self.criterion1 = nn.CrossEntropyLoss()

        self.model.to(opt.gpu_ids[0] if torch.cuda.is_available() else "cpu")
        
        # 梯度累积相关参数
        self.accumulation_steps = opt.accumulation_steps
        self.accumulation_count = 0
        # 用于跟踪实际的参数更新步骤
        self.update_steps = 0

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.crops = [[t.to(self.device) for t in sublist] for sublist in input[1]]
        self.label = input[2].to(self.device).float()

    def forward(self):
        self.get_features()
        self.output, self.weights_max, self.weights_org = self.model.forward(
            self.crops, self.features
        )
        self.output = self.output.view(-1)
        # 分别保存两个损失值
        self.loss_ral = self.criterion(self.weights_max, self.weights_org)
        self.loss_ce = self.criterion1(self.output, self.label)
        # 根据项目规范，CE损失项应乘以0.5的权重系数
        self.loss = self.loss_ral + 0.5 * self.loss_ce

    def get_loss(self):
        loss = self.loss.data.tolist()
        return loss[0] if isinstance(loss, type(list())) else loss

    # 添加获取单独损失值的方法
    def get_individual_losses(self):
        loss_ral = self.loss_ral.data.tolist()
        loss_ral = loss_ral[0] if isinstance(loss_ral, type(list())) else loss_ral
        loss_ce = self.loss_ce.data.tolist()
        loss_ce = loss_ce[0] if isinstance(loss_ce, type(list())) else loss_ce
        return loss_ral, loss_ce

    def optimize_parameters(self):
        # 梯度累积实现
        # 除以accumulation_steps以获得平均梯度
        loss_scaled = self.loss / self.accumulation_steps
        
        # 反向传播
        loss_scaled.backward()
        
        # 根据项目规范，添加梯度裁剪来防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.accumulation_count += 1
        
        # 当达到累积步数时，更新参数并清零梯度
        if self.accumulation_count == self.accumulation_steps:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulation_count = 0
            self.update_steps += 1  # 记录实际的参数更新次数
            
            # 更新学习率调度器（如果启用）
            if self.scheduler is not None:
                self.scheduler.step()
        # 如果不使用梯度累积（accumulation_steps=1），也要确保调度器更新
        elif self.accumulation_steps == 1:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.update_steps += 1
            
            # 更新学习率调度器（如果启用）
            if self.scheduler is not None:
                self.scheduler.step()

    def get_features(self):
        self.features = self.model.get_features(self.input).to(
            self.device
        )  # shape: (batch_size

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, save_filename)

        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)

        # serialize model and optimizer to dict
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_steps": self.update_steps,  # 保存实际更新步骤数
        }

        torch.save(state_dict, save_path)