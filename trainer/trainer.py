import os
import torch
import torch.nn as nn
from models import build_model, get_loss
import torch.cuda.amp as amp


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
        # self.opt = opt  # 重复赋值，可以移除
        self.model = build_model(opt.arch)

        # === 新增/修改：先初始化 scaler，以便加载状态 ===
        # 即使在加载模型时尚未决定是否启用AMP，先创建实例也无妨
        # 它的启用与否由 self.opt.amp 在 forward 和 optimize_parameters 中控制
        self.scaler = amp.GradScaler()

        self.step_bias = (
            0
            if not opt.fine_tune
            else int(opt.pretrained_model.split("_")[-1].split(".")[0]) + 1
        )
        if opt.fine_tune:
            print(f"Loading model from: {opt.pretrained_model}")
            state_dict = torch.load(opt.pretrained_model, map_location="cpu")
            self.model.load_state_dict(state_dict["model"])
            self.total_steps = state_dict.get("total_steps", 0) # 使用 .get() 增加鲁棒性

            # === 修改/新增：加载 GradScaler 的状态 ===
            # 只有当命令行启用 amp 且检查点中包含 scaler 状态时才加载
            if opt.amp and "scaler" in state_dict:
                self.scaler.load_state_dict(state_dict["scaler"])
                print("Successfully loaded GradScaler state.")

            print(f"Model loaded successfully. Resuming from step {self.total_steps}.")


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

        # === 修改/新增：加载优化器状态 (通常在微调时也需要) ===
        if opt.fine_tune and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            print("Successfully loaded optimizer state.")

        self.criterion = get_loss().to(self.device)
        self.criterion1 = nn.CrossEntropyLoss()

        self.model.to(self.device) # 简化了设备指定

    def adjust_learning_rate(self, min_lr=1e-8):
        for param_group in self.optimizer.param_groups:
            if param_group["lr"] < min_lr:
                return False
            param_group["lr"] /= 10.0
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.crops = [[t.to(self.device) for t in sublist] for sublist in input[1]]
        self.label = input[2].to(self.device).float()

    def forward(self):
        # autocast 的 enabled 参数会处理 opt.amp 是 True 还是 False 的情况
        with amp.autocast(enabled=self.opt.amp):
            self.get_features()
            self.output, self.weights_max, self.weights_org = self.model.forward(
                self.crops, self.features
            )
            self.output = self.output.view(-1)
            self.loss = self.criterion(
                self.weights_max, self.weights_org
            ) + self.criterion1(self.output, self.label)

    def get_loss(self):
        # 建议使用 .item() 来获取标量值，更安全和标准
        return self.loss.item()

    def optimize_parameters(self):
        # 根据 self.opt.amp 的值来决定是否使用 scaler
        if self.opt.amp:
            self.optimizer.zero_grad()
            self.scaler.scale(self.loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

    def get_features(self):
        self.features = self.model.get_features(self.input).to(self.device)

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, save_filename)

        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }

        # === 修改/新增：如果启用了 AMP，则保存 scaler 的状态 ===
        if self.opt.amp:
            state_dict["scaler"] = self.scaler.state_dict()

        torch.save(state_dict, save_path)
        print(f"Saved checkpoint to {save_path}")