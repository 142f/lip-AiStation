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

        self.criterion = get_loss().to(self.device)
        self.criterion1 = nn.CrossEntropyLoss()

        self.model.to(opt.gpu_ids[0] if torch.cuda.is_available() else "cpu")

    def adjust_learning_rate(self, min_lr=1e-8):
        for param_group in self.optimizer.param_groups:
            if param_group["lr"] < min_lr:
                return False
            param_group["lr"] /= 10.0
        return True

    def cosine_annealing_lr(self, epoch, total_epochs, min_lr=1e-10):
        """
        简单的余弦退火学习率调度
        lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * epoch / total_epochs))
        """
        if not hasattr(self, 'initial_lr'):
            self.initial_lr = self.optimizer.param_groups[0]['lr']
        
        # 余弦退火公式
        lr = min_lr + 0.5 * (self.initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))
        
        # 更新所有参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

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
        self.loss = self.loss_ral + self.loss_ce

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
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

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

        # serialize model and optimizer to dict
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }

        torch.save(state_dict, save_path)