import time
import sys
import os
import torch  # 需要显式导入 torch，否则 clip_grad_norm_ 会报错
from datetime import datetime, timezone
from validate import validate
from data import create_dataloader
from trainer.trainer import Trainer
from options.train_options import TrainOptions
from utils import set_seed


# 添加日志类，用于同时输出到控制台和文件
class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def get_val_opt(opt): # [修改] 传入 opt 参数，避免依赖全局变量导致可能的 NameError
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.data_label = "val"
    # val_opt.real_list_path = r"/3240608030/val/0_real"
    # val_opt.fake_list_path = r"/3240608030/val/1_fake"

    # 使用命令行参数控制验证集路径，如果没有提供则使用默认路径
    val_opt.real_list_path = opt.val_real_list_path
    val_opt.fake_list_path = opt.val_fake_list_path
    return val_opt

def format_options(opt, parser):
    """格式化选项信息，与BaseOptions.print_options方法保持一致"""
    message = ""
    message += "----------------- Options ---------------\n"
    for k, v in sorted(vars(opt).items()):
        comment = ""
        try:
            default = parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
        except Exception:
            pass
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "----------------- End -------------------"
    return message

if __name__ == "__main__":
    train_options = TrainOptions()
    opt = train_options.parse(print_options=False)  # 禁用自动打印选项
    set_seed(opt.seed)
    torch.backends.cudnn.benchmark = True 
    val_opt = get_val_opt(opt) # [修改] 传入 opt
    model = Trainer(opt)

    # [新增] 如果 PyTorch 版本 >= 2.0
    if int(torch.__version__.split('.')[0]) >= 2:
        print("Compiling model with torch.compile...")
        # mode 可以选 'default', 'reduce-overhead', 'max-autotune' (最慢编译，最快运行)
        model.model = torch.compile(model.model, mode='default') 

    # 创建日志目录和文件（优化：放在项目根路径下的logs文件夹）
    log_dir = os.path.join("./logs", opt.name)
    os.makedirs(log_dir, exist_ok=True)
    # 优化日志文件名格式为{实验名称}_{年月日}_{时分秒}.log
    log_file = os.path.join(log_dir, f"model_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    # 重定向标准输出到日志文件和控制台
    logger = Logger(log_file)
    sys.stdout = logger

    # 将训练选项写入日志文件
    print(format_options(opt, train_options.parser))
    print("\n")

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("\n")
    
    # 显示是否使用混合精度训练
    print(f"使用混合精度训练: {'是' if opt.use_amp else '否'}")
    print("\n")

    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    print("Length of data loader: %d" % (len(data_loader)))
    print("Length of val  loader: %d" % (len(val_loader)))

    # 初始化最佳性能跟踪变量
    best_acc = 0.0
    best_ap = 0.0
    best_auc = 0.0
    best_epoch = 0

    # 整个实验的总开始时间
    experiment_start_time = time.time()
    start_time = time.time()
    for epoch in range(opt.epoch):
        # 记录每个epoch的开始时间
        epoch_start_time = time.time()
        model.train()
        print(f"epoch: {epoch + model.step_bias}")
        
        # 应用余弦退火学习率（如果启用）
        if opt.cosine_annealing:
            # 注意：PyTorch的CosineAnnealingWarmRestarts调度器会在optimizer.step()中自动更新学习率
            # 增加epoch计数器并更新学习率调度器
            model.scheduler_epoch += 1
            model.scheduler.step(model.scheduler_epoch)
            current_lr = model.optimizer.param_groups[0]['lr']
            # 获取中国时区（UTC+8）的时间，无论服务器位于哪里
            from datetime import timedelta
            # 创建UTC+8时区
            china_tz = timezone(timedelta(hours=8))
            # 获取当前时间并转换为中国时区
            current_time = datetime.now(china_tz).strftime("%Y-%m-%d %H:%M:%S")
            print(f"当前学习率: {current_lr:.2e} | 系统时间: {current_time}")
        
        for i, (img, crops , label) in enumerate(data_loader):
            model.total_steps += 1

            model.set_input((img, crops , label))
            model.forward()
            loss = model.get_loss()

            model.optimize_parameters()

            # 修改损失打印条件，使其与参数更新同步
            # 如果不使用梯度累积，则每次迭代都打印
            # 如果使用梯度累积，则只在完成一次参数更新后打印
            should_print_loss = False
            if opt.accumulation_steps <= 1:
                # 不使用梯度累积，按照原来的频率打印
                should_print_loss = (model.total_steps % opt.loss_freq == 0)
            else:
                # 使用梯度累积，按照参数更新频率打印
                # 只有当完成一次完整的梯度累积周期时才打印
                should_print_loss = (model.update_steps > 0 and model.update_steps % (opt.loss_freq // opt.accumulation_steps) == 0 and model.accumulation_count == 0)

            if should_print_loss:
                end_time = time.time()
                elapsed_time = end_time - start_time
                # 获取并打印单独的损失值和总和
                loss_ral, loss_ce = model.get_individual_losses()
                total_loss = model.get_loss()
                if opt.accumulation_steps <= 1:
                    print(
                        "Step {:6d} | loss RAL: {:8.4f} | loss CE: {:8.4f} | Total loss: {:8.4f} | Time: {:6.2f}s".format(
                            model.total_steps, loss_ral, loss_ce, total_loss, elapsed_time
                        )
                    )
                else:
                    print(
                        "Update Step {:6d} (Total Step {:6d}) | loss RAL: {:8.4f} | loss CE: {:8.4f} | Total loss: {:8.4f} | Time: {:6.2f}s".format(
                            model.update_steps, model.total_steps, loss_ral, loss_ce, total_loss, elapsed_time
                        )
                    )
                start_time = time.time()

        # ====================================================================
        # [重要修正] Epoch 结束时的剩余梯度处理
        # 必须先 Unscale, 再 Clip, 最后 Step，防止梯度爆炸导致 NaN
        # ====================================================================
        if hasattr(model, 'accumulation_count') and model.accumulation_count > 0:
            if model.use_amp:
                # 1. Unscale (修正：必须先还原梯度)
                model.scaler.unscale_(model.optimizer)
                # 2. Clip (修正：对还原后的梯度裁剪)
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
                # 3. Step
                model.scaler.step(model.optimizer)
                model.scaler.update()
            else:
                # 修正：非 AMP 模式也需要 Clip
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
                model.optimizer.step()
            
            model.optimizer.zero_grad()
            model.accumulation_count = 0
            model.update_steps += 1  # 更新步骤数增加
            
            # 如果在epoch结束时强制更新了参数，也需要检查是否需要打印损失
            if opt.accumulation_steps > 1 and model.update_steps % (opt.loss_freq // opt.accumulation_steps) == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                loss_ral, loss_ce = model.get_individual_losses()
                total_loss = model.get_loss()
                print(
                    "Update Step {:6d} (Total Step {:6d}) | loss RAL: {:8.4f} | loss CE: {:8.4f} | Total loss: {:8.4f} | Time: {:6.2f}s".format(
                        model.update_steps, model.total_steps, loss_ral, loss_ce, total_loss, elapsed_time
                    )
                )
                start_time = time.time()

        model.eval()
        ap, fpr, fnr, acc, auc = validate(model.model, val_loader, opt.gpu_ids)
        print(
            "(Val @ epoch {}) auc: {:.4f} | ap: {:.4f} | acc: {:.4f} | fpr: {:.4f} | fnr: {:.4f}".format(
                epoch + model.step_bias, auc, ap, acc, fpr, fnr
            )
        )

        # 只在验证性能超过历史最佳时才保存模型
        # 保存准则按优先级排序：AUC > AP > ACC
        current_epoch = epoch + model.step_bias
        is_better = False
        # 优先比较 AUC
        if auc > best_auc:
            is_better = True
        elif auc == best_auc:
            # AUC 相同时比较 AP
            if ap > best_ap:
                is_better = True
            elif ap == best_ap:
                # AP 也相同时比较 ACC
                if acc > best_acc:
                    is_better = True

        if is_better:
            # 更新最佳性能指标（按 AUC/AP/ACC 的优先级）
            best_acc = acc
            best_ap = ap
            best_auc = auc
            best_epoch = current_epoch

            print(f" 发现新的最佳模型 (epoch {current_epoch}): auc={best_auc:.4f}, ap={best_ap:.4f}, acc={best_acc:.4f}")
            model.save_networks("best_model.pth")
            # 可选：同时保存带epoch编号的模型用于记录
            model.save_networks(f"model_epoch_{current_epoch}.pth")
        else:
            print(f" 当前性能未超过最佳 (最佳: auc={best_auc:.4f}, ap={best_ap:.4f}, acc={best_acc:.4f} @ epoch {best_epoch})")
        
        # 计算并打印当前epoch的总时间
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + model.step_bias} 总耗时: {epoch_time:.2f}秒")

    # 计算整个实验的总时间
    experiment_total_time = time.time() - experiment_start_time
    hours, remainder = divmod(experiment_total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 训练结束后打印最终的最佳性能和总时间
    print(f"\n 训练完成！最佳模型性能:")
    print(f"   AUC(auc): {best_auc:.4f}")
    print(f"   AP(ap): {best_ap:.4f}")
    print(f"   ACC(acc): {best_acc:.4f}")
    print(f"   所在轮次: {best_epoch}")
    print(f"   最佳模型文件: best_model.pth")
    print(f"   整个实验总耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    
    # 关闭日志文件
    logger.close()
    sys.stdout = logger.terminal  # 恢复标准输出