import time
from validate import validate
from data import create_dataloader
from trainer.trainer import Trainer
from options.train_options import TrainOptions


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.data_label = "val"
    # val_opt.real_list_path = r"/3240608030/val/0_real"
    # val_opt.fake_list_path = r"/3240608030/val/1_fake"

    # 使用命令行参数控制验证集路径，如果没有提供则使用默认路径
    val_opt.real_list_path = opt.val_real_list_path
    val_opt.fake_list_path = opt.val_fake_list_path
    return val_opt


if __name__ == "__main__":
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
    model = Trainer(opt)

    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    print("Length of data loader: %d" % (len(data_loader)))
    print("Length of val  loader: %d" % (len(val_loader)))

    # 初始化最佳性能跟踪变量
    best_acc = 0.0
    best_ap = 0.0
    best_epoch = 0

    start_time = time.time()
    for epoch in range(opt.epoch):
        model.train()
        print("epoch: ", epoch + model.step_bias)
        
        # 应用余弦退火学习率（如果启用）
        if opt.cosine_annealing:
            current_lr = model.cosine_annealing_lr(epoch, opt.epoch)
            if epoch % 1 == 0:  # 每1个epoch打印一次学习率
                print(f"当前学习率: {current_lr:.2e}")
        
        for i, (img, crops, label) in enumerate(data_loader):
            model.total_steps += 1

            model.set_input((img, crops, label))
            model.forward()
            loss = model.get_loss()

            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(
                    "Train loss: {}\tstep: {}\t{} steps time: {:.2f}s".format(
                        model.get_loss(), model.total_steps, opt.loss_freq, elapsed_time
                    )
                )
                start_time = time.time()

        model.eval()
        ap, fpr, fnr, acc = validate(model.model, val_loader, opt.gpu_ids)
        print(
            "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
                epoch + model.step_bias, acc, ap, fpr, fnr
            )
        )

        # 只在验证性能超过历史最佳时才保存模型
        current_epoch = epoch + model.step_bias
        if acc > best_acc or (acc == best_acc and ap > best_ap):
            # 更新最佳性能指标
            best_acc = acc
            best_ap = ap
            best_epoch = current_epoch
            
            print(f" 发现新的最佳模型 (epoch {current_epoch}): acc={acc:.4f}, ap={ap:.4f}")
            model.save_networks("best_model.pth")
            # 可选：同时保存带epoch编号的模型用于记录
            model.save_networks(f"model_epoch_{current_epoch}.pth")
        else:
            print(f" 当前性能未超过最佳 (最佳: acc={best_acc:.4f}, ap={best_ap:.4f} @ epoch {best_epoch})")

    # 训练结束后打印最终的最佳性能
    print(f"\n 训练完成！最佳模型性能:")
    print(f"   准确率 (acc): {best_acc:.4f}")
    print(f"   AP值 (ap): {best_ap:.4f}")
    print(f"   所在轮次: {best_epoch}")
    print(f"   最佳模型文件: best_model.pth")