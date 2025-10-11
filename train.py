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

    for epoch in range(opt.epoch):
        model.train()
        print("epoch: ", epoch + model.step_bias)
        for i, (img, crops, label) in enumerate(data_loader):
            model.total_steps += 1

            model.set_input((img, crops, label))
            model.forward()
            loss = model.get_loss()

            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(
                    "Train loss: {}\tstep: {}".format(
                        model.get_loss(), model.total_steps
                    )
                )

        if epoch % opt.save_epoch_freq == 0:
            print("saving the model at the end of epoch %d" % (epoch + model.step_bias))
            model.save_networks("model_epoch_%s.pth" % (epoch + model.step_bias))

        model.eval()
        ap, fpr, fnr, acc = validate(model.model, val_loader, opt.gpu_ids)
        print(
            "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
                epoch + model.step_bias, acc, ap, fpr, fnr
            )
        )