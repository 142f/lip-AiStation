import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from .datasets import AVLip


def get_bal_sampler(dataset):

     # 直接使用数据集对象上的 targets 属性（这是我们在 AVLip 中添加的）
    targets = dataset.targets

    # 确保 targets 是 numpy 数组，以便 np.bincount 正确工作
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)


    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights)
    )
    return sampler


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = AVLip(opt)

    sampler = get_bal_sampler(dataset) if opt.class_bal else None
    # 若传入 sampler，则 DataLoader 的 shuffle 必须为 False
    if sampler is not None:
        shuffle = False
        print("使用类别均衡采样器，已将 shuffle 设为 False。")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,   # 使用计算好的 shuffle（而不是硬编码 True）
        sampler=sampler,
        num_workers=int(opt.num_threads),
    )
    print("DataLoader 创建完成：batch_size={}, shuffle={}, sampler={}".format(
        opt.batch_size, shuffle, "有" if sampler is not None else "无"
    ))
    return data_loader
