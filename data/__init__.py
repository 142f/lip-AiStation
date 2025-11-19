import torch
import numpy as np
import os
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from .datasets import AVLip


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights)
    )
    return sampler


def create_dataloader(opt, distributed=False):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = AVLip(opt)

    sampler = get_bal_sampler(dataset) if opt.class_bal else None
    
    # 如果是分布式训练，使用DistributedSampler
    if distributed:
        # 当同时使用类别平衡采样器和分布式采样器时，优先使用类别平衡采样器
        if sampler is not None:
            print("Warning: Both class balanced sampler and distributed sampler are used. Using class balanced sampler.")
        else:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        # 分布式训练时，shuffle应该由DistributedSampler控制
        shuffle = False

    # 优化 num_workers 配置
    num_workers = int(opt.num_threads)
    if num_workers == 0:
        # 自动检测：基于CPU核心数、batch_size和GPU数量优化设置
        cpu_count = os.cpu_count() or 4
        # 根据batch_size动态调整num_workers
        # batch_size越大，每个样本处理越耗时，可以适当增加num_workers
        batch_size_factor = max(1, opt.batch_size // 10)
        # 计算最优num_workers，考虑batch_size因素和硬件配置
        optimal_workers = min(
            max(cpu_count // batch_size_factor, 2),
            cpu_count,  # 不超过逻辑CPU核心数
            16  # 设置上限防止过多进程
        )
        num_workers = optimal_workers
        print(f"[数据加载优化] 自动设置 num_workers={num_workers} (逻辑CPU核心数: {cpu_count}, batch_size: {opt.batch_size})")
    
    # 检查是否使用 GPU，如果使用则启用 pin_memory
    use_pin_memory = len(opt.gpu_ids) > 0 and torch.cuda.is_available()
    
    # 根据num_workers动态调整prefetch_factor
    prefetch_factor = 2 if num_workers > 0 else None
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,       # 固定内存加速 GPU 传输
        prefetch_factor=prefetch_factor, # 预加载因子，加快数据加载
        persistent_workers=(num_workers > 0),  # 保持worker进程存活
    )
    return data_loader