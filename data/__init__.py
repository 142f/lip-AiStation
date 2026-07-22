import torch
import numpy as np
import os
import random
import hashlib
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler
from .datasets import AVLip


SAMPLER_PROTOCOL_VERSION = 2


def derive_seed(base_seed, epoch, stream_name):
    payload = f"lipfd:{int(base_seed)}:{int(epoch)}:{stream_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def create_data_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class EpochSeededRandomSampler(Sampler):
    def __init__(self, data_source, base_seed):
        self.data_source, self.base_seed, self.epoch = data_source, int(base_seed), 0
        self.stream_name = "sampler"

    def set_epoch(self, epoch): self.epoch = int(epoch)

    @property
    def derived_seed(self): return derive_seed(self.base_seed, self.epoch, self.stream_name)

    def materialize_indices(self):
        return torch.randperm(len(self.data_source), generator=create_data_generator(self.derived_seed)).tolist()

    def __iter__(self): return iter(self.materialize_indices())
    def __len__(self): return len(self.data_source)


class EpochSeededWeightedRandomSampler(Sampler):
    def __init__(self, weights, num_samples, base_seed, replacement=True):
        self.weights = torch.as_tensor(weights, dtype=torch.double, device="cpu")
        self.num_samples, self.base_seed = int(num_samples), int(base_seed)
        self.replacement, self.epoch = bool(replacement), 0
        self.stream_name = "weighted_sampler"

    def set_epoch(self, epoch): self.epoch = int(epoch)

    @property
    def derived_seed(self): return derive_seed(self.base_seed, self.epoch, self.stream_name)

    def materialize_indices(self):
        return torch.multinomial(
            self.weights, self.num_samples, self.replacement,
            generator=create_data_generator(self.derived_seed),
        ).tolist()

    def __iter__(self): return iter(self.materialize_indices())
    def __len__(self): return self.num_samples


def set_dataloader_epoch(data_loader, epoch):
    if hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(int(epoch))


def _sequence_hash(values, batch_size=None, drop_last=False):
    digest = hashlib.sha256()
    if batch_size is None:
        batches = [values]
    else:
        batches = [values[i:i + batch_size] for i in range(0, len(values), batch_size)]
    for batch in batches:
        if batch_size is not None and drop_last and len(batch) < batch_size:
            break
        digest.update(b"BATCH\0")
        for value in batch:
            digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def sampler_index_manifest(data_loader, epoch):
    set_dataloader_epoch(data_loader, epoch)
    sampler = data_loader.sampler
    indices = sampler.materialize_indices() if hasattr(sampler, "materialize_indices") else list(iter(sampler))
    batch_size = int(data_loader.batch_size or 1)
    return {
        "protocol_version": SAMPLER_PROTOCOL_VERSION, "epoch": int(epoch),
        "base_seed": int(getattr(sampler, "base_seed", getattr(data_loader, "repro_base_seed", 0))),
        "derived_seed": getattr(sampler, "derived_seed", None), "sampler": type(sampler).__name__,
        "num_indices": len(indices), "indices_sha256": _sequence_hash(indices),
        "batches_sha256": _sequence_hash(indices, batch_size, bool(data_loader.drop_last)),
        "first20": indices[:20], "last20": indices[-20:], "drop_last": bool(data_loader.drop_last),
        "batch_size": batch_size, "class_bal": bool(getattr(data_loader, "repro_class_bal", False)),
        "distributed": bool(getattr(data_loader, "repro_distributed", False)),
        "rank": int(getattr(sampler, "rank", 0)), "world_size": int(getattr(sampler, "num_replicas", 1)),
    }


def validate_sampling_configuration(distributed, class_bal):
    if distributed and class_bal:
        raise NotImplementedError("Distributed class-balanced sampling is not implemented safely.")


def avlip_collate_fn(batch):
    """Pack crops as one contiguous ``(S, R, B, C, H, W)`` tensor.

    Dataset 已返回打包好的 crops (S, R, C, H, W)，collate 只需沿 batch 维堆叠。
    """
    images = torch.stack([sample[0] for sample in batch])
    crops = torch.stack([sample[1] for sample in batch], dim=2)
    labels = torch.as_tensor([sample[2] for sample in batch])
    return images, crops, labels


def _loader_seed(opt) -> int:
    offsets = {"train": 0, "val": 1_000_000, "test": 2_000_000}
    return int(opt.seed) + offsets.get(str(opt.data_label), 3_000_000)


def get_bal_sampler(dataset, base_seed):
    targets = list(dataset.targets)
    if not targets:
        raise ValueError("Cannot build class-balanced sampler for an empty dataset")
    ratio = np.bincount(targets)
    if len(ratio) < 2 or np.any(ratio == 0):
        raise ValueError(f"Class-balanced sampling requires both classes, counts={ratio.tolist()}")
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = EpochSeededWeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        base_seed=base_seed,
    )
    return sampler


def create_dataloader(opt, distributed=False):
    validate_sampling_configuration(distributed, bool(opt.class_bal))
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = AVLip(opt)
    if len(dataset) == 0:
        raise ValueError(
            f"No images found for data_label={opt.data_label}: "
            f"real={opt.real_list_path}, fake={opt.fake_list_path}"
        )

    base_seed = _loader_seed(opt)
    worker_generator = create_data_generator(derive_seed(base_seed, 0, "worker"))
    sampler = (
        get_bal_sampler(dataset, base_seed=base_seed)
        if opt.class_bal
        else None
    )
    if sampler is None and shuffle and not distributed:
        sampler = EpochSeededRandomSampler(dataset, base_seed=base_seed)
        shuffle = False
    
    # 如果是分布式训练，使用DistributedSampler
    if distributed:
        # 当同时使用类别平衡采样器和分布式采样器时，优先使用类别平衡采样器
        if sampler is None:
            sampler = DistributedSampler(
                dataset,
                shuffle=shuffle,
                seed=base_seed,
            )
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
            4  # 设置上限防止过多进程
        )
        num_workers = optimal_workers
        print(f"[数据加载优化] 自动设置 num_workers={num_workers} (逻辑CPU核心数: {cpu_count}, batch_size: {opt.batch_size})")
    
    # 检查是否使用 GPU，如果使用则启用 pin_memory
    use_pin_memory = len(opt.gpu_ids) > 0 and torch.cuda.is_available()
    
    # 根据num_workers动态调整prefetch_factor
    prefetch_factor = getattr(opt, "prefetch_factor", 2) if num_workers > 0 else None
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,       # 固定内存加速 GPU 传输
        prefetch_factor=prefetch_factor, # 预加载因子，加快数据加载
        persistent_workers=(num_workers > 0),  # 保持worker进程存活
        collate_fn=avlip_collate_fn,
        worker_init_fn=seed_worker,
        generator=worker_generator,
    )
    # Public metadata used by set_dataloader_epoch() and reproducibility tests.
    data_loader.repro_worker_generator = worker_generator
    data_loader.repro_base_seed = base_seed
    data_loader.repro_class_bal = bool(opt.class_bal)
    data_loader.repro_distributed = bool(distributed)
    return data_loader
