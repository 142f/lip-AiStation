from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """
    This class includes training options that are unique to the training process.
    It inherits from BaseOptions.
    """

    def initialize(self, parser):
        # First, initialize the parent class options (data, model, environment, etc.)
        parser = BaseOptions.initialize(self, parser)

        # ===================================================================
        # 1. 训练周期与数据划分 (Epochs and Data Splits)
        # ===================================================================
        parser.add_argument('--epoch', type=int, default=100, help='Total number of training epochs.')
        parser.add_argument('--train_split', type=str, default='train', help='String identifier for the training data split.')
        parser.add_argument('--val_split', type=str, default='val', help='String identifier for the validation data split.')

        # ===================================================================
        # 2. 优化器与学习率 (Optimizer and Learning Rate)
        # ===================================================================
        parser.add_argument('--optim', type=str, default='adamw', help='Optimizer to use [sgd, adam, adamw].')
        parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
        parser.add_argument('--beta1', type=float, default=0.9, help='Momentum term for the Adam optimizer.')
        parser.add_argument('--beta2', type=float, default=0.999, help='Second-moment decay for Adam/AdamW.')
        parser.add_argument('--cosine_annealing', action='store_true', help='Use cosine annealing learning rate scheduler.')
        
        # [新增] 预热参数 Warmup
        parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs before cosine annealing.')

        # [新增] 余弦退火（不重启）下的最小学习率
        # 建议不要太低（例如 1e-8/1e-7），否则后期几乎不更新。
        parser.add_argument('--eta_min', type=float, default=1e-6, help='Minimum learning rate for cosine annealing (no restart).')

        # ===================================================================
        # 3. 检查点与日志 (Checkpoints and Logging)
        # ===================================================================
        parser.add_argument('--loss_freq', type=int, default=100, help='Frequency of logging loss (in steps).')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='[已废弃] 请使用 --milestone_save_freq 替代。')
        parser.add_argument('--resume_save_freq', type=int, default=10, help='断点续训文件保存频率（epoch）。每N轮保存一次完整Resume状态。')
        parser.add_argument('--milestone_save_freq', type=int, default=20, help='里程碑快照保存频率（epoch）。每N轮保存一次轻量推理权重。')
        parser.add_argument('--keep_milestone_checkpoints', type=int, default=2, help='磁盘上最多保留的里程碑快照数量。')
        
        # ===================================================================
        # 4. 梯度累积参数 (Gradient Accumulation Parameters)
        # ===================================================================
        parser.add_argument('--accumulation_steps', type=int, default=4, help='Number of gradient accumulation steps. Simulates larger batch sizes.')
        parser.add_argument(
            '--region_checkpoint_chunk_size', type=int, default=0,
            help='Region activation-checkpoint chunk: 0 off, positive explicit.'
        )
        parser.add_argument('--region_local_forward_chunk_size', type=int, default=0,
                            help='Region local-backbone forward chunk: 0 off.')
        parser.add_argument('--region_weight_chunk_size', type=int, default=0,
                            help='Region weight-head chunk: 0 keeps the strict baseline path.')
        
        # ===================================================================
        # 5. 微调与预训练 (Finetuning and Pretraining)
        # ===================================================================
        parser.add_argument('--fine-tune', action='store_true', help='If specified, enables finetuning from a pretrained model.')
        parser.add_argument('--pretrained_model', type=str, default='./checkpoints/experiment_name/model_epoch_29.pth', help='Path to the pretrained model for finetuning.')
        parser.add_argument('--resume', action='store_true', help='Resume full training state from --pretrained_model.')
        parser.add_argument('--allow_data_path_mismatch', action='store_true', help='Allow resume when only dataset root paths differ; use only after confirming the data is identical.')
        parser.add_argument('--allow_incomplete_resume', action='store_true', help='Allow approximate resume from a legacy checkpoint missing full training state.')
        parser.add_argument('--allow_partial_load', action='store_true', help='Explicitly allow partial model loading for intentional cross-structure fine-tuning.')
        
        # ===================================================================
        # 6. 混合精度训练 (Mixed Precision Training)
        # ===================================================================
        parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision (AMP) training')
        parser.add_argument('--amp_dtype', choices=['float16', 'bfloat16'], default='float16',
                            help='Autocast dtype when AMP is enabled.')
        parser.add_argument('--use_ema', action='store_true', help='If specified, use EMA (Exponential Moving Average) for model weights.')
        parser.add_argument('--ema_decay', type=float, default=0.995, help='Decay rate for EMA.')
        parser.add_argument('--max_consecutive_amp_skips', type=int, default=8, help='Abort after this many consecutive AMP overflow skips.')
        parser.add_argument('--max_consecutive_nonamp_skips', type=int, default=3, help='Abort after this many consecutive non-AMP non-finite-gradient skips.')
        
        # ===================================================================
        # 7. 性能分析 (Profiling)
        # ===================================================================
        parser.add_argument('--profile', action='store_true', help='Run torch.profiler to diagnose performance bottlenecks.')
        parser.add_argument('--compile', action='store_true', help='Explicitly enable torch.compile (default: eager).')
        parser.add_argument('--compile_mode', choices=['default', 'reduce-overhead'], default='default')
        parser.add_argument('--no_compile', action='store_true', help='Deprecated compatibility flag; eager is already the default.')
        parser.add_argument('--allow_tf32', action='store_true',
                            help='Allow CUDA TF32 matmul/cuDNN kernels (result-sensitive).')

        # Loss/gradient values are explicit so resume can reject objective drift.
        parser.add_argument('--ra_margin', type=float, default=0.15)
        parser.add_argument('--ra_loss_weight', type=float, default=0.01)
        parser.add_argument('--ce_loss_weight', type=float, default=1.0)
        parser.add_argument('--label_smoothing', type=float, default=0.1)
        parser.add_argument('--grad_clip_norm', type=float, default=1.0)
        
        self.isTrain = True
        return parser
