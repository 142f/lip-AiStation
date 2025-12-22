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
        parser.add_argument('--cosine_annealing', action='store_true', help='Use cosine annealing learning rate scheduler.')

        # ===================================================================
        # 3. 检查点与日志 (Checkpoints and Logging)
        # ===================================================================
        parser.add_argument('--loss_freq', type=int, default=100, help='Frequency of logging loss (in steps).')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='Frequency of saving checkpoints (in epochs).')
        
        # ===================================================================
        # 4. 梯度累积参数 (Gradient Accumulation Parameters)
        # ===================================================================
        parser.add_argument('--accumulation_steps', type=int, default=4, help='Number of gradient accumulation steps. Simulates larger batch sizes.')
        
        # ===================================================================
        # 5. 微调与预训练 (Finetuning and Pretraining)
        # ===================================================================
        parser.add_argument('--fine-tune', action='store_true', help='If specified, enables finetuning from a pretrained model.')
        parser.add_argument('--pretrained_model', type=str, default='./checkpoints/experiment_name/model_epoch_29.pth', help='Path to the pretrained model for finetuning.')
        
        # ===================================================================
        # 6. 混合精度训练 (Mixed Precision Training)
        # ===================================================================
        parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision (AMP) training')
        parser.add_argument('--use_ema', action='store_true', help='If specified, use EMA (Exponential Moving Average) for model weights.')
        parser.add_argument('--ema_decay', type=float, default=0.995, help='Decay rate for EMA.')
        
        self.isTrain = True
        return parser