import os
import argparse
import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):      
        # ===================================================================
        # 1. 数据与路径参数 (Data and Path Arguments)
        # ===================================================================
        parser.add_argument("--real_list_path", type=str, default=r"E:\data\0_real", help="Path to the list of real images for training.")
        parser.add_argument("--fake_list_path", type=str, default=r"E:\data\1_fake", help="Path to the list of fake images for training.")
        parser.add_argument("--val_real_list_path", type=str, default=r"/3240608030/val/0_real", help="Path to the list of real images for validation.")
        parser.add_argument("--val_fake_list_path", type=str, default=r"/3240608030/val/1_fake", help="Path to the list of fake images for validation.")
        parser.add_argument("--data_label", type=str, default="train", help="Label to decide whether it's a train or validation dataset.")

        # ===================================================================
        # 2. 模型与训练策略参数 (Model and Training Strategy Arguments)
        # ===================================================================
        parser.add_argument("--arch", type=str, default="DFN:ViT-L/14", help="Model architecture. See models/__init__.py for options.")
        parser.add_argument("--batch_size", type=int, default=8, help="Input batch size for training.")
        parser.add_argument("--fix_backbone", action="store_true", help="If specified, freezes the backbone weights during training.")
        parser.add_argument("--fix_encoder", action="store_true",help="If specified, freezes the encoder weights during training.")
        parser.add_argument("--serial_batches", action="store_true", help="If true, takes images in order to make batches, otherwise takes them randomly.")
        
        # ===================================================================
        # 3. 模型加载与保存参数 (Model Loading and Saving Arguments)
        # ===================================================================
        parser.add_argument("--name", type=str, default="experiment_name", help="Name of the experiment. Determines where to store samples and models.")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Directory where models are saved.")

        # ===================================================================
        # 4. 硬件与环境参数 (Hardware and Environment Arguments)
        # ===================================================================
        parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
        parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs to use. E.g., '0', '0,1,2', or '-1' for CPU.")
        parser.add_argument("--num_threads", type=int, default=0, help="Number of threads for data loading.")

     
        # 新增参数 --错误点
        parser.add_argument("--suffix", type=str, default="", help="customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}")
        parser.add_argument('--rz_interp', type=str, default='bilinear', help='interpolation method for resizing')
        parser.add_argument('--blur_sig', type=str, default='0.0,3.0', help='sigma for Gaussian blur, comma-separated')
        parser.add_argument('--jpg_method', type=str, default='cv2', help='jpeg compression method, e.g., cv2 or pil')
        parser.add_argument('--jpg_qual', type=str, default='75,100', help='jpeg quality range, comma-separated')
        parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for optimizers')
        parser.add_argument('--class_bal', action='store_true', help='if specified, use class balancing for the dataset loader')

        # ===================================================================
        # 5. 数据增强参数 (Data Augmentation)
        # ===================================================================
        parser.add_argument('--use_aug', action='store_true', help='Enable training-time data augmentation (recommended).')
        # 对“上半频谱”做 SpecAugment 风格遮挡（不做时间平移，避免破坏音画对齐）
        parser.add_argument('--spec_aug', action='store_true', help='Apply SpecAugment-style masks to the spectrogram (top half) during training.')
        parser.add_argument('--spec_num_masks', type=int, default=2, help='Number of spec masks to apply when --spec_aug is enabled.')
        parser.add_argument('--spec_time_mask', type=float, default=0.12, help='Max fraction of width to mask (time-axis) in spectrogram region.')
        parser.add_argument('--spec_freq_mask', type=float, default=0.18, help='Max fraction of height to mask (freq-axis) in spectrogram region.')
        # 轻度分辨率退化（模拟转码/低清），会 resize 回原大小
        parser.add_argument('--rz_scale', type=str, default='0.7,1.0', help='Random resize scale range (min,max) for degradation, e.g., 0.7,1.0')

        # =========================
        # [关键] 消融实验开关
        # =========================
        # 总开关：开启后强制回退到 Baseline (只保留 conv1 + 原生 CLIP)
        parser.add_argument("--no_innov", action="store_true", help="[Ablation] Master switch: Baseline Mode")
        # 分项开关
        parser.add_argument("--no_modality_bias", action="store_true", help="[Ablation] Disable BERT Modality Embedding")
        parser.add_argument("--no_attn_bias",     action="store_true", help="[Ablation] Disable Attention Bias")
        parser.add_argument("--no_se_fusion",     action="store_true", help="[Ablation] Disable SE Fusion")
        parser.add_argument("--no_residual_cls",  action="store_true", help="[Ablation] Disable Residual CLS")

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        # get the basic options 重大问题
    #     parser.parse_args(): 会处理所有已知的参数。对于那些在命令行中未提供的参数，它会使用default值来创建对应的属性。所以，即使你不提供 --suffix，opt.suffix 也会存在（值为它的默认值，比如 ''）。
    # parser.parse_known_args(): 只会为那些在命令行中实际出现的参数创建属性。如果命令行中没有任何参数，它会返回一个几乎为空的Namespace对象，上面没有任何你自定义的属性。
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        
        # Print Env Var for verification
        no_innov = os.environ.get("LIPFD_NO_INNOV", "Unknown")
        message += f"{'[Env] NO_INNOV':>25}: {no_innov}\n"

        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        # util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self, print_options=False):
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # =================================================================
        # [修复方案] 将参数注入环境变量，确保 LipFD 能读到
        # =================================================================
        os.environ["LIPFD_NO_INNOV"]       = "1" if opt.no_innov else "0"
        os.environ["LIPFD_NO_MODALITY_BIAS"] = "1" if opt.no_modality_bias else "0"
        os.environ["LIPFD_NO_ATTN_BIAS"]     = "1" if opt.no_attn_bias else "0"
        os.environ["LIPFD_NO_SE_FUSION"]     = "1" if opt.no_se_fusion else "0"
        os.environ["LIPFD_NO_RESIDUAL_CLS"]  = "1" if opt.no_residual_cls else "0"

        # additional
        # opt.classes = opt.classes.split(',')
        opt.rz_interp = opt.rz_interp.split(",")
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(",")]
        opt.jpg_method = opt.jpg_method.split(",")
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(",")]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt