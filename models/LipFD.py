import torch
import torch.nn as nn
from .clip import clip
import os
try:
    import open_clip
except Exception:
    open_clip = None
from .region_awareness import get_backbone, SELayerVec

# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class LipFD(nn.Module):
    def __init__(self, name, num_classes=1):
        super(LipFD, self).__init__()
        self.name = name

        # =================================================================
        # [配置控制] 统一从环境变量读取消融配置
        # 避免与 options/base_options.py 的注入逻辑不一致
        # =================================================================
        self.no_innov = (os.getenv("LIPFD_NO_INNOV", "0") == "1")
        
        # 打印消融状态，方便调试
        if self.no_innov:
            print(f"[LipFD] !!! 消融实验模式开启 (Ablation Mode) !!!")
            print(f"[LipFD] 创新模块将被物理移除，参数量应显著下降。")
            self.use_modality_bias = False
            self.use_attn_bias     = False
            self.use_se_fusion     = False
            self.use_residual_cls  = False
        else:
            print(f"[LipFD] 完整模式开启 (Full Mode)")
            self.use_modality_bias = (os.getenv("LIPFD_NO_MODALITY_BIAS", "0") != "1")
            self.use_attn_bias     = (os.getenv("LIPFD_NO_ATTN_BIAS", "0") != "1")
            self.use_se_fusion     = (os.getenv("LIPFD_NO_SE_FUSION", "0") != "1")
            self.use_residual_cls  = (os.getenv("LIPFD_NO_RESIDUAL_CLS", "0") != "1")

        # --- 1. 卷积下采样层 ---
        if "@336px" in name:
            self.conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=3)
        else:
            self.conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=5)

        # --- 2. 加载 CLIP / DFN 模型 ---
        if name.startswith("DFN:"):
            if open_clip is None:
                raise ImportError(
                    "DFN backbone requires `open_clip` but it is not available. "
                    "Please install a working open-clip package."
                )
            print(f"[LipFD] 正在加载 Apple DFN 模型: {name}")
            self.encoder, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14', pretrained='dfn2b', device='cpu'
            )
        else:
            clean_name = name.replace("CLIP:", "")
            print(f"[LipFD] 正在加载 OpenAI CLIP 模型: {clean_name}")
            self.encoder, self.preprocess = clip.load(clean_name, device="cpu")

        self.backbone = get_backbone(pretrained=True)

        # --- 3. 策略组件初始化 (核心修改：物理隔离) ---
        if hasattr(self.encoder, 'visual') and hasattr(self.encoder.visual, 'transformer'):
            visual = self.encoder.visual
            self.vit_width = visual.transformer.width
            self.output_dim = visual.output_dim

            if not self.no_innov:
                # 【正常模式】初始化所有创新模块参数
                print("[LipFD] 初始化创新模块参数...")
                
                # 策略 1: 模态偏置
                self.modality_bias = nn.Parameter(torch.zeros(2, self.vit_width))
                nn.init.normal_(self.modality_bias, std=0.02)

                # 策略 2: SE 融合
                self.fusion_se = SELayerVec(self.vit_width * 2, reduction=16)
                self.fusion_proj = nn.Linear(self.vit_width * 2, self.output_dim)
                nn.init.normal_(self.fusion_proj.weight, std=self.vit_width ** -0.5)

                # 策略 3: 残差连接
                self.cls_residual_scale = nn.Parameter(torch.tensor(0.5))

                # 策略 4: 注意力注入
                self._build_attention_bias(visual)
            else:
                # 【消融模式】不初始化参数，设为 None
                print("[LipFD] 跳过创新模块参数初始化 (节省显存/参数量)")
                self.modality_bias = None
                self.fusion_se = None
                self.fusion_proj = None
                self.cls_residual_scale = None
                # 注意：不需要 build attention bias

    def _build_attention_bias(self, visual):
        patch_size = visual.conv1.kernel_size[0]
        grid_size = visual.input_resolution // patch_size
        total_tokens = grid_size ** 2 + 1
        split_idx = (total_tokens - 1) // 2

        attn_bias = torch.zeros(total_tokens, total_tokens)
        audio_start, audio_end = 1, 1 + split_idx
        video_start, video_end = 1 + split_idx, total_tokens

        bonus = 1.0
        attn_bias[audio_start:audio_end, video_start:video_end] = bonus
        attn_bias[video_start:video_end, audio_start:audio_end] = bonus

        self.register_buffer("attn_bias", attn_bias, persistent=False)
        self.inject_layers = [0, 1, 2]

    def _apply_attention_bias(self, visual, device, dtype):
        # 只有在非消融模式且开启 attention bias 时才应用
        if not self.no_innov and self.use_attn_bias and hasattr(self, "attn_bias"):
            bias = self.attn_bias.to(device=device, dtype=dtype)
            for i in self.inject_layers:
                if i < len(visual.transformer.resblocks):
                    visual.transformer.resblocks[i].attn_mask = bias
        else:
            # 清理状态
            for i in getattr(self, "inject_layers", []):
                if i < len(visual.transformer.resblocks):
                    visual.transformer.resblocks[i].attn_mask = None

    def forward(self, x, feature):
        return self.backbone(x, feature)

    def get_features(self, x):
        # 适配新版 API：使用 torch.amp.autocast 消除警告
        device_type = 'cuda' if x.is_cuda else 'cpu'
        
        with torch.amp.autocast(device_type=device_type, enabled=False):
            # 1. 基础下采样
            x = self.conv1(x) 
            
            # 2. Baseline 判定 (消融模式直接返回)
            if self.no_innov:
                # 确保清理掉可能存在的 attention mask (虽然理论上没初始化)
                if hasattr(self.encoder, 'visual'):
                     # 传入 dummy device 即可，主要为了触发清理逻辑
                    self._apply_attention_bias(self.encoder.visual, x.device, x.dtype)
                return self.encoder.encode_image(x)

            # 3. 手动前向传播 (仅在完整模式下执行)
            # 这里的代码只有在 no_innov=False 时才会走到，保证了安全性
            visual = self.encoder.visual
            x = x.float()

            x = visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

            cls_token = visual.class_embedding.float().view(1, 1, -1).expand(x.shape[0], 1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = x + visual.positional_embedding.float()

            # 模态注入
            n_patches = x.shape[1] - 1
            split_idx = n_patches // 2
            
            # 安全检查：确保 parameter 存在
            if self.use_modality_bias and self.modality_bias is not None:
                bias_audio = self.modality_bias[0].view(1, 1, -1).float()
                bias_video = self.modality_bias[1].view(1, 1, -1).float()
                x_cls = x[:, :1, :]
                x_audio = x[:, 1:1 + split_idx, :] + bias_audio
                x_video = x[:, 1 + split_idx:, :] + bias_video
                x = torch.cat([x_cls, x_audio, x_video], dim=1)

            x = visual.ln_pre(x)

            # 注意力偏置注入
            self._apply_attention_bias(visual, device=x.device, dtype=x.dtype)

            # Transformer 前向
            x = x.permute(1, 0, 2)
            out = visual.transformer(x)
            
            # 动态解包 Transformer 输出
            if isinstance(out, torch.Tensor):
                x = out
            elif isinstance(out, tuple) or isinstance(out, list):
                x = out[0] if isinstance(out[0], torch.Tensor) else out[1]
            elif isinstance(out, dict):
                if 'last_hidden_state' in out:
                    x = out['last_hidden_state']
                elif 'x' in out:
                    x = out['x']
                else:
                    x = next(v for v in out.values() if isinstance(v, torch.Tensor))
            else:
                raise TypeError(f"无法处理的 Transformer 输出类型: {type(out)}")

            x = x.permute(1, 0, 2) # LND -> NLD

            # 后处理
            patch_tokens = x[:, 1:, :]
            cls_token_t = visual.ln_post(x[:, 0, :])

            feat_audio = patch_tokens[:, :split_idx, :].mean(dim=1)
            feat_video = patch_tokens[:, split_idx:, :].mean(dim=1)
            combined = torch.cat([feat_audio, feat_video], dim=1)

            # SE Fusion
            if self.use_se_fusion and self.fusion_se is not None:
                combined = self.fusion_se(combined)
            
            # 如果 fusion_proj 为 None (消融模式)，这里会报错，但前面已经 return 了，所以安全
            features = self.fusion_proj(combined)

            # Residual CLS
            if self.use_residual_cls and self.cls_residual_scale is not None:
                proj_cls = cls_token_t @ visual.proj.float() if visual.proj is not None else cls_token_t
                features = features + self.cls_residual_scale.float() * proj_cls

            return features

class RALoss(nn.Module):
    def __init__(self, margin=0.15):
        super(RALoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, alphas_max, alphas_org):
        diff = (alphas_org + self.margin) - alphas_max
        return self.relu(diff).mean()
