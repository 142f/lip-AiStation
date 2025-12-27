import torch
import torch.nn as nn
from .clip import clip
import os
import open_clip
from .region_awareness import get_backbone, SELayerVec

# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class LipFD(nn.Module):
    def __init__(self, name, num_classes=1):
        super(LipFD, self).__init__()
        self.name = name

        # --- 1. 卷积下采样层 ---
        # 你说你现在不用 336px，这里仍保持你原逻辑
        if "@336px" in name:
            self.conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=3)
        else:
            self.conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=5)

        # --- 2. 加载 CLIP / DFN 模型 ---
        if name.startswith("DFN:"):
            print(f"[LipFD] Loading Apple DFN model: {name}")
            self.encoder, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14', pretrained='dfn2b', device='cpu'
            )
        else:
            clean_name = name.replace("CLIP:", "")
            print(f"[LipFD] Loading OpenAI CLIP model: {clean_name}")
            self.encoder, self.preprocess = clip.load(clean_name, device="cpu")

        # 加载 Region Awareness 骨干
        self.backbone = get_backbone(pretrained=True)

        # =================================================================
        # [核心策略集] Deepfake 检测专用优化
        # =================================================================
        if hasattr(self.encoder, 'visual') and hasattr(self.encoder.visual, 'transformer'):
            visual = self.encoder.visual

            # 获取 ViT 宽度
            self.vit_width = visual.transformer.width
            # 获取输出维度
            self.output_dim = visual.output_dim

            # --- 策略 1: BERT 式模态嵌入 (Segment Embedding) ---
            self.modality_bias = nn.Parameter(torch.zeros(2, self.vit_width))
            nn.init.normal_(self.modality_bias, std=0.02)

            # --- 策略 2: 动态门控融合 (SE Fusion) ---
            self.fusion_se = SELayerVec(self.vit_width * 2, reduction=16)

            # --- 策略 3: 残差 CLS 连接 (Residual CLS) ---
            self.cls_residual_scale = nn.Parameter(torch.tensor(0.5))

            # 融合投影层: (width*2) -> (output_dim)
            self.fusion_proj = nn.Linear(self.vit_width * 2, self.output_dim)
            nn.init.normal_(self.fusion_proj.weight, std=self.vit_width ** -0.5)

            # --- 策略 4: 跨模态注意力偏置注入 (Attention Bias Injection) ---
            # [MOD] 不再直接把 attn_mask 固死在 resblock 上（会导致 device/dtype 不一致）
            #       改为：在本模块里注册 buffer，forward 时搬到正确 device 再注入
            self._build_attention_bias(visual)

            print(f"[LipFD] ✅ 已启用全栈优化策略:")
            print(f"         1. Modality Embeddings (Audio/Video)")
            print(f"         2. Attention Bias Injection (Layer 0-2)")
            print(f"         3. Dynamic SE Fusion (FP16 Safe)")
            print(f"         4. Residual CLS Connection")

    # =========================
    # [MOD] Attention Bias 改造
    # =========================
    def _build_attention_bias(self, visual):
        """
        [技巧] 构建结构化 Attention Bias（不直接绑定到 block，避免 device/dtype 坑）
        """
        patch_size = visual.conv1.kernel_size[0]
        grid_size = visual.input_resolution // patch_size
        total_tokens = grid_size ** 2 + 1
        split_idx = (total_tokens - 1) // 2

        # Bias 矩阵 (N, N) - 先在 CPU 上建，后续 forward 时 to(device)
        attn_bias = torch.zeros(total_tokens, total_tokens)

        audio_start, audio_end = 1, 1 + split_idx
        video_start, video_end = 1 + split_idx, total_tokens

        bonus = 1.0
        attn_bias[audio_start:audio_end, video_start:video_end] = bonus
        attn_bias[video_start:video_end, audio_start:audio_end] = bonus

        # [MOD] 注册为 buffer（不参与训练，但会随 model.to(device) 迁移）
        self.register_buffer("attn_bias", attn_bias, persistent=False)
        self.inject_layers = [0, 1, 2]

    # [MOD] 每次 forward 前把 mask 注入到正确层，并保证 device/dtype 对齐
    def _apply_attention_bias(self, visual, device, dtype):
        if not hasattr(self, "attn_bias"):
            return
        bias = self.attn_bias.to(device=device, dtype=dtype)
        for i in getattr(self, "inject_layers", []):
            if i < len(visual.transformer.resblocks):
                visual.transformer.resblocks[i].attn_mask = bias

    def forward(self, x, feature):
        return self.backbone(x, feature)

    def get_features(self, x):
        """
        特征提取主流程: Manual CLIP Forward + Deepfake Optimizations

        [MOD]
        - 方案2：在本函数内关闭 autocast，强制 fp32
        - 折中：patch tokens 不经过 ln_post；cls 才经过 ln_post（更贴近 CLIP）
        - Attention bias 运行时注入，保证 device/dtype 一致
        """
        # =========================
        # [MOD] 方案2：关 autocast，全程 fp32
        # =========================
        autocast = torch.cuda.amp.autocast
        with autocast(enabled=False):
            x = self.conv1(x)  # fp32

            if hasattr(self, 'modality_bias'):
                visual = self.encoder.visual

                # 强制 fp32（避免 AMP/half 混入）
                x = x.float()

                # --- A. Patch Embedding ---
                x = visual.conv1(x)  # fp32
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1)  # (B, N, width)

                # 拼接 CLS & PosEmb（全部 fp32）
                cls_token = visual.class_embedding.float().view(1, 1, -1).expand(
                    x.shape[0], 1, x.shape[-1]
                )
                x = torch.cat([cls_token, x], dim=1)
                x = x + visual.positional_embedding.float()

                # --- B. 注入模态身份 (Segment Embedding) ---
                n_patches = x.shape[1] - 1
                split_idx = n_patches // 2

                bias_audio = self.modality_bias[0].view(1, 1, -1).float()
                bias_video = self.modality_bias[1].view(1, 1, -1).float()

                # 显式污染特征：前半 patch=Audio，后半 patch=Video
                x[:, 1:1 + split_idx, :] = x[:, 1:1 + split_idx, :] + bias_audio
                x[:, 1 + split_idx:, :]  = x[:, 1 + split_idx:, :] + bias_video

                # ln_pre（fp32）
                x = visual.ln_pre(x)

                # --- C. Transformer（含 Attention Bias Injection）---
                # [MOD] 运行时注入 mask，保证 device/dtype
                self._apply_attention_bias(visual, device=x.device, dtype=x.dtype)

                x = x.permute(1, 0, 2)  # NLD -> LND
                out = visual.transformer(x)
                # open_clip/clip 可能返回 tuple，也可能直接返回 Tensor
                if isinstance(out, tuple):
                    x = out[1]
                else:
                    x = out
                x = x.permute(1, 0, 2)  # LND -> NLD

                # --- D. 后处理（折中方案）---
                # [MOD] patch tokens 不过 ln_post；cls 才过 ln_post
                # patch tokens: 用 transformer 输出直接做 pooling（更接近 CLIP 的分布）
                patch_tokens = x[:, 1:, :]  # (B, n_patches, width)

                # cls token: 单独 ln_post（CLIP 原生做法）
                cls_token_t = x[:, 0, :]  # (B, width)
                cls_token_t = visual.ln_post(cls_token_t)

                # --- E. 提取特征（patch pooling）---
                feat_audio = patch_tokens[:, :split_idx, :].mean(dim=1)
                feat_video = patch_tokens[:, split_idx:, :].mean(dim=1)
                combined = torch.cat([feat_audio, feat_video], dim=1)  # (B, width*2)

                # --- F. 动态 SE 加权（数值稳：在 fp32 下做）---
                combined_weighted = self.fusion_se(combined)  # fp32

                # 投影到目标维度
                features = self.fusion_proj(combined_weighted)  # fp32 (B, output_dim)

                # --- G. Residual CLS（cls 走 proj，再残差）---
                if visual.proj is not None:
                    proj_cls = cls_token_t @ visual.proj.float()  # (B, output_dim)
                else:
                    proj_cls = cls_token_t

                final_features = features + self.cls_residual_scale.float() * proj_cls
                return final_features

            else:
                # 没启用策略时，走原 encoder
                return self.encoder.encode_image(x)


class RALoss(nn.Module):
    def __init__(self, margin=0.15):
        super(RALoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, alphas_max, alphas_org):
        diff = (alphas_org + self.margin) - alphas_max
        loss = self.relu(diff).mean()
        return loss
