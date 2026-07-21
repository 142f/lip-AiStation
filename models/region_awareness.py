import torch
from torch import Tensor
import torch.nn as nn
import os
from typing import Type, Any, Callable, Union, List, Optional, Sequence, Tuple
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from .offline_paths import torch_checkpoint_dir

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def _group_norm_32(channels: int) -> nn.GroupNorm:
    if channels % 32 != 0:
        raise ValueError(f"GroupNorm channels must be divisible by 32, got {channels}")
    return nn.GroupNorm(32, channels)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ---------------------------
# 新增：针对展平特征的通道注意力（向量版 SE）
# 说明：
#  - 原始 SE 是对 4D feature map 做全局池化再两个 FC；这里我们对 (N, C) 向量做同样思想的通道重标定，
#    便于在不修改 backbone 的情况下把 SE 思想加到 feat_cat（拼接后向量）上。
# ---------------------------
class SELayerVec(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super(SELayerVec, self).__init__()
        hidden = max(1, channel // reduction)
        # 中文注释：通道压缩 -> ReLU -> 通道恢复 -> Sigmoid，输出每个通道的缩放系数（每个样本独立）
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C)
        # 输出与 x 相同形状，按通道放缩
        scale = self.fc(x)  # (N, C)
        return x * scale


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        ## 融合了局部细节特征与全局语义特征
        # [修改] 去掉 Sigmoid，直接输出 logits，配合后续 softmax 使用
        self.get_weight = nn.Linear(512 * block.expansion + 768, 1)
        self.fc = nn.Linear(512 * block.expansion + 768, num_classes)

        # ---------------------------
        # [关键修改] 自动从环境变量读取消融配置
        # ---------------------------
        # 默认值为 "0" (代表启用)，如果环境变量设为 "1" 则代表禁用
        self.with_pe = os.getenv("REGION_NO_PE", "0") != "1"
        self.with_se = os.getenv("REGION_NO_SE", "0") != "1"

        if not self.with_pe or not self.with_se:
            print(f"[RegionAwareness] 消融实验配置生效: PE={self.with_pe}, SE={self.with_se}")

        # ---------------------------
        # 新增：位置编码与 SE 向量模块的初始化
        # 说明：
        #  - 默认假设 num_scales = 3, num_regions = 5（与数据预处理一致）
        #  - positional_encoding 的维度与 feat_cat 的维度一致（512*expansion + 768）
        #  - se_layer 在 feat_flat 上做通道重标定
        # ---------------------------
        self.num_scales = 3
        self.num_regions = 5
        # Runtime-only memory policy. Trainer freezes these values in its
        # training_config because they deliberately do not enter state_dict.
        self.checkpoint_chunk_size = 0
        self.local_forward_chunk_size = 0
        self.weight_chunk_size = 0
        self._reported_chunk_paths = set()
        feat_dim = 512 * block.expansion + 768
        
        # [控制] 始终初始化位置编码与门控系数（保证模型结构一致，避免加载权重报错）
        # 即使 with_pe=False，这些参数也会存在（但不参与计算/更新）
        self.positional_encoding = nn.Parameter(
            torch.empty(self.num_scales * self.num_regions, feat_dim)
        )
        nn.init.normal_(self.positional_encoding, std=0.02)
        self.pe_alpha = nn.Parameter(torch.tensor(0.0))
        if not self.with_pe:
            self.positional_encoding.requires_grad_(False)
            self.pe_alpha.requires_grad_(False)
        
        # [控制] 始终初始化 SE 模块（保证模型结构一致）
        self.se_layer = SELayerVec(feat_dim, reduction=16)
        if not self.with_se:
            for p in self.se_layer.parameters():
                p.requires_grad_(False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def _extract_local_features(self, images):
        f = self.conv1(images)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.avgpool(f)
        return torch.flatten(f, 1)

    def _pack_crops(self, x) -> Tuple[Tensor, int, int, int]:
        if torch.is_tensor(x):
            if x.ndim != 6:
                raise ValueError(
                    "Packed crops must have shape "
                    "(scales, regions, batch, channels, height, width), "
                    f"got {tuple(x.shape)}"
                )
            num_scales, num_regions, batch_size = map(int, x.shape[:3])
            if min(num_scales, num_regions, batch_size) <= 0:
                raise ValueError(f"Packed crop dimensions must be non-zero: {tuple(x.shape)}")
            if x.shape[3] != 3:
                raise ValueError(f"Crop channels must be 3, got {x.shape[3]}")
            return x.reshape(-1, *x.shape[3:]), num_scales, num_regions, batch_size

        if not isinstance(x, Sequence) or len(x) == 0:
            raise TypeError("crops must be a 6D Tensor or a non-empty [scale][region] sequence")
        num_scales = len(x)
        if not isinstance(x[0], Sequence) or len(x[0]) == 0:
            raise ValueError("Each crop scale must contain at least one region")
        num_regions = len(x[0])
        flat = []
        reference_shape = None
        reference_device = None
        reference_dtype = None
        for scale_index, scale in enumerate(x):
            if not isinstance(scale, Sequence) or len(scale) != num_regions:
                raise ValueError(
                    f"Crop grid must be rectangular: scale 0 has {num_regions} regions, "
                    f"scale {scale_index} has {len(scale) if isinstance(scale, Sequence) else 'invalid'}"
                )
            for region_index, crop in enumerate(scale):
                if not torch.is_tensor(crop) or crop.ndim != 4:
                    raise TypeError(
                        f"crops[{scale_index}][{region_index}] must be a 4D Tensor"
                    )
                if crop.shape[1] != 3:
                    raise ValueError(
                        f"crops[{scale_index}][{region_index}] channels must be 3, "
                        f"got {crop.shape[1]}"
                    )
                current_shape = tuple(crop.shape)
                if reference_shape is None:
                    reference_shape = current_shape
                    reference_device = crop.device
                    reference_dtype = crop.dtype
                elif current_shape != reference_shape:
                    raise ValueError(
                        f"All crop batches must share shape {reference_shape}; "
                        f"crops[{scale_index}][{region_index}] is {current_shape}"
                    )
                elif crop.device != reference_device or crop.dtype != reference_dtype:
                    raise ValueError("All crop tensors must share device and dtype")
                flat.append(crop)
        batch_size = int(reference_shape[0])
        if batch_size <= 0:
            raise ValueError("Crop batch size must be positive")
        return torch.cat(flat, dim=0), num_scales, num_regions, batch_size

    def _validate_feature_input(self, feature, all_images, batch_size):
        if not torch.is_tensor(feature) or feature.ndim != 2:
            raise TypeError("feature must be a 2D Tensor shaped (batch, 768)")
        if feature.shape != (batch_size, 768):
            raise ValueError(
                f"feature must have shape ({batch_size}, 768), got {tuple(feature.shape)}"
            )
        if feature.device != all_images.device:
            raise ValueError(
                f"feature and crops must share a device: {feature.device} != {all_images.device}"
            )
        if not feature.is_floating_point() or not all_images.is_floating_point():
            raise TypeError("feature and crops must use floating-point dtypes")
        # Do not require identical dtypes: torch.cat intentionally preserves
        # the legacy promotion behavior (for example FP16 feature + FP32 crops).

    def _run_local_backbone(self, all_images):
        checkpoint_chunk = int(self.checkpoint_chunk_size)
        use_checkpoint = (
            checkpoint_chunk > 0
            and self.training
            and torch.is_grad_enabled()
            and all_images.shape[0] > checkpoint_chunk
        )
        if use_checkpoint:
            if "checkpoint" not in self._reported_chunk_paths:
                print(
                    "[RegionAwareness] activation checkpoint chunk triggered: "
                    f"total={all_images.shape[0]}, chunk={checkpoint_chunk}"
                )
                self._reported_chunk_paths.add("checkpoint")
            return torch.cat(
                [
                    activation_checkpoint(
                        self._extract_local_features,
                        image_chunk,
                        use_reentrant=False,
                    )
                    for image_chunk in torch.split(all_images, checkpoint_chunk, dim=0)
                ],
                dim=0,
            )
        local_chunk = int(self.local_forward_chunk_size)
        if local_chunk > 0 and all_images.shape[0] > local_chunk:
            if "local_forward" not in self._reported_chunk_paths:
                print(
                    "[RegionAwareness] local forward chunk triggered: "
                    f"total={all_images.shape[0]}, chunk={local_chunk}"
                )
                self._reported_chunk_paths.add("local_forward")
            return torch.cat(
                [
                    self._extract_local_features(image_chunk)
                    for image_chunk in torch.split(all_images, local_chunk, dim=0)
                ],
                dim=0,
            )
        return self._extract_local_features(all_images)

    def _forward_impl(self, x, feature, chunk_size: Optional[int] = None):
        """
        高效版前向传播（可直接替换原 _forward_impl）：
          - 一次性批量处理所有尺度与区域，充分利用 GPU 并行；
          - 通过 expand 减少全局特征重复内存；
          - 支持分块计算 get_weight，防止显存溢出；
          - 返回值与原函数保持一致 (pred_score, weights_max, weights_org)。

        参数:
            x: List[List[Tensor]]，形状为 [num_scales][num_regions][B, 3, H, W]
            feature: Tensor，全局特征 (B, feat_g)
            chunk_size: 可选的 get_weight 分块覆盖值；None 使用显式配置的
                weight_chunk_size（严格基线默认 0，即关闭）。

        返回:
            pred_score: Tensor，分类得分 (B, num_classes)
            weights_max: Tensor，每个区域在所有尺度下的最大权重
            weights_org: Tensor，每个区域在第一个尺度下的权重
        """

        # ---------------------------
        # 基础维度信息
        # ---------------------------
        all_images, num_scales, num_regions, batch_size = self._pack_crops(x)
        self._validate_feature_input(feature, all_images, batch_size)

        # ---------------------------
        # Step 2: 一次性通过 backbone 提取局部特征
        # ---------------------------
        f = self._run_local_backbone(all_images)

        # ---------------------------
        # Step 3: reshape 回原结构，方便后续处理
        # ---------------------------
        f = f.view(num_scales, num_regions, batch_size, -1)  # (S, R, B, feat_local)

        # ---------------------------
        # Step 4: 拼接全局特征（使用 expand 减少显存复制）
        # ---------------------------
        feat_g = feature.unsqueeze(0).unsqueeze(0).expand(num_scales, num_regions, batch_size, feature.shape[1])
        feat_cat = torch.cat([f, feat_g], dim=3)  # (S, R, B, feat_cat)
        feat_dim = feat_cat.shape[-1]

        # ---------------------------
        # [修改] Step 4.5 — 条件性添加位置编码
        # ---------------------------
        if self.with_pe and self.positional_encoding is not None:
            capacity = int(self.positional_encoding.shape[0])
            requested = int(num_scales * num_regions)
            if requested > capacity:
                raise ValueError(
                    "Positional encoding capacity exceeded while PE is enabled: "
                    f"{num_scales}x{num_regions}={requested}>{capacity}"
                )
            # 只有开启且参数存在时才执行
            pos_enc = self.positional_encoding[:num_scales * num_regions]
            pos_enc = pos_enc.view(num_scales, num_regions, 1, feat_dim).expand(-1, -1, batch_size, -1)
            # ✅ 原来：feat_cat = feat_cat + pos_enc   （强行加进去，可能伤模型）
            # ✅ 现在：默认 alpha=0 -> 等价于不加；只有有用时才学出来
            feat_cat = feat_cat + self.pe_alpha * pos_enc

        # ---------------------------
        # [修改] Step 4.6 — 条件性使用向量版 SE
        # ---------------------------
        if self.with_se:
            # 只有开启时才执行
            feat_cat_flat = feat_cat.contiguous().view(-1, feat_dim)
            feat_cat_flat = self.se_layer(feat_cat_flat)
            feat_cat = feat_cat_flat.view(num_scales, num_regions, batch_size, feat_dim)

        # ---------------------------
        # Step 5: 批量计算权重（支持分块防止显存溢出）
        # ---------------------------
        feat_cat_flat = feat_cat.contiguous().view(-1, feat_cat.shape[-1])  # (S*R*B, feat_cat)
        effective_chunk_size = (
            int(self.weight_chunk_size) if chunk_size is None else int(chunk_size)
        )
        if effective_chunk_size > 0 and feat_cat_flat.shape[0] > effective_chunk_size:
            if "weight" not in self._reported_chunk_paths:
                print(
                    "[RegionAwareness] weight-head chunk triggered: "
                    f"total={feat_cat_flat.shape[0]}, chunk={effective_chunk_size}"
                )
                self._reported_chunk_paths.add("weight")
            # 分块计算
            weights_list = []
            for chunk in torch.split(feat_cat_flat, effective_chunk_size, dim=0):
                w_chunk = self.get_weight(chunk)
                weights_list.append(w_chunk)
            weights_all = torch.cat(weights_list, dim=0)
        else:
            # 一次性计算
            weights_all = self.get_weight(feat_cat_flat)

        # ---------------------------
        # Step 6: reshape 权重回原结构
        # ---------------------------
        weights = weights_all.view(num_scales, num_regions, batch_size, -1)  # (S, R, B, weight_dim)
        # 假设 get_weight 输出 (N,1)，则压缩最后一维
        weights_scalar = weights.squeeze(-1)  # (S, R, B)

        # ---------------------------
        # Step 7: 对每个区域在尺度维做 softmax（用于尺度注意力）
        # ---------------------------
        weights_soft = torch.softmax(weights_scalar, dim=0)  # 在尺度维 S 上归一化

        # ---------------------------
        # Step 8: 加权融合局部与全局特征
        # ---------------------------
        # 广播相乘并在尺度维求和
        fused_parts = (feat_cat * weights_soft.unsqueeze(-1)).sum(dim=0)  # (R, B, C)
        # 防止数值问题，可再除以权重和
        weights_sum = weights_soft.sum(dim=0) + 1e-8  # (R, B)
        fused_parts = fused_parts / weights_sum.unsqueeze(-1)  # (R, B, C)

        # ---------------------------
        # Step 9: 区域聚合（平均所有区域特征）
        # ---------------------------
        out = fused_parts.sum(dim=0).div(fused_parts.shape[0])  # (B, C)

        # ---------------------------
        # Step 10: 分类层
        # ---------------------------
        pred_score = self.fc(out)  # (B, num_classes)

        # ---------------------------
        # Step 11: 计算附加权重输出（保持接口一致）
        # ---------------------------
        weights_max = weights_soft.max(dim=0)[0]  # (R, B)，每区域在不同尺度下的最大权重
        weights_org = weights_soft[0]             # (R, B)，第一个尺度的权重

        # ---------------------------
        # Step 12: 返回三元组（保持兼容）
        # ---------------------------
        return pred_score, weights_max, weights_org

    def forward(self, x, feature):
        return self._forward_impl(x, feature)


def _get_backbone(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, num_classes=2, **kwargs)

    if pretrained:
        model_dir = torch_checkpoint_dir()
        os.makedirs(model_dir, exist_ok=True)
        state_dict = load_state_dict_from_url(
            model_urls[arch],
            model_dir=model_dir,
            progress=progress,
        )

        # ==========================================================
        # ✅ 最稳加载：只加载 key 存在且 shape 完全一致的参数
        #    - 自动跳过 fc.*（1000类 vs 2类/2816维）
        #    - 未来改结构也不会再因为 shape mismatch 崩溃
        # ==========================================================
        model_dict = model.state_dict()
        filtered = {}

        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered[k] = v

        # 用过滤后的权重更新当前模型参数
        model_dict.update(filtered)
        model.load_state_dict(model_dict, strict=False)

        skipped = len(state_dict) - len(filtered)
        print(f"[Backbone] Loaded pretrained params: {len(filtered)}/{len(state_dict)} "
              f"(skipped {skipped} mismatched keys)")

    return model



def get_backbone(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # 使用 GroupNorm 替代 BatchNorm，解决小 Batch Size 问题
    return _get_backbone('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, norm_layer=_group_norm_32, **kwargs)


if __name__ == '__main__':
    model = get_backbone(pretrained=False).eval()
    data = [[torch.rand((1, 3, 32, 32)) for _ in range(5)] for _ in range(3)]
    feature = torch.rand((1, 768))
    with torch.inference_mode():
        pred_score, weights_max, weights_org = model(data, feature)
    print(pred_score.shape, weights_max.shape, weights_org.shape)
