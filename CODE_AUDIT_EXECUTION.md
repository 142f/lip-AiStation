# LipFD 代码审计整改执行记录

更新时间：2026-07-20  
适用分支：`main2-cx`

## 训练准入结论

当前代码已完成 P0 兼容性修复和 CPU 自动回归，可进入固定配置的
100–500 update 短程双跑；尚未完成真实 DFN、真实训练 checkpoint、CUDA
FP32 对照和多种子短程训练，因此仍不允许直接启动正式完整训练。

所有性能收益均为理论预期。当前没有目标 GPU 上的吞吐、显存或训练时间
数据，不声明具体加速比例。

## 已完成修改

### 模型兼容性

- PE 容量检查仅在 `with_pe=True` 时生效。PE-off 的 4×5 网格支持前向和反向；
  PE-on 超过 15 个位置会在模型入口给出明确异常。
- 保留 feature/crops 的浮点类型、设备、batch 和维度检查，但不要求 dtype
  完全一致，恢复 FP16/BF16 feature 与 FP32 crops 的旧版类型提升行为。
- Transformer 输出仅接受明确契约：Tensor、包含唯一顶层 Tensor 的
  tuple/list，或 `last_hidden_state`/`x` 字段；同时校验并规范 LND/NLD 布局。
  不再递归猜测嵌套结构中的第一个 Tensor。
- attention mask 增加 token shape、device、dtype 和 resblock 契约检查；保留
  eager identity cache，并通过 `--disable_attn_mask_cache` 显式控制和记录。
- `LipFD` 明确只支持二分类，`num_classes=2` 正常，其他值立即失败。
- `RALoss` 拒绝形状广播，避免静默改变损失语义。

### 性能开关与配置冻结

- `region_checkpoint_chunk_size`、`region_local_forward_chunk_size` 和
  `region_weight_chunk_size` 均成为显式 CLI 配置，严格基线默认全部为 0。
- chunk 首次真正触发时打印总样本数和 chunk size，避免“配置了但未触发”或
  “已触发但日志不可见”。
- checkpoint 的 `training_config` 现记录：
  - LR、Adam betas、weight decay；
  - RALoss margin、RAL/CE 权重、label smoothing、梯度裁剪；
  - 全部 LipFD/Region 消融开关和 attention-mask cache；
  - seed、CuDNN deterministic/benchmark、TF32；
  - AMP dtype、compile mode 和三个 chunk；
  - Git commit、branch、dirty 状态、tracked diff 和 untracked content 哈希；
  - 训练/验证数据目录的成员、大小和 mtime 清单哈希。
- resume 沿用严格配置比较；上述结果敏感配置不一致时拒绝恢复。
- checkpoint 额外保存 Python/PyTorch/CUDA/cuDNN/GPU 环境清单，以及加载来源
  checkpoint 的 SHA-256。

说明：数据目录默认保存的是成员和元数据哈希，不会在每次启动时重新读取全部
大型图片内容。若数据可能在保持文件名、大小和 mtime 不变的情况下被替换，正式
实验仍应使用数据发布版本或离线全内容 manifest 作为第二层校验。

### 当前项目状态的更正

以下项目在当前分支已经完成，不再列为待优化项：

- Dataset/`collate_fn` 已输出 `(S,R,B,C,H,W)` packed crops，正常路径单次 H2D；
- transforms 已在 Dataset 构造阶段复用；
- AMP 已使用 `torch.amp`；
- EMA 仅在 optimizer 确实 step 后更新；
- `Trainer.train(mode)`/`eval()` 已调用父类状态管理；
- compile 为显式 `--compile`，默认 eager；
- optimizer 清零已集中到更新完成/异常处理路径；
- loss 日志读取已使用 `.item()`；`Trainer.test()` 已改为
  `torch.inference_mode()`。

## 自动化测试范围

默认 CPU pytest 覆盖：

- 3×5 完整 BasicBlock/GN/PE/SE/权重头/分类头前向与反向，和旧版公式严格对照；
- PE on/off 的 1×1、2×3、3×5、4×5；
- FP16/BF16 feature + FP32 crops；
- 3×5 list/packed 前向与反向；
- checkpoint/local/weight 三种分块路径单独触发；
- 空输入、非矩形网格、错误通道和错误 feature 维度；
- Tensor、tuple、dict-LND、dict-NLD Transformer 返回契约及梯度；
- 未知/歧义 Transformer 返回结构明确失败；
- attention mask 连续前向、shape、dtype 和 cache identity；
- 二分类参数和 RALoss 形状契约。

资源型测试默认 skip，并通过环境变量显式开启：

- `RUN_FULL_RESNET50_TESTS=1`：完整 ResNet-50 3×5 反向；
- `RUN_REAL_MODEL_TESTS=1`：真实 OpenAI CLIP/DFN 前后向；
- `REAL_TRAINING_CHECKPOINT=...`：真实 checkpoint `strict=True`；
- `RUN_COMPILE_TESTS=1`：compile default/reduce-overhead 对照。

## 已取得的证据

- 默认 CPU 自动测试：26 passed，6 skipped（资源型测试）。
- `RUN_FULL_RESNET50_TESTS=1`：完整 ResNet-50 3×5 反向通过。
- `RUN_COMPILE_TESTS=1`：fake CLIP 的 compile default/reduce-overhead
  CPU 对照通过；该结果不替代真实 CUDA 编译验证。
- 本地真实 OpenAI CLIP ViT-L/14：
  - 输入 `(1,3,1120,1120)`；
  - 输出 `(1,768)` FP32，全部有限；
  - attention mask `(257,257)`；
  - 完整输入反向梯度有限。
- 静态语法检查和 `git diff --check` 通过。

## 尚未闭合的风险

1. 本机没有可确认的 DFN 权重验证结果；DFN 返回结构仍需真实测试。
2. 未提供实际训练 checkpoint，旧训练权重 `strict=True` 仍未验证。
3. 未完成 CUDA FP32 原版/当前版固定 batch 的 logits、loss 和梯度对照。
4. 未完成 100–500 update 固定种子双跑。
5. AMP、compile、channels-last 和 chunk 尚无目标 GPU 性能基准。
6. compile 与前向 attention-mask 属性注入的组合只建立了可选测试和警告，
   未取得目标训练环境通过证据。

## 完整训练前强制检查

1. 将工作区提交为干净 commit；禁止从 dirty worktree 启动正式实验。
2. 固定数据版本并保存独立 manifest。
3. 三个 chunk 维持 0，AMP/compile/TF32 关闭，建立 CUDA FP32 基线。
4. 完成真实 DFN 与真实 checkpoint 验证。
5. 完成 CUDA 固定 batch 对照及 100–500 update 双跑。
6. 性能开关逐项单因素开启；误差和收益分别记录，不允许组合后归因。
7. 仅当短程 loss、梯度范数和验证指标无系统性偏离时进入完整训练。
