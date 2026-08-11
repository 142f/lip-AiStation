# LipFD 严格性能与冗余审计报告

> 完成日期：2026-08-11  
> 工作区：`E:\Project\lip-AiStation`，基线提交 `6466ce6`  
> 环境：Windows、Python 3.10、PyTorch 2.6.0+cu124、RTX 4060 Laptop GPU 8 GB  
> 结论：已实施的低风险优化在三套完整测试集（共 6,082 帧）、FP32 五步训练轨迹及 AMP/GradScaler 五步恢复轨迹上均严格一致，观测到的最大绝对误差为 **0.0**。

## 审计范围与验收方法

- 权重：`E:\best_model.pth` 用于完整推理；`E:\latest_checkpoint.pth` 用于训练恢复。
- 数据：LAV-DF 3,183 帧（1,593 real / 1,590 fake）、FakeAVCeleb 1,499 帧（500 / 999）、test-pro 1,400 帧（700 / 700）。
- checkpoint 按实际内容验收：
  - `best_model.pth`：`model`、`total_steps`、`update_steps`；
  - `latest_checkpoint.pth`：`model`、`optimizer`、`scaler`、`total_steps`、`update_steps`；
  - 两者均没有 scheduler、EMA、RNG snapshot 或 training_config，因此只标记为该文件的恢复能力限制，不推断保存代码不支持这些状态。
- 严格训练设置：恢复相同 RNG snapshot，关闭 TF32/compile；FP32 与 AMP 分轨；启用 deterministic algorithms 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。未启用这些确定性约束时，同一 CUDA 基线重跑自身也会出现梯度/optimizer hash 差异，因此不能作为 `<1e-7` 的有效对照。
- 整数、路径、顺序和 hash 要求完全一致；uint8 要求 `array_equal=True`；float 要求 shape/dtype/finite mask 一致且 `max_abs_error < 1e-7`。

## 1. 发现的冗余或性能瓶颈

| 优先级 | 问题 | 审计结论 |
| --- | --- | --- |
| P0 | validation/test 每 batch 对 CUDA tensor 调 `.tolist()`/`.cpu()` | 已优化并严格通过 |
| P0 | 上一 batch 大 tensor 与下一 batch H2D 短时共存 | 已优化并严格通过 |
| P0 | 视频 key、label/prob 数组在五种聚合和 CSV 中重复构建 | 已优化并严格通过 |
| P1 | sampler manifest 与 DataLoader 重复生成随机索引 | 运行时确认原代码无 cache；已加 epoch cache 并严格通过 |
| P1 | AVLips/FakeAVCeleb mel PNG 写盘后立即读回 | 改为相同 Matplotlib codec 的 BytesIO，像素逐位一致 |
| P1 | validation/test/single-video 仍使用 `no_grad` | 已改为 `inference_mode` |
| P1 | 日志中三个 loss 分别 `.item()` | 实测合并方案反而更慢，未采用 |
| P2 | Region softmax 后再 sum/divide | 数学上近似冗余，但会改变浮点路径，未采用 |
| P2 | AVLips/FakeAVCeleb 顺序 decode/face inference/write | 存在流水线空间；本轮不改变预处理调度 |

已确认无需重复修改：global uint8 H2D/GPU normalization、3×5 crop packed tensor、Region 单次批量 backbone forward、persistent workers，以及 `Trainer.test()` 已有的 `inference_mode`。

## 2. 每个问题所在文件、类和函数

- `validate.py::validate`：评估上下文及 prediction 收集。
- `test.py::test`、`build_video_groups`、`build_video_level_arrays`、`save_score_csv`：预测收集、视频分组及 CSV。
- `single_video_test.py::run_inference`：单视频预测收集。
- `trainer/trainer.py::Trainer.set_input`：batch tensor 生命周期。
- `data/__init__.py::EpochSeededRandomSampler`、`EpochSeededWeightedRandomSampler`：epoch 索引 materialization。
- `preprocess_avlips_multimodal.py::get_spectrogram`、`preprocess_fakeavceleb_multimodal.py::get_mel_image_from_wav`：mel PNG 编解码。
- `models/region_awareness.py::RegionAwareness.forward`：softmax 后重复归一化，仅审计未修改。
- `train.py` 的 loss 日志路径：三次 `.item()` 候选，仅实验未修改。

## 3. 当前计算为什么低效

- CUDA tensor 的 `.tolist()`/`.cpu()` 会迫使 CPU 每个 batch 等待 GPU，破坏异步流水线。
- Python 在右侧新 H2D 完成后才替换旧属性，旧/新 input 和 crops 会在 batch 边界同时驻留显存。
- 五种视频聚合原本分别把相同 Python list 转 NumPy，并重复从路径解析 video key。
- sampler 的 manifest 和迭代分别调用 `randperm`/`multinomial`，既重复计算又会额外消费 RNG。
- mel 数据只需 PNG codec 的确定性像素映射，临时磁盘文件没有算法贡献。
- 把三个 scalar 先 `stack` 再搬到 CPU 虽减少同步次数，却新增 GPU kernel；实际延迟高于三个 `.item()`。

## 4. 建议优化方式

- GPU 上保存小型 prediction chunks，循环后一次 `cat -> CPU -> NumPy float64`。
- 下一次 H2D 前删除上一 batch 的 input/crops/features/output/weights 引用，保留日志使用的 loss 标量。
- 每次测试只构建一次 video keys、labels 和 probabilities，复用于所有聚合与 CSV。
- sampler 以 epoch 为 cache key；相同 epoch 的 manifest 和 DataLoader 共享完全相同列表，换 epoch 才失效。
- 保持 Matplotlib 的 PNG encoder/decoder 和参数不变，只用 `BytesIO` 替代磁盘文件。
- 日志 loss 保持现状；若未来已有聚合 kernel 结果可直接复用，再重新评估。

## 5. 预期与实测影响

| 修改 | 主要影响 | 实测 |
| --- | --- | ---: |
| 延迟 prediction D2H | 减少每 batch GPU 同步 | ABBA×10、500×32：7.355±0.211 -> 0.374±0.160 ms（median±MAD，-94.9%） |
| 缓存视频解析/数组 | 降低测试报告 CPU 开销 | test-pro 五聚合：18.990 ms -> 4.857 ms（-74.4%） |
| 释放旧 batch | 降低 batch 边界峰值显存 | B=16 合成边界：736.001 -> 368.001 MiB（-368 MiB） |
| sampler epoch cache | 避免重复采样和 RNG 消耗 | 100k 索引双 materialize：20.028 -> 18.093 ms；顺序完全一致 |
| mel BytesIO | 消除临时文件 I/O | 30 次中位数：12.351 -> 4.685 ms（-62.1%） |
| 三 loss 合并同步候选 | 原期望减少同步 | 35.7 -> 47.8 µs，变慢 33.9%，已撤销 |

这些修改不改变模型 FLOPs、网络结构、loss、数据划分、随机种子或测试协议。完整模型端到端速度受 GPU 温度、Windows 调度和数据缓存波动明显，故速度结论以预热的局部路径和独立 fresh-process 显存测试为准；完整数据运行用于数值验收，不把非 ABBA 的墙钟差异归因于代码修改。

## 6. 是否改变计算顺序或数值结果

- 模型 forward、loss、backward、optimizer 的数学运算顺序未改变。
- prediction 仅推迟传输时间，元素顺序、float32 值及进入 sklearn 前的 float64 转换保持一致。
- `inference_mode` 仅用于本来就不需要 autograd 的入口。
- sampler 缓存复用首次生成的索引；同一 epoch 顺序不变，并避免第二次随机采样造成 RNG 漂移。
- mel 使用完全相同 codec；decoded uint8 像素 `array_equal=True`。
- 所有已采用项的最终观测 `max_abs_error=0.0`。

## 7. 实际实施的代码修改

1. `utils.py` 新增统一的 `prediction_chunks_to_numpy`。
2. `validate.py`、`test.py`、`single_video_test.py` 使用 `inference_mode` 与延迟 D2H。
3. `test.py` 复用 video keys、标签/概率数组和分组结果。
4. `Trainer.set_input` 在下一批 H2D 前释放上一批大 tensor 引用。
5. 两个 epoch sampler 新增 cache 及正确失效逻辑。
6. AVLips/FakeAVCeleb 的 mel 临时 PNG 改为 BytesIO；AVLips 保留兼容参数用于严格对照。
7. 新增 `tools/performance_audit_inference.py`、`tools/performance_audit_training.py` 和 `tests/test_performance_equivalence.py`，可复跑完整推理、五步训练和单元等价测试。
8. 三 loss 合并方案经 benchmark 后已撤销，`train.py` 不含该优化。

## 8. 优化前后 benchmark

### 完整真实数据推理（数值验收运行）

| 数据集 | 帧数 | 基线时间 / 吞吐 | 候选时间 / 吞吐 | peak allocated / reserved（基线；候选） |
| --- | ---: | ---: | ---: | ---: |
| LAV-DF | 3,183 | 689.559 s / 4.616 sps | 1,051.151 s / 3.028 sps | 4503.768/5790；4503.864/5790 MiB |
| FakeAVCeleb | 1,499 | 488.966 s / 3.066 sps | 510.159 s / 2.938 sps | 4503.768/5790；4503.813/5790 MiB |
| test-pro | 1,400 | 470.069 s / 2.978 sps | 483.672 s / 2.895 sps | 4503.768/5790；4503.810/5790 MiB |

该完整 runner 为保存逐帧严格证据，每 batch 还会同步保存 logits，且两版本不是交替 ABBA；尤其 LAV-DF 候选运行受后续环境/热状态影响，不能据此判定核心模型变慢。可归因的局部 ABBA/预热结果及 fresh-process 显存结果见第 5 节。

### 训练轨迹（5 updates，每 update 累积 4 个真实 batch，随后固定 40 帧验证）

| 轨道 | 基线 | 候选 | peak allocated | 结果 |
| --- | ---: | ---: | ---: | --- |
| FP32 strict | 44.729 s | 44.282 s | 3753.529 MiB（两者） | PASS，max error 0 |
| AMP/GradScaler resume | 38.592 s | 37.844 s | 4094.876 -> 4091.599 MiB | PASS，max error 0 |

### 其他基线剖析

- Dataset 单样本中位耗时：LAV-DF 47.913 ms、FakeAVCeleb 44.399 ms、test-pro 44.223 ms。
- 模型吞吐随 batch 从 B=1 的 4.91 samples/s 增至 B=12 的 5.58 samples/s；B=16 为 5.54 samples/s，说明 8 GB 环境的合理推理区间约为 B=8–16。

## 9. 优化前后最大数值误差

| 对照项 | 最大绝对误差 / 一致性 |
| --- | --- |
| 三数据集预处理 tensor（代码未改变的 dataset 路径） | 基线 SHA256 保持；shape/dtype 不变 |
| mel disk PNG vs BytesIO PNG | uint8 `array_equal=True`，max 0 |
| 6,082 帧 labels、paths、logits、probabilities | 完全一致，max 0 |
| mean/max/median/top3_mean/top5_mean 视频聚合 | labels/order 完全一致，probability max 0 |
| FP32 五步 logits、RALoss、CE、total loss、grad norm | max 0 |
| FP32 gradients、parameters、AdamW states | SHA256 每步完全一致 |
| AMP 五步上述状态及 scaler scale/growth tracker | 完全一致，max 0；第 4 步两边均 skip，其余均未 skip |
| Python/NumPy/Torch CPU/Torch CUDA RNG | 每步 SHA256 完全一致 |
| 训练后固定 40 帧 validation logits/probabilities/metrics | max 0 |

## 10. Forward/backward/test 是否严格通过一致性验证

| 验证 | 状态 |
| --- | --- |
| `best_model.pth` strict load 与 CUDA FP32 forward | PASS |
| 三套完整测试集 frame prediction/order | PASS（6,082/6,082） |
| 五种视频级聚合及指标 | PASS |
| CUDA FP32 full backward，连续 5 updates | PASS（B=1、accumulation=4） |
| CPU full backward | 未用来替代 CUDA；本轮 CUDA 已可完成，故无需以 CPU 结论代替 |
| FP32 optimizer/AdamW/RNG 轨迹 | PASS |
| AMP/GradScaler resume 独立轨道 | PASS |
| test-pro 训练后固定验证 | PASS；基线/候选 AUC=1、AP=1、ACC=1，prediction max error 0 |
| 自动化回归测试 | `8 passed, 2 skipped`；跳过项为需显式开启的完整音频慢测 |
| Python compile 与 `git diff --check` | PASS |

完整测试指标前后相同：

- LAV-DF frame AUC `0.6872097660`、AP `0.6815920657`、ACC `0.6374489475`、top3 video AUC `0.7326593230`。
- FakeAVCeleb frame AUC `0.9296256256`、AP `0.9655410231`、ACC `0.8619079386`、top3 video AUC `0.9504`。
- test-pro frame AUC `0.9394755102`、AP `0.9138534299`、ACC `0.8892857143`、top3 video AUC `0.9561224490`。

## 11. 未采用的优化及原因

- **三 loss 合并同步**：数值等价，但新增 `stack` kernel 后实测 47.8 µs，高于原 35.7 µs；因性能负收益撤销，不是数值失败。
- **local crops uint8 H2D / GPU resize-normalize**：会把 `normalized float -> slice -> torchvision Resize` 改成另一浮点路径，预计不能保证 `<1e-7`，未改。
- **删除 softmax 后 `weights_sum` 除法**：数学上冗余但浮点 sum 不保证精确为 1，可能传播至 forward/backward，未改。
- **attention/mean/sum/top-k 重排、TF32、compile、channels-last**：可能改变 kernel 或归约顺序，不符合 strict 主轨道，未启用。
- **默认 AMP**：AMP 仅作为 checkpoint resume 的独立一致性轨道验证，不替代 FP32，也不改变默认训练协议。
- **`grab/retrieve` 视频读取、多进程 InsightFace**：可能受 decoder/backend 和 GPU context 竞争影响；尚无逐帧像素全零误差及吞吐证据，未改。
- **activation chunk/checkpoint**：会改变执行/归约行为且当前 8 GB 下 B=16 推理、B=1 full backward 已可运行，收益不足以承担 strict 风险。
- **删除未引用模型文件**：对运行时 FLOPs/显存无收益，可能破坏历史模型或外部脚本兼容，未删。

## 复现入口

```powershell
python -m pytest tests/test_performance_equivalence.py -q
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python tools/performance_audit_training.py --checkpoint E:\latest_checkpoint.pth --data-root E:\data\test-pro --output .tmp\train.json --mode candidate --precision fp32 --updates 5 --accumulation 4
python tools/performance_audit_inference.py --checkpoint E:\best_model.pth --data-root E:\data\test-pro --output-prefix .tmp\testpro --collection deferred --context inference_mode
python tools/performance_audit_microbench.py
```
