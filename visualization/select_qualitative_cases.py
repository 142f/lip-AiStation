#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从三组视频级分数 CSV 中筛选 MB-ViT 成功/失败案例。

设计目标：
1. 正文推荐使用 same_sample：每个案例行固定同一个 video_key，
   比较 w/o BGI、w/o RAE、Full MB-ViT 在同一输入上的视频级 fake score。
2. diverse_samples 可用于补充材料：仍按 video_key 对齐三组模型分数，但更强调代表性场景覆盖。
3. Failure / Hard Case 的原因根据错误类型自动生成，避免对真实样本误报和伪造样本漏检使用同一解释。

输入 CSV 至少包含：
video_key,label,video_score

如果 Full CSV 中包含 top_frame_paths/top_frame_scores，会原样保留，供绘图脚本展示代表帧。
"""

import argparse
import csv
import math
import os
from typing import Dict, Iterable, List, Optional


REQUIRED_COLUMNS = {"video_key", "label", "video_score"}


def read_score_table(csv_path: str, score_name: str, keep_extra: bool = False) -> List[Dict]:
    """读取视频级分数 CSV，并将 video_score 重命名为指定模型分数字段。"""
    rows: List[Dict] = []

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 文件为空或无法读取表头: {csv_path}")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV 缺少字段 {sorted(missing)}: {csv_path}")

        for item in reader:
            row = {
                "video_key": item["video_key"],
                "label": int(float(item["label"])),
                score_name: float(item["video_score"]),
            }

            if keep_extra:
                row["top_frame_paths"] = item.get("top_frame_paths", "")
                row["top_frame_scores"] = item.get("top_frame_scores", "")
                row["num_frames"] = item.get("num_frames", "")
                row["agg_method"] = item.get("agg_method", "")
                row["model_name"] = item.get("model_name", "")

            rows.append(row)

    if not rows:
        raise RuntimeError(f"CSV 中没有有效样本: {csv_path}")

    return rows


def predict(score: float, threshold: float) -> int:
    """根据分数和阈值生成预测标签。"""
    return int(float(score) >= threshold)


def merge_scores(full_rows: List[Dict], wo_bgi_rows: List[Dict], wo_region_rows: List[Dict]) -> List[Dict]:
    """按 video_key 合并三组模型结果，保证同一行是同一个视频样本。"""
    wo_bgi_by_key = {row["video_key"]: row for row in wo_bgi_rows}
    wo_region_by_key = {row["video_key"]: row for row in wo_region_rows}

    merged: List[Dict] = []
    dropped = 0

    for full in full_rows:
        key = full["video_key"]
        if key not in wo_bgi_by_key or key not in wo_region_by_key:
            dropped += 1
            continue

        row = dict(full)
        row["wo_bgi_score"] = wo_bgi_by_key[key]["wo_bgi_score"]
        row["wo_region_score"] = wo_region_by_key[key]["wo_region_score"]
        merged.append(row)

    if dropped > 0:
        print(f"[案例筛选] 有 {dropped} 个 Full 视频未能在消融结果中对齐，已跳过。")

    return merged


def add_predictions(rows: Iterable[Dict], threshold: float) -> None:
    """为合并后的结果添加三组模型预测标签。"""
    for row in rows:
        row["full_pred"] = predict(row["full_score"], threshold)
        row["wo_bgi_pred"] = predict(row["wo_bgi_score"], threshold)
        row["wo_region_pred"] = predict(row["wo_region_score"], threshold)


def sort_by(rows: List[Dict], key: str, reverse: bool = False) -> List[Dict]:
    """按指定字段排序。"""
    return sorted(rows, key=lambda row: row.get(key, 0.0), reverse=reverse)


def with_case(row: Dict, case_type: str, reason: str) -> Dict:
    """复制样本并添加案例类型和说明。"""
    item = dict(row)
    item["case_type"] = case_type
    item["reason"] = reason
    return item


def infer_failure_reason(row: Dict, threshold: float) -> str:
    """根据错误类型生成更严谨的失败原因。"""
    label = int(row.get("label", 0))
    pred = int(row.get("full_pred", 0))
    score = float(row.get("full_score", 0.0))

    if label == 0 and pred == 1:
        return "Possible reason: low visual quality, unusual pose, or domain bias"

    if label == 1 and pred == 0:
        return "Possible reason: local manipulation, weak audio-visual mismatch, or diluted fake evidence"

    if abs(score - threshold) <= 0.10:
        return "Possible reason: ambiguous fake score near the decision boundary"

    return "Possible reason: challenging sample under the current video-level protocol"


def choose_real_correct(rows: List[Dict]) -> Dict:
    """选择真实视频正确识别案例。"""
    candidates = [row for row in rows if row["label"] == 0 and row["full_pred"] == 0]

    if candidates:
        row = sort_by(candidates, "full_score")[0]
        return with_case(row, "Real Correct", "Low fake score on real video")

    print("[案例筛选] 未找到 Full 正确分类的真实样本，改选 Full 分数最低的真实样本。")
    fallback = [row for row in rows if row["label"] == 0]
    if fallback:
        row = sort_by(fallback, "full_score")[0]
        return with_case(row, "Real Correct", "Fallback: lowest fake score among real videos")

    print("[案例筛选] 未找到真实样本，改选全体样本中 Full 分数最低者。")
    row = sort_by(rows, "full_score")[0]
    return with_case(row, "Real Correct", "Fallback: lowest fake score in all videos")


def choose_fake_correct(rows: List[Dict]) -> Dict:
    """选择伪造视频正确检出案例。"""
    candidates = [row for row in rows if row["label"] == 1 and row["full_pred"] == 1]

    if candidates:
        row = sort_by(candidates, "full_score", reverse=True)[0]
        return with_case(row, "Fake Correct", "High fake score on fake video")

    print("[案例筛选] 未找到 Full 正确分类的伪造样本，改选 Full 分数最高的伪造样本。")
    fallback = [row for row in rows if row["label"] == 1]
    if fallback:
        row = sort_by(fallback, "full_score", reverse=True)[0]
        return with_case(row, "Fake Correct", "Fallback: highest fake score among fake videos")

    print("[案例筛选] 未找到伪造样本，改选全体样本中 Full 分数最高者。")
    row = sort_by(rows, "full_score", reverse=True)[0]
    return with_case(row, "Fake Correct", "Fallback: highest fake score in all videos")


def choose_full_improves(rows: List[Dict], threshold: float) -> Dict:
    """选择消融模型错误但 Full 正确的案例。"""
    strict = [
        row for row in rows
        if row["full_pred"] == row["label"]
        and (row["wo_bgi_pred"] != row["label"] or row["wo_region_pred"] != row["label"])
    ]

    if strict:
        for row in strict:
            row["_margin"] = abs(row["full_score"] - threshold)
        row = sort_by(strict, "_margin", reverse=True)[0]
        return with_case(row, "Full Improves", "Ablated variant fails, Full MB-ViT succeeds")

    print("[案例筛选] 未找到消融模型错误且 Full 正确的严格案例，改选 Full 判别分数改善最明显的样本。")
    for row in rows:
        if row["label"] == 1:
            gap = (row["full_score"] - row["wo_bgi_score"]) + (row["full_score"] - row["wo_region_score"])
        else:
            gap = (row["wo_bgi_score"] - row["full_score"]) + (row["wo_region_score"] - row["full_score"])
        row["_improve_gap"] = gap

    improved = [row for row in rows if row.get("_improve_gap", -999.0) > 0]
    if improved:
        row = sort_by(improved, "_improve_gap", reverse=True)[0]
    else:
        row = sort_by(rows, "_improve_gap", reverse=True)[0]

    return with_case(row, "Full Improves", "Full MB-ViT gives a more discriminative score")


def choose_failure_or_hard(
    rows: List[Dict],
    threshold: float,
    failure_policy: str = "confident",
    failure_type: str = "any",
) -> Dict:
    """选择 Full 失败案例；可指定误报 FP 或漏检 FN。若没有对应错误，则退化为困难样本。"""
    all_failures = [row for row in rows if row["full_pred"] != row["label"]]

    if failure_type == "false_positive":
        failures = [row for row in all_failures if row["label"] == 0 and row["full_pred"] == 1]
        if not failures:
            print("[案例筛选] 未找到 GT=Real, Pred=Fake 的误报样本，将退化为任意错误样本。")
            failures = all_failures
    elif failure_type == "false_negative":
        failures = [row for row in all_failures if row["label"] == 1 and row["full_pred"] == 0]
        if not failures:
            print("[案例筛选] 未找到 GT=Fake, Pred=Real 的漏检样本，将退化为任意错误样本。")
            failures = all_failures
    else:
        failures = all_failures

    if failures:
        for row in failures:
            row["_confidence"] = abs(row["full_score"] - threshold)

        if failure_policy == "near_threshold":
            row = sort_by(failures, "_confidence")[0]
        else:
            row = sort_by(failures, "_confidence", reverse=True)[0]

        return with_case(row, "Failure / Hard Case", infer_failure_reason(row, threshold))

    print("[案例筛选] 未找到 Full 错误样本，改选 Full 分数最接近阈值的困难样本。")
    for row in rows:
        row["_hardness"] = abs(row["full_score"] - threshold)

    row = sort_by(rows, "_hardness")[0]
    return with_case(row, "Failure / Hard Case", "Possible reason: ambiguous fake score near the decision boundary")


def choose_focus_sample(rows: List[Dict], focus_video_key: Optional[str], threshold: float) -> List[Dict]:
    """选择指定 video_key 的单样本对比，用于极简 same-sample 图。"""
    if not focus_video_key:
        row = choose_full_improves(rows, threshold)
    else:
        matched = [row for row in rows if row["video_key"] == focus_video_key]
        if not matched:
            raise RuntimeError(f"未找到指定 video_key: {focus_video_key}")
        row = with_case(matched[0], "Selected Sample", "Same input compared across model variants")

    return [row]


def clean_case(row: Dict) -> Dict[str, str]:
    """只保留论文定性图需要的稳定字段。"""
    fields = [
        "case_type",
        "video_key",
        "label",
        "full_score",
        "wo_bgi_score",
        "wo_region_score",
        "full_pred",
        "wo_bgi_pred",
        "wo_region_pred",
        "reason",
        "top_frame_paths",
        "top_frame_scores",
        "num_frames",
        "agg_method",
    ]

    clean: Dict[str, str] = {}
    for field in fields:
        value = row.get(field, "")
        if isinstance(value, float) and math.isfinite(value):
            value = f"{value:.6f}"
        clean[field] = value

    return clean


def save_cases(cases: List[Dict], output: str) -> None:
    """保存案例 CSV。"""
    if not cases:
        raise RuntimeError("没有可保存的案例。")

    output_dir = os.path.dirname(os.path.abspath(output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    clean_cases = [clean_case(row) for row in cases]
    with open(output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean_cases[0].keys()))
        writer.writeheader()
        writer.writerows(clean_cases)

    print(f"[案例筛选] 已保存案例列表: {output}")


def main(args) -> None:
    """主流程：读取三组 CSV，合并结果，筛选案例，并保存案例表。"""
    full = read_score_table(args.full_csv, "full_score", keep_extra=True)
    wo_bgi = read_score_table(args.wo_bgi_csv, "wo_bgi_score")
    wo_region = read_score_table(args.wo_region_csv, "wo_region_score")

    rows = merge_scores(full, wo_bgi, wo_region)
    if not rows:
        raise RuntimeError("三组 CSV 按 video_key 合并后为空，请检查测试清单是否一致。")

    add_predictions(rows, args.threshold)

    if args.only_focus_sample:
        cases = choose_focus_sample(rows, args.focus_video_key, args.threshold)
    else:
        # 正文推荐 same_sample：每一行都是同一个 video_key 的三模型对比。
        # diverse_samples 用于补充材料，强调案例类型覆盖。
        cases = [
            choose_real_correct(rows),
            choose_fake_correct(rows),
            choose_full_improves(rows, args.threshold),
            choose_failure_or_hard(rows, args.threshold, args.failure_policy, args.failure_type),
        ]

    save_cases(cases, args.output)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="筛选 MB-ViT 成功/失败定性案例。")
    parser.add_argument("--full_csv", required=True, help="Full MB-ViT 的 video_scores CSV")
    parser.add_argument("--wo_bgi_csv", required=True, help="w/o BGI 的 video_scores CSV")
    parser.add_argument("--wo_region_csv", required=True, help="w/o RAE 或 RAE 相关消融模型的 video_scores CSV，保持参数名兼容旧脚本。")
    parser.add_argument("--output", required=True, help="输出 qualitative_cases.csv")
    parser.add_argument("--threshold", type=float, default=0.5, help="案例筛选阈值，不一定等同论文最终决策阈值")
    parser.add_argument(
        "--case_mode",
        choices=["same_sample", "diverse_samples"],
        default="same_sample",
        help="same_sample 推荐正文使用；diverse_samples 推荐补充材料使用。",
    )
    parser.add_argument(
        "--failure_policy",
        choices=["confident", "near_threshold"],
        default="confident",
        help="confident 选择置信度较高的错误案例；near_threshold 选择靠近阈值的困难案例。",
    )
    parser.add_argument(
        "--failure_type",
        choices=["any", "false_positive", "false_negative"],
        default="any",
        help="选择失败案例类型：any 任意错误；false_positive 为 GT=Real, Pred=Fake；false_negative 为 GT=Fake, Pred=Real。",
    )
    parser.add_argument(
        "--focus_video_key",
        default="",
        help="指定单个 video_key 生成极简 same-sample 图。",
    )
    parser.add_argument(
        "--only_focus_sample",
        action="store_true",
        help="只输出一个指定样本或自动选择的 Full Improves 样本。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
