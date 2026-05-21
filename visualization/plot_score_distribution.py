import argparse
import csv
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = {"label", "video_score"}


def set_paper_style():
    """设置更适合论文排版的 Matplotlib 风格。"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        "axes.linewidth": 0.75,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 8.6,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })


def normalize_title(raw_title):
    """把模型名整理成论文图中更稳定的写法。"""
    if raw_title is None:
        return None

    title = str(raw_title).strip()
    mapping = {
        "baseline": "Baseline",

        "wo_bgi": "w/o BGI",
        "w/o_bgi": "w/o BGI",
        "without_bgi": "w/o BGI",
        "global_ablation": "w/o BGI",
        "wo_global_innov": "w/o BGI",
        "w/o_global_innov": "w/o BGI",
        "without_global_innov": "w/o BGI",

        "wo_rae": "w/o RAE",
        "w/o_rae": "w/o RAE",
        "without_rae": "w/o RAE",

        "region_pe_se_ablation": "w/o Region PE/SE",
        "wo_region_pe_se": "w/o Region PE/SE",
        "w/o_region_pe_se": "w/o Region PE/SE",
        "without_region_pe_se": "w/o Region PE/SE",
        "w/o_region_pe+se": "w/o Region PE/SE",

        "full_model": "Full MB-ViT",
        "full": "Full MB-ViT",
        "full_mbvit": "Full MB-ViT",
        "full_mbv-it": "Full MB-ViT",
        "mb-vit": "Full MB-ViT",
        "mbvit": "Full MB-ViT",
    }

    key = title.lower().replace(" ", "_").replace("-", "_")
    return mapping.get(key, title)


def validate_csv_header(fieldnames, csv_path):
    """检查 CSV 是否包含必要字段。"""
    if fieldnames is None:
        raise ValueError(f"CSV 文件为空或无法读取表头: {csv_path}")

    missing = REQUIRED_COLUMNS - set(fieldnames)
    if missing:
        raise ValueError(f"CSV 缺少必要字段 {sorted(missing)}: {csv_path}")


def read_video_scores(csv_path, override_title=None):
    """读取 test.py 导出的 video_scores CSV，并进行合法性检查。"""
    labels = []
    scores = []
    model_name = None

    csv_path = str(csv_path)
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        validate_csv_header(reader.fieldnames, csv_path)

        for row_idx, row in enumerate(reader, start=2):
            try:
                label = int(row["label"])
                score = float(row["video_score"])
            except Exception as exc:
                raise ValueError(f"第 {row_idx} 行 label 或 video_score 无法解析: {csv_path}") from exc

            if label not in (0, 1):
                raise ValueError(f"第 {row_idx} 行 label 必须为 0 或 1，当前为 {label}: {csv_path}")

            if not np.isfinite(score):
                raise ValueError(f"第 {row_idx} 行 video_score 不是有限数值: {csv_path}")

            if score < -1e-6 or score > 1.0 + 1e-6:
                raise ValueError(
                    f"第 {row_idx} 行 video_score 超出 [0, 1] 范围，当前为 {score}: {csv_path}"
                )

            labels.append(label)
            scores.append(min(max(score, 0.0), 1.0))
            model_name = row.get("model_name") or model_name

    if len(labels) == 0:
        raise ValueError(f"CSV 中没有有效样本: {csv_path}")

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    if np.sum(labels == 0) == 0:
        print(f"[警告] 该 CSV 没有 Real 样本，图中 Real 曲线将为空: {csv_path}")
    if np.sum(labels == 1) == 0:
        print(f"[警告] 该 CSV 没有 Fake 样本，图中 Fake 曲线将为空: {csv_path}")

    if override_title is not None:
        title = override_title
    else:
        title = model_name or Path(csv_path).parent.name or Path(csv_path).stem

    return {
        "path": csv_path,
        "title": normalize_title(title),
        "labels": labels,
        "scores": scores,
    }


def smooth_histogram(values, bins):
    """用平滑直方图近似密度，避免额外依赖 scipy。"""
    if values.size == 0:
        return np.zeros(len(bins) - 1, dtype=np.float64)

    hist, _ = np.histogram(values, bins=bins, density=True)

    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel = kernel / kernel.sum()

    smoothed = np.convolve(hist, kernel, mode="same")

    # 卷积边界会轻微改变面积，这里重新归一化，保证仍可解释为密度曲线
    bin_width = bins[1] - bins[0]
    area = np.sum(smoothed) * bin_width
    if area > 0:
        smoothed = smoothed / area

    return smoothed


def compute_auc(labels, scores):
    """计算 AUC；不依赖 sklearn。"""
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    pos = labels == 1
    neg = labels == 0

    n_pos = np.sum(pos)
    n_neg = np.sum(neg)

    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(scores)
    sorted_scores = scores[order]

    ranks = np.zeros_like(scores, dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1

        # rank 从 1 开始；相同分数取平均 rank
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    sum_pos_ranks = np.sum(ranks[pos])
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_ap(labels, scores):
    """计算 Average Precision；不依赖 sklearn。"""
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    n_pos = np.sum(labels == 1)
    if n_pos == 0:
        return np.nan

    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tp = np.cumsum(sorted_labels == 1)
    ranks = np.arange(1, len(sorted_labels) + 1)
    precision = tp / ranks

    ap = np.sum(precision[sorted_labels == 1]) / n_pos
    return float(ap)


def compute_youden_accuracy(labels, scores):
    """计算 Youden 最优阈值下的 ACC，用于和 test.py 的报告口径对齐。"""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    if labels.size == 0 or len(np.unique(labels)) < 2:
        return np.nan

    thresholds = np.unique(scores)
    best_j = -np.inf
    best_acc = np.nan

    for threshold in thresholds:
        preds = (scores >= threshold).astype(np.int64)

        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        j_score = tpr - fpr

        if j_score > best_j:
            best_j = j_score
            best_acc = (tp + tn) / labels.size

    return float(best_acc)


def format_metric(value):
    """格式化指标显示。"""
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.3f}"


def get_layout(num_items):
    """根据模型数量自动选择子图布局。"""
    if num_items == 1:
        return 1, 1, (3.20, 2.35)
    if num_items == 2:
        return 1, 2, (6.10, 2.35)
    if num_items == 3:
        return 1, 3, (7.10, 2.25)
    if num_items == 4:
        return 2, 2, (6.80, 4.80)
    raise ValueError("主文图最多建议展示 4 个模型。")


def build_metric_text(labels, scores, metric_mode):
    """生成子图内指标文本。"""
    if metric_mode == "none":
        return None

    auc = compute_auc(labels, scores)

    if metric_mode == "auc_ap":
        ap = compute_ap(labels, scores)
        return f"AUC={format_metric(auc)}\nAP={format_metric(ap)}"

    if metric_mode == "auc_acc":
        acc = compute_youden_accuracy(labels, scores)
        return f"AUC={format_metric(auc)}\nACC={format_metric(acc)}"

    raise ValueError(f"未知的指标显示模式: {metric_mode}")


def plot_score_distribution(
    csv_paths,
    output_prefix,
    titles=None,
    bins_count=41,
    metric_mode="auc_ap",
    threshold=0.5,
    show_threshold=False,
    dpi=600,
    formats=("pdf", "png", "svg"),
):
    """绘制多个模型的视频级 Real/Fake 分数分布图。"""
    set_paper_style()

    if len(csv_paths) == 0:
        raise ValueError("至少需要提供一个 video_scores CSV 文件。")

    if len(csv_paths) > 4:
        raise ValueError("主文分数分布图建议最多展示 4 个模型，请减少 CSV 数量。")

    if titles is not None and len(titles) != len(csv_paths):
        raise ValueError("--titles 的数量必须与 CSV 文件数量一致。")

    if bins_count < 10:
        raise ValueError("--bins 过小，建议至少为 10。")

    allowed_formats = {"pdf", "png", "svg"}
    unknown_formats = set(formats) - allowed_formats
    if unknown_formats:
        raise ValueError(f"不支持的输出格式: {sorted(unknown_formats)}")

    items = []
    for idx, path in enumerate(csv_paths):
        title = titles[idx] if titles is not None else None
        items.append(read_video_scores(path, override_title=title))

    bins = np.linspace(0.0, 1.0, bins_count)
    centers = (bins[:-1] + bins[1:]) / 2.0

    rows, cols, figsize = get_layout(len(items))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = np.asarray(axes).reshape(-1)

    # 色盲友好配色；同时使用线型区分，便于黑白打印
    real_color = "#0072B2"
    fake_color = "#D55E00"

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(items):
            ax.axis("off")
            continue

        item = items[ax_idx]
        labels = item["labels"]
        scores = item["scores"]

        real_scores = scores[labels == 0]
        fake_scores = scores[labels == 1]

        real_density = smooth_histogram(real_scores, bins)
        fake_density = smooth_histogram(fake_scores, bins)

        ax.plot(
            centers,
            real_density,
            color=real_color,
            linewidth=1.55,
            linestyle="-",
            label="Real",
        )
        ax.fill_between(
            centers,
            real_density,
            color=real_color,
            alpha=0.12,
            linewidth=0,
        )

        ax.plot(
            centers,
            fake_density,
            color=fake_color,
            linewidth=1.55,
            linestyle="--",
            label="Fake",
        )
        ax.fill_between(
            centers,
            fake_density,
            color=fake_color,
            alpha=0.12,
            linewidth=0,
        )

        if show_threshold:
            ax.axvline(
                threshold,
                color="0.25",
                linestyle=":",
                linewidth=0.9,
                zorder=0,
            )

        panel_label = f"({string.ascii_lowercase[ax_idx]})"
        ax.set_title(f"{panel_label} {item['title']}", pad=4)

        ax.set_xlim(0.0, 1.0)
        ax.grid(True, linestyle=":", linewidth=0.45, alpha=0.38)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        metric_text = build_metric_text(labels, scores, metric_mode)
        if metric_text is not None:
            ax.text(
                0.035,
                0.94,
                metric_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=6.8,
                linespacing=1.05,
                bbox={
                    "boxstyle": "round,pad=0.20",
                    "facecolor": "white",
                    "edgecolor": "0.82",
                    "linewidth": 0.45,
                    "alpha": 0.88,
                },
            )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        handlelength=2.7,
        columnspacing=1.7,
    )

    fig.supxlabel("Predicted fake probability", fontsize=8.9)
    fig.supylabel("Density", fontsize=8.9)

    if rows == 1:
        fig.tight_layout(rect=(0.02, 0.08, 1.0, 0.86), w_pad=1.15)
    else:
        fig.tight_layout(rect=(0.04, 0.04, 1.0, 0.91), w_pad=1.0, h_pad=1.0)

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        out_path = output_prefix.with_suffix(f".{fmt}")
        if fmt == "png":
            fig.savefig(out_path, dpi=dpi, facecolor="white")
        else:
            fig.savefig(out_path, facecolor="white")
        saved_paths.append(out_path)

    plt.close(fig)

    for out_path in saved_paths:
        print(f"[可视化] 已保存分数分布图: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="绘制论文风格的视频级 Real/Fake 分数分布图")

    parser.add_argument(
        "csv",
        nargs="+",
        help="一个或多个 video_scores CSV 文件，建议顺序为 w/o BGI, w/o Region PE/SE, Full MB-ViT",
    )
    parser.add_argument(
        "--output",
        default="./vis_outputs/fig_score_distribution",
        help="输出文件前缀，不需要写扩展名",
    )
    parser.add_argument(
        "--titles",
        nargs="+",
        default=None,
        help="可选：为每个 CSV 指定子图标题，数量必须与 CSV 一致",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=41,
        help="直方图分箱数量，默认 41",
    )
    parser.add_argument(
        "--metric_mode",
        choices=["auc_ap", "auc_acc", "none"],
        default="auc_ap",
        help="子图指标显示模式：auc_ap 显示 AUC/AP；auc_acc 显示 AUC/ACC；none 不显示指标框",
    )
    parser.add_argument(
        "--show_threshold",
        action="store_true",
        help="显示参考阈值线。注意：如果不是实际决策阈值，论文图注应写 reference threshold。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="参考阈值线位置，默认 0.5，仅在 --show_threshold 开启时显示",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png", "svg"],
        choices=["pdf", "png", "svg"],
        help="输出格式，默认同时保存 pdf png svg",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG 输出分辨率，默认 600",
    )

    return parser.parse_args()


def main():
    """命令行入口。"""
    try:
        args = parse_args()
        plot_score_distribution(
            csv_paths=args.csv,
            output_prefix=args.output,
            titles=args.titles,
            bins_count=args.bins,
            metric_mode=args.metric_mode,
            threshold=args.threshold,
            show_threshold=args.show_threshold,
            dpi=args.dpi,
            formats=tuple(args.formats),
        )
    except Exception as exc:
        print(f"[错误] 绘图失败: {exc}")
        raise


if __name__ == "__main__":
    main()