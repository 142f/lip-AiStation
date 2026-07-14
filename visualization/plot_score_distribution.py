import argparse
import csv
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = {"label", "video_score"}


def set_paper_style(dpi: int = 600, font_scale: float = 1.0) -> None:
    """设置更适合论文排版的 Matplotlib 全局风格。"""
    base = float(font_scale)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        "axes.linewidth": 0.70,
        "axes.titlesize": 8.2 * base,
        "axes.labelsize": 7.8 * base,
        "xtick.labelsize": 7.0 * base,
        "ytick.labelsize": 7.0 * base,
        "legend.fontsize": 7.6 * base,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
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

        # 兼容旧实验目录名，但论文图中统一显示为 w/o SRAC。
        # 这里的 region_pe_se_ablation 实际对应关闭 SRAC 中的尺度--区域身份编码与 SE 重标定，
        # 不是完整移除 RAE。
        "region_pe_se_ablation": "w/o SRAC",
        "wo_region_pe_se": "w/o SRAC",
        "w/o_region_pe_se": "w/o SRAC",
        "without_region_pe_se": "w/o SRAC",
        "w/o_region_pe+se": "w/o SRAC",
        "w/o_region_pe/se": "w/o SRAC",
        "wo_srac": "w/o SRAC",
        "w/o_srac": "w/o SRAC",
        "without_srac": "w/o SRAC",
        "srac_ablation": "w/o SRAC",

        "full_model": "Full MB-ViT",
        "full": "Full MB-ViT",
        "full_mbvit": "Full MB-ViT",
        "full_mbv_it": "Full MB-ViT",
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

    # 卷积边界会轻微改变面积，这里重新归一化，保证仍可解释为密度曲线。
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

        # rank 从 1 开始；相同分数取平均 rank。
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


def get_layout(num_items, fig_width=None, fig_height=None):
    """根据模型数量自动选择更紧凑的论文图布局。"""
    if num_items == 1:
        default = (3.20, 2.05)
        return 1, 1, (fig_width or default[0], fig_height or default[1])
    if num_items == 2:
        default = (5.80, 2.05)
        return 1, 2, (fig_width or default[0], fig_height or default[1])
    if num_items == 3:
        default = (7.20, 2.18)
        return 1, 3, (fig_width or default[0], fig_height or default[1])
    if num_items == 4:
        default = (6.60, 4.30)
        return 2, 2, (fig_width or default[0], fig_height or default[1])
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


def get_layout_params(rows):
    """返回紧凑排版参数。

    说明：
    - legend 和全局 x 轴标题不以整张画布中心为准，而是以子图绘图区中心为准；
    - 这样可以避免左侧 y 轴标题占用边距后，图例和横轴标题看起来偏左。
    """
    if rows == 1:
        left = 0.075
        right = 0.995
        bottom = 0.245
        top = 0.720
        return {
            "left": left,
            "right": right,
            "bottom": bottom,
            "top": top,
            "wspace": 0.300,
            "hspace": None,
            "plot_center": (left + right) / 2.0,
            "legend_y": 0.965,
            "xlabel_y": 0.052,
            "ylabel_x": 0.014,
        }

    left = 0.085
    right = 0.995
    bottom = 0.105
    top = 0.855
    return {
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
        "wspace": 0.280,
        "hspace": 0.430,
        "plot_center": (left + right) / 2.0,
        "legend_y": 0.985,
        "xlabel_y": 0.025,
        "ylabel_x": 0.018,
    }


def apply_compact_layout(fig, rows):
    """压缩无效留白，并让全局坐标标题相对绘图区居中。"""
    params = get_layout_params(rows)

    adjust_kwargs = {
        "left": params["left"],
        "right": params["right"],
        "bottom": params["bottom"],
        "top": params["top"],
        "wspace": params["wspace"],
    }
    if params["hspace"] is not None:
        adjust_kwargs["hspace"] = params["hspace"]

    fig.subplots_adjust(**adjust_kwargs)
    fig.supxlabel(
        "Predicted fake probability",
        fontsize=8.0,
        x=params["plot_center"],
        y=params["xlabel_y"],
        ha="center",
    )
    fig.supylabel(
        "Density",
        fontsize=8.0,
        x=params["ylabel_x"],
        ha="center",
    )


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
    fig_width=None,
    fig_height=None,
    font_scale=1.0,
):
    """绘制多个模型的视频级 Real/Fake 分数分布图。"""
    set_paper_style(dpi=dpi, font_scale=font_scale)

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

    rows, cols, figsize = get_layout(len(items), fig_width=fig_width, fig_height=fig_height)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = np.asarray(axes).reshape(-1)

    # 色盲友好且适合打印的低饱和配色；线型用于辅助区分。
    real_color = "#1F77B4"
    fake_color = "#C95C0A"

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
            linewidth=1.35,
            linestyle="-",
            label="Real",
        )
        ax.fill_between(
            centers,
            real_density,
            color=real_color,
            alpha=0.09,
            linewidth=0,
        )

        ax.plot(
            centers,
            fake_density,
            color=fake_color,
            linewidth=1.35,
            linestyle="--",
            label="Fake",
        )
        ax.fill_between(
            centers,
            fake_density,
            color=fake_color,
            alpha=0.08,
            linewidth=0,
        )

        if show_threshold:
            ax.axvline(
                threshold,
                color="0.35",
                linestyle=":",
                linewidth=0.80,
                zorder=0,
            )

        panel_label = f"({string.ascii_lowercase[ax_idx]})"
        ax.set_title(f"{panel_label} {item['title']}", pad=3.0)

        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.grid(True, linestyle=":", linewidth=0.42, alpha=0.32)

        ax.tick_params(axis="both", which="major", length=2.6, width=0.65, pad=1.8)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.70)

        metric_text = build_metric_text(labels, scores, metric_mode)
        if metric_text is not None:
            ax.text(
                0.035,
                0.93,
                metric_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=6.25 * font_scale,
                linespacing=0.98,
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "0.82",
                    "linewidth": 0.40,
                    "alpha": 0.90,
                },
            )

    layout_params = get_layout_params(rows)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(layout_params["plot_center"], layout_params["legend_y"]),
        handlelength=2.2,
        columnspacing=1.25,
        borderaxespad=0.0,
    )

    apply_compact_layout(fig, rows)

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        out_path = output_prefix.with_suffix(f".{fmt}")
        fig.savefig(
            out_path,
            dpi=dpi if fmt == "png" else None,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.018,
        )
        saved_paths.append(out_path)

    plt.close(fig)

    for out_path in saved_paths:
        print(f"[可视化] 已保存分数分布图: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="绘制论文风格的视频级 Real/Fake 分数分布图")

    parser.add_argument(
        "csv",
        nargs="+",
        help="一个或多个 video_scores CSV 文件，建议顺序为 w/o BGI, w/o SRAC, Full MB-ViT",
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
        "--fig_width",
        type=float,
        default=None,
        help="手动指定图宽，单位 inch；不设置时按子图数量自动选择。",
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=None,
        help="手动指定图高，单位 inch；不设置时按子图数量自动选择。",
    )
    parser.add_argument(
        "--font_scale",
        type=float,
        default=1.0,
        help="整体字体缩放系数，论文正文图建议 0.95--1.05。",
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
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            font_scale=args.font_scale,
        )
    except Exception as exc:
        print(f"[错误] 绘图失败: {exc}")
        raise


if __name__ == "__main__":
    main()
