#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 MB-ViT 成功/失败案例可视化图（正式论文版，可直接替换）。

设计目标：
1. 输出优先面向 LaTeX 论文插图，推荐 PDF/SVG，PNG 仅作预览；
2. 术语与论文消融表保持一致，默认使用 w/o BGI、w/o SRAC、Full MB-ViT；
3. 使用独立表头行，保证列标题居中；
4. 压缩列距和行距，减少不必要留白；
5. 分数面板使用克制的灰蓝配色和浅色背景轨道，避免过度装饰；
6. 默认仅在最后一行显示 Fake score 的 0/1 端点和轴名，降低重复信息；
7. 自动裁掉代表输入图中的近白色空白边，并采用 cover 方式铺满目标区域，减少上下白边；
8. 支持同一样本对比：每一行固定同一个视频样本，对比不同模型变体在同一输入上的视频级分数；
9. 加宽左侧案例说明列并统一正文文字颜色/字号，避免左侧标签截断和左右文字风格不一致。

输入 CSV 至少需要字段：
case_type, video_key, label, full_score, full_pred, top_frame_paths

可选字段：
wo_bgi_score, wo_region_score, reason

说明：
- wo_region_score 在当前论文中对应 w/o SRAC，即移除 RAE 中的尺度--区域感知校准模块。
- 为兼容旧实验文件名，脚本内部仍读取 wo_region_score 字段，但图中默认显示为 w/o SRAC。
- 如果你的 CSV 中 wo_region_score 确实表示完整移除 RAE，可在命令行使用：
  --region_label "w/o RAE"
  --region_label_full "w/o RAE"
"""

import argparse
import csv
import os
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps


# =========================
# 模型分数列与默认显示标签
# =========================
MODEL_SCORE_COLUMNS = ["wo_bgi_score", "wo_region_score", "full_score"]
MODEL_COLORS = ["#B9C0CB", "#8FA3B5", "#2E557A"]
MODEL_TRACK_COLOR = "#EEF1F4"
TEXT_COLOR = "#111111"
MUTED_TEXT_COLOR = "#4A4A4A"
BORDER_COLOR = "#D8DDE3"


# =========================
# 基础工具函数
# =========================
def set_paper_style(dpi: int, font_scale: float = 1.0) -> None:
    """设置适合论文图的 Matplotlib 全局风格。"""
    base = float(font_scale)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.7,
        "axes.titlesize": 8.0 * base,
        "axes.labelsize": 7.2 * base,
        "xtick.labelsize": 6.6 * base,
        "ytick.labelsize": 6.6 * base,
        "legend.fontsize": 6.6 * base,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def safe_float(value, default: float = 0.0) -> float:
    """安全转换浮点数。"""
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default: int = 0) -> int:
    """安全转换整数。"""
    try:
        return int(float(value))
    except Exception:
        return default


def label_name(value) -> str:
    """将 0/1 标签映射为 Real/Fake。"""
    return "Fake" if safe_int(value) == 1 else "Real"


def pred_name(row: Dict[str, str]) -> str:
    """获取 Full 模型预测的类别名。"""
    return label_name(row.get("full_pred", row.get("pred", "0")))


def read_cases(case_csv: str, max_cases: int = 4) -> List[Dict[str, str]]:
    """读取案例 CSV。"""
    with open(case_csv, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"案例 CSV 为空: {case_csv}")

    required = {"case_type", "video_key", "label", "full_score", "top_frame_paths"}
    missing = required - set(rows[0].keys())
    if missing:
        raise RuntimeError(f"案例 CSV 缺少必要字段: {sorted(missing)}")

    return rows[:max_cases]


def apply_crop(img: Image.Image, crop_mode: str) -> Image.Image:
    """根据模式裁剪输入图像。"""
    if crop_mode == "none":
        return img

    width, height = img.size
    if width <= 1 or height <= 1:
        return img

    if crop_mode == "bottom_half":
        # 适用于早期融合 A/V 输入：保留下半部分视频帧区域。
        top = int(height * 0.50)
        return img.crop((0, top, width, height))

    if crop_mode == "center":
        # 中心裁剪，适合已经是人脸或口部区域的图片。
        crop_w = int(width * 0.74)
        crop_h = int(height * 0.74)
        left = max((width - crop_w) // 2, 0)
        top = max((height - crop_h) // 2, 0)
        return img.crop((left, top, left + crop_w, top + crop_h))

    return img


def trim_near_white_margin(img: Image.Image,
                           white_thr: int = 245,
                           pad: int = 0) -> Image.Image:
    """自动裁掉四周接近白色的空白边。"""
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[0] <= 1 or arr.shape[1] <= 1:
        return img

    # 只要某个像素任一通道低于阈值，就认为其属于“非白区域”。
    mask = np.any(arr < white_thr, axis=2)
    if not mask.any():
        return img

    ys, xs = np.where(mask)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1

    if pad > 0:
        y1 = max(0, y1 - pad)
        y2 = min(arr.shape[0], y2 + pad)
        x1 = max(0, x1 - pad)
        x2 = min(arr.shape[1], x2 + pad)

    return img.crop((x1, y1, x2, y2))


def resize_cover(img: Image.Image,
                 size: Tuple[int, int]) -> Image.Image:
    """按 cover 方式缩放并中心裁切，确保图像完全铺满目标区域。"""
    target_w, target_h = size
    src_w, src_h = img.size

    if src_w <= 0 or src_h <= 0:
        return Image.new("RGB", size, color=(248, 248, 248))

    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)

    return img.crop((left, top, left + target_w, top + target_h))


def read_image_safe(path: str,
                    size: Tuple[int, int] = (116, 76),
                    crop_mode: str = "none",
                    add_border: bool = True,
                    trim_white: bool = True,
                    white_thr: int = 245,
                    white_pad: int = 0) -> Image.Image:
    """安全读取图片；自动裁白边并铺满目标区域。"""
    try:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(path)

        img = Image.open(path).convert("RGB")
        img = apply_crop(img, crop_mode)

        if trim_white:
            img = trim_near_white_margin(img, white_thr=white_thr, pad=white_pad)

        img = resize_cover(img, size)

        if add_border:
            img = ImageOps.expand(img, border=1, fill=(220, 224, 230))
            img = img.resize(size, Image.Resampling.LANCZOS)
        return img
    except Exception as exc:
        print(f"[绘图警告] 无法读取图片: {path}，原因: {exc}")
        return Image.new("RGB", size, color=(248, 248, 248))


def make_frame_strip(paths_text: str,
                     image_size: Tuple[int, int] = (116, 76),
                     crop_mode: str = "none",
                     frame_gap: int = 3,
                     add_border: bool = True,
                     trim_white: bool = True,
                     white_thr: int = 245,
                     white_pad: int = 0) -> Image.Image:
    """把 top-k 代表输入横向拼接成一条。"""
    paths = str(paths_text).split("|") if paths_text else []
    paths = [p.strip() for p in paths if p.strip()]
    images = [
        read_image_safe(
            p,
            image_size,
            crop_mode,
            add_border=add_border,
            trim_white=trim_white,
            white_thr=white_thr,
            white_pad=white_pad,
        )
        for p in paths[:3]
    ]

    while len(images) < 3:
        images.append(Image.new("RGB", image_size, color=(248, 248, 248)))

    width = image_size[0] * 3 + frame_gap * 2
    height = image_size[1]
    canvas = Image.new("RGB", (width, height), color="white")

    for idx, img in enumerate(images):
        canvas.paste(img, (idx * (image_size[0] + frame_gap), 0))

    return canvas


def auto_input_title(crop_mode: str) -> str:
    """根据裁剪模式自动生成输入列标题。"""
    if crop_mode == "none":
        return "Representative A/V inputs"
    return "Selected video frames"


def output_paths(output: str, formats: Iterable[str]) -> List[Path]:
    """根据输出前缀和格式生成目标路径。"""
    output_path = Path(output)
    base = output_path.with_suffix("") if output_path.suffix else output_path
    base.parent.mkdir(parents=True, exist_ok=True)
    return [base.with_suffix(f".{fmt.lower().lstrip('.')}" ) for fmt in formats]


def normalize_model_label(label: str) -> str:
    """统一模型标签，避免旧术语进入正式论文图。"""
    text = str(label).strip()
    key = text.lower().replace(" ", "").replace("_", "").replace("-", "")
    old_region_names = {
        "w/oreg.pe+se",
        "w/oregionpe+se",
        "w/oregionpe/se",
        "withoutregionpe+se",
        "withoutregionpe/se",
        "regionpe+seablation",
        "regionpeseablation",
    }
    if key in old_region_names or "regionpe" in key or "reg.pe" in key:
        return "w/o SRAC"
    return text


def build_model_labels(args: argparse.Namespace) -> List[str]:
    """根据命令行参数生成模型标签。"""
    if args.model_label_style == "full":
        labels = [args.bgi_label, args.region_label_full, args.full_label]
    else:
        labels = [args.bgi_label, args.region_label, args.full_label]
    return [normalize_model_label(label) for label in labels]


# =========================
# 绘制函数
# =========================
def draw_header_cell(ax, title: str, font_scale: float) -> None:
    """绘制独立标题行中的列标题。"""
    ax.axis("off")
    ax.text(
        0.5,
        0.52,
        title,
        ha="center",
        va="center",
        fontsize=7.2 * font_scale,
        fontweight="semibold",
        color=TEXT_COLOR,
    )


def draw_case_text(ax,
                   row: Dict[str, str],
                   font_scale: float,
                   show_id: bool) -> None:
    """绘制左侧案例说明。"""
    ax.axis("off")

    case_type = row.get("case_type", "Case")
    video_key = row.get("video_key", "")
    short_key = os.path.basename(str(video_key))[:30]

    lines = [
        case_type,
        f"GT: {label_name(row.get('label', 0))}",
        f"Pred: {pred_name(row)}",
    ]
    if show_id and short_key:
        lines.append(f"ID: {short_key}")

    ax.text(
        0.01,
        0.50,
        "\n".join(lines),
        ha="left",
        va="center",
        fontsize=6.6 * font_scale,
        linespacing=1.16,
        fontweight="semibold",
        color=TEXT_COLOR,
        clip_on=False,
    )


def draw_frames(ax,
                row: Dict[str, str],
                crop_mode: str,
                image_size: Tuple[int, int],
                frame_gap: int,
                add_frame_border: bool,
                trim_white: bool,
                white_thr: int,
                white_pad: int) -> None:
    """绘制代表输入图条带。"""
    ax.axis("off")
    strip = make_frame_strip(
        row.get("top_frame_paths", ""),
        image_size=image_size,
        crop_mode=crop_mode,
        frame_gap=frame_gap,
        add_border=add_frame_border,
        trim_white=trim_white,
        white_thr=white_thr,
        white_pad=white_pad,
    )
    ax.imshow(strip)


def draw_scores_panel(ax,
                      row: Dict[str, str],
                      model_labels: Sequence[str],
                      font_scale: float,
                      show_score_axis: bool,
                      show_score_grid: bool = False,
                      show_score_track: bool = True) -> None:
    """绘制视频级 fake score 条形图面板。"""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.16, 3.02)
    ax.axis("off")

    scores = [safe_float(row.get(col, 0.0)) for col in MODEL_SCORE_COLUMNS]
    y_positions = [2.35, 1.42, 0.49]

    label_x = 0.02
    bar_x = 0.55
    bar_w_max = 0.38
    bar_h = 0.36

    if show_score_grid:
        for tick in np.linspace(0.0, 1.0, 6):
            x = bar_x + tick * bar_w_max
            ax.plot([x, x], [0.12, 2.68], color="#E6EAF0", linewidth=0.55, zorder=0)

    for label, score, color, y in zip(model_labels, scores, MODEL_COLORS, y_positions):
        score = min(max(score, 0.0), 1.0)
        is_full = label.lower().startswith("full")

        ax.text(
            label_x,
            y,
            label,
            ha="left",
            va="center",
            fontsize=6.6 * font_scale,
            fontweight="bold" if is_full else "semibold",
            color=TEXT_COLOR,
        )

        if show_score_track:
            ax.add_patch(
                Rectangle(
                    (bar_x, y - bar_h / 2.0),
                    bar_w_max,
                    bar_h,
                    facecolor=MODEL_TRACK_COLOR,
                    edgecolor="none",
                    zorder=1,
                )
            )

        ax.add_patch(
            Rectangle(
                (bar_x, y - bar_h / 2.0),
                score * bar_w_max,
                bar_h,
                facecolor=color,
                edgecolor="none",
                zorder=2,
            )
        )

        value_x = min(bar_x + score * bar_w_max + 0.010, 0.95)
        value_ha = "left"
        if score > 0.86:
            value_x = max(bar_x + score * bar_w_max - 0.010, bar_x + 0.03)
            value_ha = "right"

        ax.text(
            value_x,
            y,
            f"{score:.3f}",
            ha=value_ha,
            va="center",
            fontsize=6.6 * font_scale,
            fontweight="bold" if is_full else "normal",
            color=TEXT_COLOR,
        )

    if show_score_axis:
        ax.text(bar_x, -0.02, "0", ha="center", va="bottom", fontsize=6.3 * font_scale, color=TEXT_COLOR)
        ax.text(bar_x + bar_w_max, -0.02, "1", ha="center", va="bottom", fontsize=6.3 * font_scale, color=TEXT_COLOR)
        ax.text(bar_x + bar_w_max / 2.0, -0.10, "Fake score", ha="center", va="top", fontsize=6.6 * font_scale, color=TEXT_COLOR)


def normalize_reason_text(reason: str, case_type: str) -> str:
    """规范右侧 Observation 文本，使其更适合论文图。"""
    reason = str(reason or "").strip()
    lower = reason.lower()

    # 当前论文图中的失败样本说明宜更克制，避免口语化的长句。
    if "low visual quality" in lower and "unusual pose" in lower and ("domain bias" in lower or "domain shift" in lower):
        return "Low visual quality / unusual pose / possible domain shift"

    # 对自动生成但未加前缀的失败说明，补充谨慎前缀。
    if ("failure" in case_type.lower() or "hard" in case_type.lower()) and reason:
        if not lower.startswith("possible") and not lower.startswith("low visual quality"):
            return "Possible reason: " + reason

    return reason


def draw_reason(ax,
                row: Dict[str, str],
                font_scale: float,
                note_wrap_width: int) -> None:
    """绘制右侧说明。

    为保证整张图的文字风格一致，这里与左侧案例说明统一使用
    相同的颜色、字号和字重，避免 Observation 看起来像“另一套字体”。
    """
    ax.axis("off")

    reason = normalize_reason_text(row.get("reason", ""), row.get("case_type", ""))
    wrapped = "\n".join(textwrap.wrap(reason, width=note_wrap_width))
    ax.text(
        0.5,
        0.50,
        wrapped,
        ha="center",
        va="center",
        fontsize=6.6 * font_scale,
        linespacing=1.16,
        fontweight="semibold",
        color=TEXT_COLOR,
    )


# =========================
# 主绘图函数
# =========================
def plot_cases(rows: List[Dict[str, str]], args: argparse.Namespace) -> None:
    """绘制最终论文图。"""
    set_paper_style(dpi=args.dpi, font_scale=args.font_scale)

    model_labels = build_model_labels(args)
    nrows = len(rows)
    fig_height = max(args.row_height * nrows + args.header_height + (0.18 if args.show_title else 0.02), 3.10)

    fig = plt.figure(figsize=(args.fig_width, fig_height), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=nrows + 1,
        ncols=4,
        height_ratios=[args.header_height] + [1.0] * nrows,
        # 列宽比例：加宽左侧案例说明列，避免 Failure / Hard Case 等长标签被截断。
        width_ratios=[1.24, 3.26, 2.08, 1.42],
        left=args.left_margin,
        right=args.right_margin,
        top=args.top_margin if not args.show_title else min(args.top_margin - 0.035, 0.94),
        bottom=args.bottom_margin,
        wspace=args.col_gap,
        hspace=args.row_gap,
    )

    input_title = args.input_title if args.input_title else auto_input_title(args.crop_mode)
    draw_header_cell(fig.add_subplot(grid[0, 0]), "", args.font_scale)
    draw_header_cell(fig.add_subplot(grid[0, 1]), input_title, args.font_scale)
    draw_header_cell(fig.add_subplot(grid[0, 2]), "Video-level fake score", args.font_scale)
    draw_header_cell(fig.add_subplot(grid[0, 3]), args.note_title, args.font_scale)

    for row_idx, row in enumerate(rows):
        grid_row = row_idx + 1
        if args.score_axis_mode == "all":
            show_score_axis = True
        elif args.score_axis_mode == "none":
            show_score_axis = False
        else:
            show_score_axis = (row_idx == nrows - 1)

        draw_case_text(fig.add_subplot(grid[grid_row, 0]), row, args.font_scale, args.show_id)
        draw_frames(
            fig.add_subplot(grid[grid_row, 1]),
            row,
            args.crop_mode,
            image_size=(args.frame_width, args.frame_height),
            frame_gap=args.frame_gap,
            add_frame_border=not args.no_frame_border,
            trim_white=not args.no_trim_white,
            white_thr=args.white_thr,
            white_pad=args.white_pad,
        )
        draw_scores_panel(
            fig.add_subplot(grid[grid_row, 2]),
            row,
            model_labels,
            args.font_scale,
            show_score_axis=show_score_axis,
            show_score_grid=args.show_score_grid,
            show_score_track=not args.no_score_track,
        )
        draw_reason(
            fig.add_subplot(grid[grid_row, 3]),
            row,
            args.font_scale,
            note_wrap_width=args.note_wrap_width,
        )

    if args.show_title:
        fig.suptitle(
            "Qualitative Case Analysis of MB-ViT",
            y=0.987,
            fontsize=8.8 * args.font_scale,
            fontweight="bold",
        )

    paths = output_paths(args.output, args.formats)
    for path in paths:
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=args.pad_inches)
        print(f"[绘图] 已保存: {path}")

    plt.close(fig)


# =========================
# 命令行参数
# =========================
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="绘制 MB-ViT 成功/失败案例图（正式论文版）。")
    parser.add_argument("--case_csv", required=True, help="select_qualitative_cases.py 输出的案例 CSV")
    parser.add_argument("--output", required=True, help="输出图像前缀或完整路径")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png", "svg"], help="输出格式，论文优先使用 pdf")
    parser.add_argument("--dpi", type=int, default=600, help="输出分辨率，PNG 预览建议 600")
    parser.add_argument("--max_cases", type=int, default=4, help="最多展示多少个案例")

    parser.add_argument(
        "--crop_mode",
        choices=["none", "bottom_half", "center"],
        default="none",
        help="none 显示完整 A/V 输入；bottom_half 只显示下半视频帧；center 做中心裁剪。",
    )
    parser.add_argument(
        "--model_label_style",
        choices=["short", "full"],
        default="short",
        help="short 使用紧凑标签；full 使用完整标签。",
    )
    parser.add_argument("--bgi_label", default="w/o BGI", help="第一组模型标签")
    parser.add_argument("--region_label", default="w/o SRAC", help="第二组模型短标签，当前论文默认对应 w/o SRAC")
    parser.add_argument("--region_label_full", default="w/o SRAC", help="第二组模型完整标签")
    parser.add_argument("--full_label", default="Full MB-ViT", help="完整模型标签")
    parser.add_argument("--note_title", default="Observation", help="右侧说明列标题")

    parser.add_argument("--show_title", action="store_true", help="图内显示大标题；正文通常不建议打开。")
    parser.add_argument("--show_id", action="store_true", help="显示样本 ID；正文通常不建议打开。")
    parser.add_argument("--input_title", default="", help="手动指定输入列标题；为空则自动生成。")
    parser.add_argument("--show_score_grid", action="store_true", help="显示分数面板中的竖向网格线；正文图默认不显示。")
    parser.add_argument("--no_score_track", action="store_true", help="不显示分数条的浅色背景轨道。")
    parser.add_argument("--no_frame_border", action="store_true", help="不显示代表输入图片的浅灰边框。")

    parser.add_argument("--no_trim_white", action="store_true", help="不自动裁掉代表输入图片的近白空白边。")
    parser.add_argument("--white_thr", type=int, default=245, help="近白空白边阈值，越小裁剪越激进。")
    parser.add_argument("--white_pad", type=int, default=0, help="裁掉白边后额外保留的像素边距。")

    parser.add_argument("--font_scale", type=float, default=0.95, help="整体字体缩放系数")
    parser.add_argument("--fig_width", type=float, default=7.25, help="整张图宽度，单位 inch；适合 one-column IEEE 宽页或双栏跨栏")
    parser.add_argument("--row_height", type=float, default=0.76, help="每一行案例的高度，单位 inch")
    parser.add_argument("--frame_width", type=int, default=116, help="单张代表输入的宽度，单位像素")
    parser.add_argument("--frame_height", type=int, default=76, help="单张代表输入的高度，单位像素")
    parser.add_argument("--frame_gap", type=int, default=3, help="三张代表输入之间的间距，单位像素")
    parser.add_argument("--note_wrap_width", type=int, default=28, help="右侧说明文字自动换行宽度")

    parser.add_argument("--left_margin", type=float, default=0.040, help="左边距，占画布比例")
    parser.add_argument("--right_margin", type=float, default=0.995, help="右边距，占画布比例")
    parser.add_argument("--top_margin", type=float, default=0.975, help="上边距，占画布比例")
    parser.add_argument("--bottom_margin", type=float, default=0.055, help="下边距，占画布比例")
    parser.add_argument("--col_gap", type=float, default=0.035, help="列间距")
    parser.add_argument("--row_gap", type=float, default=0.075, help="行间距")
    parser.add_argument("--header_height", type=float, default=0.145, help="独立标题行高度比例")
    parser.add_argument("--pad_inches", type=float, default=0.012, help="保存图片时的外边距，单位 inch")
    parser.add_argument(
        "--score_axis_mode",
        choices=["last", "all", "none"],
        default="last",
        help="Fake score 轴显示方式：last 仅最后一行显示；all 每行显示；none 不显示。",
    )
    # 兼容旧脚本参数名。
    parser.add_argument(
        "--x_label_mode",
        choices=["last", "all", "none"],
        default=None,
        help="兼容旧版本参数；若设置，将覆盖 --score_axis_mode。",
    )

    args = parser.parse_args()
    if args.x_label_mode is not None:
        args.score_axis_mode = args.x_label_mode
    return args


def main() -> None:
    """主函数。"""
    args = parse_args()
    rows = read_cases(args.case_csv, max_cases=args.max_cases)
    plot_cases(rows, args)


if __name__ == "__main__":
    main()
