"""
将 doc/curve 下多个 CSV 文件(训练记录)拼接成连续的训练曲线图。
用法: conda run -n yolov8 python script/plot_curve.py
"""
import os
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 尝试使用支持中文的字体
_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "DengXian",
    "FangSong",
    "KaiTi",
    "SimSun",
]
CN_FONT = None
for _f in _FONT_CANDIDATES:
    try:
        CN_FONT = FontProperties(family=_f)
        # 验证是否可用
        if CN_FONT.get_name():
            break
    except Exception:
        continue
CURVE_DIR = os.path.join(PROJECT_ROOT, "doc/curve")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "doc")

# 要绘制的指标列（8 个），以及其中文标签
METRICS = [
    ("train/box_loss",       "盒损失"),
    ("train/cls_loss",       "类别损失"),
    ("train/dfl_loss",       "DFL 损失"),
    ("metrics/precision(B)", "精确率"),
    ("metrics/recall(B)",    "召回率"),
    ("metrics/mAP50(B)",     "mAP@50"),
    ("metrics/mAP50-95(B)",  "mAP@50-95"),
    ("val/box_loss",         "验证盒损失"),
]

# 布局：4 行 x 2 列
LAYOUT = (4, 2)
FIG_SIZE = (14, 16)


def read_csvs(directory):
    """读取目录下所有 .csv 文件，按文件名数字排序，拼接为连续 epoch 的数组。"""
    csv_files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".csv")],
        key=lambda x: int(Path(x).stem),
    )
    if not csv_files:
        print(f"[Error] No CSV files found in {directory}")
        sys.exit(1)

    all_data = []
    epoch_offset = 0

    for fname in csv_files:
        fpath = os.path.join(directory, fname)
        with open(fpath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # 清洗列名（去掉空格）
            rows = []
            for row in reader:
                cleaned = {}
                for k, v in row.items():
                    k_clean = k.strip()
                    v_clean = v.strip()
                    if k_clean == "epoch":
                        cleaned[k_clean] = int(v_clean) + epoch_offset
                    else:
                        cleaned[k_clean] = float(v_clean)
                rows.append(cleaned)
            if rows:
                epoch_offset = rows[-1]["epoch"]
            all_data.extend(rows)

    print(f"Loaded {len(csv_files)} CSV files, {len(all_data)} epochs total")
    return all_data


def plot_metrics(data):
    """绘制 8 个指标的连续曲线图。"""
    nrows, ncols = LAYOUT
    fig, axes = plt.subplots(nrows, ncols, figsize=FIG_SIZE)
    epochs = [row["epoch"] for row in data]

    for idx, (col_name, label_cn) in enumerate(METRICS):
        ax = axes[idx // ncols][idx % ncols]
        values = [row[col_name] for row in data]
        ax.plot(epochs, values, linewidth=1.0, color="#0072B2")
        ax.set_xlabel("Epoch")
        if CN_FONT:
            ax.set_ylabel(label_cn, fontproperties=CN_FONT)
            ax.set_title(f"{label_cn} ({col_name})", fontproperties=CN_FONT, fontsize=11)
        else:
            ax.set_ylabel(label_cn)
            ax.set_title(f"{label_cn} ({col_name})", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 隐藏多余的子图（如果指标数量不满布局格子数）
    for idx in range(len(METRICS), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Training Curves (Concatenated)", fontsize=14, y=0.98)
    plt.tight_layout()
    return fig


def main():
    data = read_csvs(CURVE_DIR)
    if not data:
        print("[Error] No data loaded.")
        sys.exit(1)

    fig = plot_metrics(data)
    out_path = os.path.join(OUTPUT_DIR, "curve_concat.png")
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
