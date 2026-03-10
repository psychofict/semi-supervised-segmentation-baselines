"""
Class Imbalance Study for Semi-Supervised Segmentation Baselines.

Analyzes the relationship between class frequency and per-class IoU for
UniMatch and ST++. Identifies which rare classes suffer most under
semi-supervised training, motivating FARCLUSS's frequency-adaptive
rebalancing strategy.

Author: Ebenezer Tarubinga, Korea University
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats
from tqdm import tqdm

IGNORE_INDEX: int = 255

VOC_CLASSES: List[str] = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "dining table", "dog", "horse",
    "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
]


def load_mask(path: str) -> np.ndarray:
    """Load a segmentation mask."""
    return np.array(Image.open(path), dtype=np.int32)


def compute_class_frequencies(
    gt_dir: str, num_classes: int
) -> np.ndarray:
    """Compute pixel frequency for each class across the dataset."""
    gt_path = Path(gt_dir)
    gt_files = sorted(gt_path.glob("*.png"))

    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    total_valid = 0

    for gf in tqdm(gt_files, desc="  Computing class frequencies"):
        gt = load_mask(str(gf))
        valid = gt != IGNORE_INDEX
        gt_valid = gt[valid]
        total_valid += len(gt_valid)
        for c in range(num_classes):
            pixel_counts[c] += np.sum(gt_valid == c)

    frequencies = pixel_counts.astype(np.float64) / max(total_valid, 1)
    return frequencies


def compute_per_class_iou(
    pred_dir: str, gt_dir: str, num_classes: int
) -> np.ndarray:
    """Compute per-class IoU across the dataset."""
    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)
    pred_files = sorted(pred_path.glob("*.png"))

    intersection = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)

    for pf in tqdm(pred_files, desc="  Computing per-class IoU"):
        gf = gt_path / pf.name
        if not gf.exists():
            continue

        pred = load_mask(str(pf))
        gt = load_mask(str(gf))
        valid = gt != IGNORE_INDEX

        pred_valid = pred[valid]
        gt_valid = gt[valid]

        for c in range(num_classes):
            pred_c = pred_valid == c
            gt_c = gt_valid == c
            intersection[c] += np.sum(pred_c & gt_c)
            union[c] += np.sum(pred_c | gt_c)

    ious = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if union[c] > 0:
            ious[c] = intersection[c] / union[c]
    return ious


def categorize_classes(
    frequencies: np.ndarray, class_names: List[str]
) -> Tuple[List[int], List[int], List[int]]:
    """Categorize classes into rare (< 1%), medium (1-5%), and frequent (> 5%)."""
    rare = [i for i, f in enumerate(frequencies) if f < 0.01]
    medium = [i for i, f in enumerate(frequencies) if 0.01 <= f < 0.05]
    frequent = [i for i, f in enumerate(frequencies) if f >= 0.05]
    return rare, medium, frequent


def plot_frequency_vs_iou(
    frequencies: np.ndarray,
    ious_dict: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: str,
) -> None:
    """Scatter plot of class frequency vs IoU for each method."""
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2"]
    markers = ["o", "s", "^", "D"]

    for idx, (method, ious) in enumerate(ious_dict.items()):
        nonzero = frequencies > 0
        log_freq = np.log10(frequencies[nonzero] + 1e-10)

        ax.scatter(
            log_freq, ious[nonzero],
            c=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            s=80, alpha=0.7, edgecolors="black", linewidths=0.5,
            label=method, zorder=5,
        )

        # Trendline
        valid_mask = (ious[nonzero] > 0) & np.isfinite(log_freq)
        if np.sum(valid_mask) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_freq[valid_mask], ious[nonzero][valid_mask]
            )
            x_line = np.linspace(log_freq[valid_mask].min(), log_freq[valid_mask].max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=colors[idx % len(colors)],
                    linestyle="--", linewidth=1.5, alpha=0.7)
            print(f"  {method}: r={r_value:.3f}, p={p_value:.4f}, slope={slope:.3f}")

    # Annotate class names
    nonzero = frequencies > 0
    log_freq = np.log10(frequencies[nonzero] + 1e-10)
    names_nz = [class_names[i] for i in range(len(class_names)) if nonzero[i]]
    first_ious = list(ious_dict.values())[0][nonzero]
    for i, name in enumerate(names_nz):
        ax.annotate(
            name, (log_freq[i], first_ious[i]),
            fontsize=6, ha="center", va="bottom",
            xytext=(0, 5), textcoords="offset points", alpha=0.8,
        )

    ax.set_xlabel("Class Frequency (log10 scale)", fontsize=12)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_title("Class Frequency vs. Per-Class IoU", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved frequency vs IoU plot to {save_path}")


def plot_radar_chart(
    ious_dict: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: str,
) -> None:
    """Create a radar/spider chart comparing per-class IoU across methods."""
    num_classes = len(class_names)
    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False).tolist()
    angles.append(angles[0])  # close the polygon

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"polar": True})
    colors = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2"]

    for idx, (method, ious) in enumerate(ious_dict.items()):
        values = ious.tolist()
        values.append(values[0])  # close the polygon
        ax.plot(angles, values, color=colors[idx % len(colors)],
                linewidth=2, label=method)
        ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names, fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)
    ax.set_title("Per-Class IoU Comparison (Radar Chart)",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved radar chart to {save_path}")


def plot_category_bar(
    ious_dict: Dict[str, np.ndarray],
    rare: List[int],
    medium: List[int],
    frequent: List[int],
    save_path: str,
) -> None:
    """Bar chart comparing mean IoU for rare, medium, and frequent classes."""
    categories = ["Rare (<1%)", "Medium (1-5%)", "Frequent (>5%)"]
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.8 / len(ious_dict)
    colors = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2"]

    for idx, (method, ious) in enumerate(ious_dict.items()):
        means = []
        for group in [rare, medium, frequent]:
            if len(group) > 0:
                means.append(np.mean(ious[group]))
            else:
                means.append(0.0)

        offset = (idx - len(ious_dict) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=method,
                      color=colors[idx % len(colors)],
                      edgecolor="black", linewidth=0.5)

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Mean IoU", fontsize=12)
    ax.set_title("IoU by Class Frequency Category", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved category bar chart to {save_path}")


def plot_iou_gap_bar(
    ious_dict: Dict[str, np.ndarray],
    frequencies: np.ndarray,
    class_names: List[str],
    save_path: str,
) -> None:
    """Bar chart showing per-class IoU gap between methods, sorted by frequency."""
    method_names = list(ious_dict.keys())
    if len(method_names) < 2:
        return

    ious_a = ious_dict[method_names[0]]
    ious_b = ious_dict[method_names[1]]
    gap = ious_b - ious_a

    sorted_idx = np.argsort(frequencies)
    sorted_gap = gap[sorted_idx]
    sorted_names = [class_names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors_bar = ["#4CAF50" if g > 0 else "#F44336" for g in sorted_gap]
    ax.bar(range(len(sorted_gap)), sorted_gap, color=colors_bar,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(f"IoU Gain ({method_names[1]} - {method_names[0]})", fontsize=12)
    ax.set_title(f"Per-Class IoU Difference (sorted by class frequency, rare -> frequent)",
                 fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved IoU gap chart to {save_path}")


def save_imbalance_csv(
    frequencies: np.ndarray,
    ious_dict: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: str,
) -> None:
    """Save class frequency and IoU data to CSV."""
    methods = list(ious_dict.keys())
    with open(save_path, "w") as f:
        header = "class,frequency," + ",".join(f"iou_{m}" for m in methods)
        f.write(header + "\n")
        for i, name in enumerate(class_names):
            row = f"{name},{frequencies[i]:.6f}"
            for m in methods:
                row += f",{ious_dict[m][i]:.4f}"
            f.write(row + "\n")
    print(f"  Saved imbalance CSV to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Class imbalance study for semi-supervised segmentation."
    )
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Parent directory containing method subdirectories.")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Ground truth mask directory.")
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    class_names = VOC_CLASSES[:args.num_classes]

    fig_dir = os.path.join(args.output_dir, "figures")
    tab_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    # Step 1: Compute class frequencies from ground truth
    print("\n" + "=" * 60)
    print("Computing class frequencies from ground truth")
    print("=" * 60)
    frequencies = compute_class_frequencies(args.gt_dir, args.num_classes)

    for i, name in enumerate(class_names):
        print(f"  {name:>15s}: {frequencies[i]:.4%}")

    # Categorize
    rare, medium, frequent = categorize_classes(frequencies, class_names)
    print(f"\n  Rare classes ({len(rare)}): {[class_names[i] for i in rare]}")
    print(f"  Medium classes ({len(medium)}): {[class_names[i] for i in medium]}")
    print(f"  Frequent classes ({len(frequent)}): {[class_names[i] for i in frequent]}")

    # Step 2: Compute per-class IoU for each method
    ious_dict: Dict[str, np.ndarray] = {}
    methods_dirs = {
        "ST++": os.path.join(args.pred_dir, "stpp", "pseudo_labels"),
        "UniMatch": os.path.join(args.pred_dir, "unimatch", "pseudo_labels"),
    }

    for method_name, method_pred in methods_dirs.items():
        if Path(method_pred).exists():
            print(f"\n{'='*60}")
            print(f"Computing per-class IoU: {method_name}")
            print(f"{'='*60}")
            ious = compute_per_class_iou(method_pred, args.gt_dir, args.num_classes)
            ious_dict[method_name] = ious
            miou = np.mean(ious[ious > 0])
            print(f"  Mean IoU: {miou:.4f}")
            for i, name in enumerate(class_names):
                print(f"    {name:>15s}: {ious[i]:.4f}")
        else:
            print(f"  WARNING: {method_pred} does not exist, skipping {method_name}")

    if len(ious_dict) == 0:
        print("  No method predictions found. Exiting.")
        return

    # Step 3: Generate visualizations
    print(f"\n{'='*60}")
    print("Generating visualizations")
    print(f"{'='*60}")

    plot_frequency_vs_iou(
        frequencies, ious_dict, class_names,
        os.path.join(fig_dir, "class_frequency_vs_iou.png"),
    )
    plot_radar_chart(
        ious_dict, class_names,
        os.path.join(fig_dir, "per_class_iou_radar.png"),
    )
    plot_category_bar(
        ious_dict, rare, medium, frequent,
        os.path.join(fig_dir, "iou_by_frequency_category.png"),
    )

    if len(ious_dict) >= 2:
        plot_iou_gap_bar(
            ious_dict, frequencies, class_names,
            os.path.join(fig_dir, "iou_gap_by_frequency.png"),
        )

    save_imbalance_csv(
        frequencies, ious_dict, class_names,
        os.path.join(tab_dir, "class_imbalance_results.csv"),
    )

    # Summary for FARCLUSS motivation
    print(f"\n{'='*60}")
    print("KEY FINDING FOR FARCLUSS MOTIVATION:")
    for method, ious in ious_dict.items():
        if len(rare) > 0 and len(frequent) > 0:
            rare_miou = np.mean(ious[rare]) if len(rare) > 0 else 0
            freq_miou = np.mean(ious[frequent]) if len(frequent) > 0 else 0
            gap = freq_miou - rare_miou
            print(f"  {method}: Rare mIoU={rare_miou:.3f}, Frequent mIoU={freq_miou:.3f}, Gap={gap:.3f}")
    print("  Rare classes suffer disproportionately under semi-supervised training.")
    print("  This motivates FARCLUSS's frequency-adaptive rebalancing approach.")
    print("=" * 60)
    print("\nClass imbalance study complete.")


if __name__ == "__main__":
    main()
