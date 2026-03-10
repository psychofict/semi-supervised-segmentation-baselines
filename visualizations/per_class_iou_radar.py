"""
Per-Class IoU Radar Chart Visualization.

Creates radar/spider charts comparing per-class IoU across semi-supervised
segmentation methods on Pascal VOC (21 classes). Supports multiple methods
and highlights which classes each method handles best or worst.

Author: Ebenezer Tarubinga, Korea University
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IGNORE_INDEX: int = 255

VOC_CLASSES: List[str] = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "dining table", "dog", "horse",
    "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
]


def load_mask(path: str) -> np.ndarray:
    """Load a segmentation mask."""
    return np.array(Image.open(path), dtype=np.int32)


def compute_per_class_iou_from_dirs(
    pred_dir: str, gt_dir: str, num_classes: int
) -> np.ndarray:
    """Compute per-class IoU from prediction and ground truth directories."""
    from tqdm import tqdm
    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)
    pred_files = sorted(pred_path.glob("*.png"))

    intersection = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)

    for pf in tqdm(pred_files, desc="  Computing IoU"):
        gf = gt_path / pf.name
        if not gf.exists():
            continue
        pred = load_mask(str(pf))
        gt = load_mask(str(gf))
        valid = gt != IGNORE_INDEX
        pred_v = pred[valid]
        gt_v = gt[valid]

        for c in range(num_classes):
            pred_c = pred_v == c
            gt_c = gt_v == c
            intersection[c] += np.sum(pred_c & gt_c)
            union[c] += np.sum(pred_c | gt_c)

    ious = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if union[c] > 0:
            ious[c] = intersection[c] / union[c]
    return ious


def load_iou_from_csv(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Load per-class IoU from a CSV file.

    Expected format: method,class,iou or columns containing 'iou_<method>'.
    """
    methods_ious: Dict[str, Dict[str, float]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # Check if it's the class_imbalance format
        iou_cols = [h for h in headers if h.startswith("iou_")]
        if iou_cols:
            class_names_found: List[str] = []
            rows = list(reader)
            for row in rows:
                class_names_found.append(row.get("class", ""))
            for col in iou_cols:
                method = col.replace("iou_", "")
                ious = np.zeros(len(rows), dtype=np.float64)
                for i, row in enumerate(rows):
                    try:
                        ious[i] = float(row[col])
                    except (ValueError, KeyError):
                        ious[i] = 0.0
                methods_ious[method] = {class_names_found[i]: ious[i]
                                        for i in range(len(rows))}
        else:
            for row in reader:
                method = row.get("method", "unknown")
                cls_name = row.get("class", "")
                iou_val = 0.0
                for key in ["iou", "recall", "precision"]:
                    if key in row:
                        try:
                            iou_val = float(row[key])
                        except ValueError:
                            pass
                        break
                if method not in methods_ious:
                    methods_ious[method] = {}
                methods_ious[method][cls_name] = iou_val

    # Convert to arrays
    result: Dict[str, np.ndarray] = {}
    for method, cls_dict in methods_ious.items():
        ious = np.zeros(len(VOC_CLASSES), dtype=np.float64)
        for i, name in enumerate(VOC_CLASSES):
            ious[i] = cls_dict.get(name, 0.0)
        result[method] = ious

    return result


def plot_radar_chart(
    ious_dict: Dict[str, np.ndarray],
    class_names: List[str],
    title: str,
    save_path: str,
    fill_alpha: float = 0.1,
    exclude_background: bool = True,
) -> None:
    """
    Create a radar/spider chart comparing per-class IoU across methods.
    """
    if exclude_background and len(class_names) > 1:
        display_names = class_names[1:]
        display_ious = {m: v[1:] for m, v in ious_dict.items()}
    else:
        display_names = class_names
        display_ious = ious_dict

    num_vars = len(display_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={"polar": True})

    colors = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2", "#F57C00"]
    linestyles = ["-", "--", "-.", ":", "-"]
    markers = ["o", "s", "^", "D", "v"]

    for idx, (method, ious) in enumerate(display_ious.items()):
        values = ious.tolist()
        values.append(values[0])

        color = colors[idx % len(colors)]
        ax.plot(angles, values, color=color,
                linewidth=2.5, linestyle=linestyles[idx % len(linestyles)],
                marker=markers[idx % len(markers)], markersize=5,
                label=f"{method} (mean: {np.mean(ious):.1%})")
        ax.fill(angles, values, color=color, alpha=fill_alpha)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_names, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved radar chart to {save_path}")


def plot_sorted_bar_chart(
    ious_dict: Dict[str, np.ndarray],
    class_names: List[str],
    title: str,
    save_path: str,
    exclude_background: bool = True,
) -> None:
    """Create horizontal bar chart sorted by IoU of the first method."""
    if exclude_background and len(class_names) > 1:
        display_names = class_names[1:]
        display_ious = {m: v[1:] for m, v in ious_dict.items()}
    else:
        display_names = class_names
        display_ious = ious_dict

    first_method = list(display_ious.keys())[0]
    sort_idx = np.argsort(display_ious[first_method])
    sorted_names = [display_names[i] for i in sort_idx]

    num_methods = len(display_ious)
    fig, ax = plt.subplots(figsize=(12, max(8, len(display_names) * 0.4)))
    colors = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2", "#F57C00"]
    bar_height = 0.8 / num_methods

    for idx, (method, ious) in enumerate(display_ious.items()):
        sorted_vals = ious[sort_idx]
        positions = np.arange(len(sorted_names)) + (idx - num_methods / 2 + 0.5) * bar_height
        ax.barh(positions, sorted_vals, height=bar_height,
                color=colors[idx % len(colors)],
                edgecolor="black", linewidth=0.3,
                label=method)

    ax.set_yticks(np.arange(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel("IoU", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 1.0)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved sorted bar chart to {save_path}")


def plot_iou_difference_chart(
    ious_dict: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: str,
    exclude_background: bool = True,
) -> None:
    """Plot IoU differences between methods to highlight relative strengths."""
    methods = list(ious_dict.keys())
    if len(methods) < 2:
        return

    if exclude_background and len(class_names) > 1:
        names = class_names[1:]
        ious_a = ious_dict[methods[0]][1:]
        ious_b = ious_dict[methods[1]][1:]
    else:
        names = class_names
        ious_a = ious_dict[methods[0]]
        ious_b = ious_dict[methods[1]]

    diff = ious_b - ious_a
    sort_idx = np.argsort(diff)
    sorted_diff = diff[sort_idx]
    sorted_names = [names[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(12, max(8, len(names) * 0.4)))
    bar_colors = ["#4CAF50" if d > 0 else "#F44336" for d in sorted_diff]
    ax.barh(range(len(sorted_diff)), sorted_diff, color=bar_colors,
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel(f"IoU Difference ({methods[1]} - {methods[0]})", fontsize=12)
    ax.set_title(f"Per-Class IoU Gain: {methods[1]} over {methods[0]}",
                 fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=1)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved IoU difference chart to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-class IoU radar chart visualization."
    )
    parser.add_argument("--csv_path", type=str, default=None,
                        help="CSV with per-class IoU data.")
    parser.add_argument("--pred_dir", type=str, default=None,
                        help="Parent dir with method subdirs for computing IoU.")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="Ground truth directory (needed with --pred_dir).")
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--output_dir", type=str, default="results/figures")
    parser.add_argument("--exclude_background", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    class_names = VOC_CLASSES[:args.num_classes]

    ious_dict: Dict[str, np.ndarray] = {}

    if args.csv_path and os.path.exists(args.csv_path):
        print(f"Loading IoU data from {args.csv_path}")
        ious_dict = load_iou_from_csv(args.csv_path)
    elif args.pred_dir and args.gt_dir:
        methods = {"ST++": "stpp", "UniMatch": "unimatch"}
        for method_name, subdir in methods.items():
            pred_path = os.path.join(args.pred_dir, subdir, "pseudo_labels")
            if os.path.exists(pred_path):
                print(f"\nComputing IoU for {method_name}")
                ious = compute_per_class_iou_from_dirs(
                    pred_path, args.gt_dir, args.num_classes
                )
                ious_dict[method_name] = ious
    else:
        print("No data source provided. Generating demo with synthetic data.")
        rng = np.random.default_rng(42)
        base_ious = np.array([
            0.92, 0.78, 0.55, 0.72, 0.60, 0.58, 0.85, 0.82, 0.80, 0.45,
            0.70, 0.52, 0.75, 0.76, 0.73, 0.88, 0.48, 0.71, 0.62, 0.74, 0.61
        ])
        ious_dict["ST++"] = np.clip(base_ious - rng.uniform(0.05, 0.15, 21), 0, 1)
        ious_dict["UniMatch"] = np.clip(base_ious + rng.uniform(0.0, 0.08, 21), 0, 1)
        ious_dict["CW-BASS"] = np.clip(base_ious + rng.uniform(0.02, 0.12, 21), 0, 1)
        ious_dict["FARCLUSS"] = np.clip(base_ious + rng.uniform(0.05, 0.15, 21), 0, 1)

    if not ious_dict:
        print("No IoU data available. Exiting.")
        return

    # Print summary
    for method, ious in ious_dict.items():
        miou = np.mean(ious[1:]) if len(ious) > 1 else np.mean(ious)
        print(f"  {method}: mIoU={miou:.4f}")
        worst_cls = np.argmin(ious[1:]) + 1 if len(ious) > 1 else np.argmin(ious)
        print(f"    Worst class: {class_names[worst_cls]} ({ious[worst_cls]:.4f})")

    # Generate all visualizations
    plot_radar_chart(
        ious_dict, class_names,
        "Per-Class IoU Comparison (Pascal VOC)",
        os.path.join(args.output_dir, "per_class_iou_radar.png"),
        exclude_background=args.exclude_background,
    )
    plot_sorted_bar_chart(
        ious_dict, class_names,
        "Per-Class IoU (Sorted)",
        os.path.join(args.output_dir, "per_class_iou_sorted_bar.png"),
        exclude_background=args.exclude_background,
    )

    if len(ious_dict) >= 2:
        plot_iou_difference_chart(
            ious_dict, class_names,
            os.path.join(args.output_dir, "per_class_iou_difference.png"),
            exclude_background=args.exclude_background,
        )

    print("\nRadar chart visualization complete.")


if __name__ == "__main__":
    main()
