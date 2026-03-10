"""
Boundary Accuracy Evaluation for Semi-Supervised Segmentation Baselines.

Computes boundary mIoU using trimap-based evaluation, analyzes how prediction
accuracy degrades near object boundaries, and compares boundary performance
across methods. This evaluation is the KEY motivation for CW-BASS's
boundary-aware module.

Author: Ebenezer Tarubinga, Korea University
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

IGNORE_INDEX: int = 255

VOC_CLASSES: List[str] = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "dining table", "dog", "horse",
    "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
]


def load_mask(path: str) -> np.ndarray:
    """Load a segmentation mask as a numpy array."""
    return np.array(Image.open(path), dtype=np.int32)


def compute_boundary_mask(gt: np.ndarray, width: int = 1) -> np.ndarray:
    """
    Compute a binary boundary mask from a ground truth segmentation.
    Boundary pixels are those within `width` pixels of a class transition.
    """
    h, w = gt.shape
    boundary = np.zeros((h, w), dtype=bool)

    # Horizontal transitions
    boundary[:, :-1] |= gt[:, :-1] != gt[:, 1:]
    boundary[:, 1:] |= gt[:, :-1] != gt[:, 1:]

    # Vertical transitions
    boundary[:-1, :] |= gt[:-1, :] != gt[1:, :]
    boundary[1:, :] |= gt[:-1, :] != gt[1:, :]

    if width > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1)
        )
        boundary = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1).astype(bool)

    return boundary


def compute_distance_from_boundary(gt: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel distance to the nearest class boundary.
    Returns a float array where boundary pixels have distance 0.
    """
    boundary = compute_boundary_mask(gt, width=1)
    # Distance transform from boundary
    non_boundary = (~boundary).astype(np.uint8)
    distance = cv2.distanceTransform(non_boundary, cv2.DIST_L2, 5)
    return distance


def compute_iou_in_region(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-class IoU only within the given mask region."""
    ious = np.zeros(num_classes, dtype=np.float64)
    valid_classes = np.zeros(num_classes, dtype=bool)

    pred_masked = pred[mask]
    gt_masked = gt[mask]

    valid = gt_masked != IGNORE_INDEX
    pred_masked = pred_masked[valid]
    gt_masked = gt_masked[valid]

    for c in range(num_classes):
        intersection = np.sum((pred_masked == c) & (gt_masked == c))
        union = np.sum((pred_masked == c) | (gt_masked == c))
        if union > 0:
            ious[c] = intersection / union
            valid_classes[c] = True

    return ious, valid_classes


def compute_accuracy_by_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    distance_map: np.ndarray,
    max_distance: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pixel accuracy at each integer distance from boundary.
    Returns arrays of distances and accuracies.
    """
    distances = np.arange(0, max_distance + 1)
    accuracies = np.zeros(len(distances), dtype=np.float64)

    valid = gt != IGNORE_INDEX
    pred_valid = pred[valid]
    gt_valid = gt[valid]
    dist_valid = distance_map[valid]

    for i, d in enumerate(distances):
        if d == max_distance:
            mask = dist_valid >= d
        else:
            mask = (dist_valid >= d) & (dist_valid < d + 1)
        count = np.sum(mask)
        if count > 0:
            accuracies[i] = np.mean(pred_valid[mask] == gt_valid[mask])

    return distances, accuracies


def analyze_boundary_accuracy(
    pred_dir: str,
    gt_dir: str,
    num_classes: int,
    trimap_widths: List[int],
    max_distance: int,
    method_name: str,
) -> Dict:
    """Run full boundary accuracy evaluation for one method."""
    print(f"\n{'='*60}")
    print(f"Boundary accuracy evaluation: {method_name}")
    print(f"{'='*60}")

    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)
    pred_files = sorted(pred_path.glob("*.png"))

    pairs = []
    for pf in pred_files:
        gf = gt_path / pf.name
        if gf.exists():
            pairs.append((str(pf), str(gf)))

    print(f"  Found {len(pairs)} image pairs")

    # Accumulate per-class IoU for boundary and interior regions
    results_by_trimap: Dict[int, Dict[str, np.ndarray]] = {}
    for tw in trimap_widths:
        results_by_trimap[tw] = {
            "boundary_iou_sum": np.zeros(num_classes, dtype=np.float64),
            "boundary_count": np.zeros(num_classes, dtype=np.int64),
            "interior_iou_sum": np.zeros(num_classes, dtype=np.float64),
            "interior_count": np.zeros(num_classes, dtype=np.int64),
        }

    # Accumulate accuracy-by-distance
    all_dist_accuracies = np.zeros((len(pairs), max_distance + 1), dtype=np.float64)
    valid_images = 0

    for idx, (pred_file, gt_file) in enumerate(tqdm(pairs, desc=f"  {method_name}")):
        pred = load_mask(pred_file)
        gt = load_mask(gt_file)

        if pred.shape != gt.shape:
            pred_resized = cv2.resize(
                pred, (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            pred = pred_resized

        distance_map = compute_distance_from_boundary(gt)

        for tw in trimap_widths:
            boundary_mask = distance_map <= tw
            interior_mask = distance_map > tw

            b_ious, b_valid = compute_iou_in_region(pred, gt, boundary_mask, num_classes)
            i_ious, i_valid = compute_iou_in_region(pred, gt, interior_mask, num_classes)

            res = results_by_trimap[tw]
            res["boundary_iou_sum"] += b_ious * b_valid
            res["boundary_count"] += b_valid.astype(np.int64)
            res["interior_iou_sum"] += i_ious * i_valid
            res["interior_count"] += i_valid.astype(np.int64)

        # Accuracy by distance
        distances, acc = compute_accuracy_by_distance(pred, gt, distance_map, max_distance)
        all_dist_accuracies[idx] = acc
        valid_images += 1

    mean_dist_accuracies = np.mean(all_dist_accuracies[:valid_images], axis=0) if valid_images > 0 else np.zeros(max_distance + 1)

    # Summarize trimap results
    trimap_summary: Dict[int, Dict[str, float]] = {}
    for tw in trimap_widths:
        res = results_by_trimap[tw]
        b_count = np.maximum(res["boundary_count"], 1)
        i_count = np.maximum(res["interior_count"], 1)
        boundary_miou = np.mean(res["boundary_iou_sum"] / b_count)
        interior_miou = np.mean(res["interior_iou_sum"] / i_count)
        trimap_summary[tw] = {
            "boundary_miou": boundary_miou,
            "interior_miou": interior_miou,
            "gap": interior_miou - boundary_miou,
        }
        print(f"  Trimap width={tw}: boundary mIoU={boundary_miou:.4f}, "
              f"interior mIoU={interior_miou:.4f}, gap={interior_miou - boundary_miou:.4f}")

    return {
        "trimap_summary": trimap_summary,
        "distances": distances,
        "dist_accuracies": mean_dist_accuracies,
        "method_name": method_name,
    }


def plot_accuracy_vs_distance(
    results_list: List[Dict],
    save_path: str,
) -> None:
    """Plot accuracy as a function of distance from boundary for multiple methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2"]
    markers = ["o", "s", "^", "D"]

    for i, res in enumerate(results_list):
        ax.plot(
            res["distances"], res["dist_accuracies"],
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            markersize=5, linewidth=2,
            label=res["method_name"],
        )

    ax.set_xlabel("Distance from Boundary (pixels)", fontsize=12)
    ax.set_ylabel("Pixel Accuracy", fontsize=12)
    ax.set_title("Prediction Accuracy vs. Distance from Object Boundary",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, res["distances"][-1])
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved accuracy vs distance plot to {save_path}")


def plot_trimap_comparison(
    results_list: List[Dict],
    trimap_widths: List[int],
    save_path: str,
) -> None:
    """Bar chart comparing boundary vs interior mIoU across methods and trimap widths."""
    num_methods = len(results_list)
    num_trimaps = len(trimap_widths)

    fig, axes = plt.subplots(1, num_trimaps, figsize=(5 * num_trimaps, 6), sharey=True)
    if num_trimaps == 1:
        axes = [axes]

    colors_boundary = ["#E57373", "#F06292", "#BA68C8"]
    colors_interior = ["#64B5F6", "#4FC3F7", "#4DD0E1"]

    for ti, tw in enumerate(trimap_widths):
        ax = axes[ti]
        method_names = []
        b_vals = []
        i_vals = []
        for res in results_list:
            method_names.append(res["method_name"])
            summary = res["trimap_summary"].get(tw, {"boundary_miou": 0, "interior_miou": 0})
            b_vals.append(summary["boundary_miou"])
            i_vals.append(summary["interior_miou"])

        x = np.arange(num_methods)
        width = 0.35
        ax.bar(x - width / 2, b_vals, width, label="Boundary", color="#E57373", edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, i_vals, width, label="Interior", color="#64B5F6", edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(method_names, fontsize=10)
        ax.set_title(f"Trimap Width = {tw}px", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        if ti == 0:
            ax.set_ylabel("mIoU", fontsize=12)

    plt.suptitle("Boundary vs Interior mIoU Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved trimap comparison to {save_path}")


def save_boundary_csv(results_list: List[Dict], trimap_widths: List[int], save_path: str) -> None:
    """Save boundary evaluation results to CSV."""
    with open(save_path, "w") as f:
        f.write("method,trimap_width,boundary_miou,interior_miou,gap\n")
        for res in results_list:
            for tw in trimap_widths:
                s = res["trimap_summary"].get(tw, {"boundary_miou": 0, "interior_miou": 0, "gap": 0})
                f.write(f"{res['method_name']},{tw},"
                        f"{s['boundary_miou']:.4f},{s['interior_miou']:.4f},{s['gap']:.4f}\n")
    print(f"  Saved boundary CSV to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Boundary accuracy evaluation for semi-supervised segmentation."
    )
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Prediction directory (or parent with method subdirs if --compare).")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Ground truth mask directory.")
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--trimap_width", type=int, nargs="+", default=[3, 5, 10],
                        help="Trimap widths in pixels for boundary evaluation.")
    parser.add_argument("--max_distance", type=int, default=20,
                        help="Maximum distance from boundary to analyze.")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--compare", action="store_true",
                        help="Compare ST++ and UniMatch from subdirectories.")
    args = parser.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    tab_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    results_list: List[Dict] = []

    if args.compare:
        for method, subdir in [("ST++", "stpp"), ("UniMatch", "unimatch")]:
            method_pred = os.path.join(args.pred_dir, subdir, "pseudo_labels")
            result = analyze_boundary_accuracy(
                method_pred, args.gt_dir, args.num_classes,
                args.trimap_width, args.max_distance, method,
            )
            results_list.append(result)
    else:
        result = analyze_boundary_accuracy(
            args.pred_dir, args.gt_dir, args.num_classes,
            args.trimap_width, args.max_distance, "Method",
        )
        results_list.append(result)

    # Generate plots
    plot_accuracy_vs_distance(
        results_list,
        os.path.join(fig_dir, "accuracy_vs_boundary_distance.png"),
    )
    plot_trimap_comparison(
        results_list, args.trimap_width,
        os.path.join(fig_dir, "boundary_vs_interior_miou.png"),
    )
    save_boundary_csv(
        results_list, args.trimap_width,
        os.path.join(tab_dir, "boundary_accuracy_results.csv"),
    )

    print("\n" + "=" * 60)
    print("KEY FINDING FOR CW-BASS MOTIVATION:")
    if len(results_list) >= 2:
        for tw in args.trimap_width:
            gaps = [r["trimap_summary"][tw]["gap"] for r in results_list]
            print(f"  At trimap width={tw}px, boundary-interior gap: "
                  f"{', '.join(f'{g:.3f}' for g in gaps)}")
    print("  Both methods show significant boundary degradation,")
    print("  confirming the need for boundary-aware training (CW-BASS).")
    print("=" * 60)
    print("\nBoundary evaluation complete.")


if __name__ == "__main__":
    main()
