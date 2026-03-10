"""
Failure Case Examination for Semi-Supervised Segmentation Baselines.

Identifies images where baseline methods fail most severely (lowest IoU),
categorizes failure modes into boundary errors, rare class confusion, and
scale issues, and generates annotated failure case visualizations.

Author: Ebenezer Tarubinga, Korea University
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from tqdm import tqdm

IGNORE_INDEX: int = 255

VOC_CLASSES: List[str] = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "dining table", "dog", "horse",
    "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
]

VOC_PALETTE: np.ndarray = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)

# Classes considered "rare" in Pascal VOC (low pixel frequency)
RARE_CLASSES: List[int] = [2, 4, 5, 9, 11, 16, 20]  # bicycle, boat, bottle, chair, dining table, potted plant, tv


def load_mask(path: str) -> np.ndarray:
    """Load a segmentation mask."""
    return np.array(Image.open(path), dtype=np.int32)


def load_image(path: str) -> np.ndarray:
    """Load an RGB image."""
    return np.array(Image.open(path).convert("RGB"))


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Convert index mask to RGB."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(palette)):
        colored[mask == c] = palette[c]
    colored[mask == IGNORE_INDEX] = [255, 255, 255]
    return colored


def compute_image_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> float:
    """Compute mean IoU for a single image."""
    valid = gt != IGNORE_INDEX
    pred_v = pred[valid]
    gt_v = gt[valid]
    ious = []
    for c in range(num_classes):
        intersection = np.sum((pred_v == c) & (gt_v == c))
        union = np.sum((pred_v == c) | (gt_v == c))
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def compute_boundary_error_ratio(
    pred: np.ndarray, gt: np.ndarray, boundary_width: int = 5
) -> float:
    """Compute the ratio of errors that occur near boundaries."""
    valid = gt != IGNORE_INDEX
    errors = (pred != gt) & valid
    total_errors = np.sum(errors)
    if total_errors == 0:
        return 0.0

    # Compute boundary mask
    h, w = gt.shape
    boundary = np.zeros((h, w), dtype=bool)
    boundary[:, :-1] |= gt[:, :-1] != gt[:, 1:]
    boundary[:, 1:] |= gt[:, :-1] != gt[:, 1:]
    boundary[:-1, :] |= gt[:-1, :] != gt[1:, :]
    boundary[1:, :] |= gt[:-1, :] != gt[1:, :]

    if boundary_width > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * boundary_width + 1, 2 * boundary_width + 1)
        )
        boundary = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1).astype(bool)

    boundary_errors = np.sum(errors & boundary)
    return boundary_errors / total_errors


def compute_rare_class_error_ratio(
    pred: np.ndarray, gt: np.ndarray, rare_classes: List[int]
) -> float:
    """Compute the ratio of errors involving rare classes."""
    valid = gt != IGNORE_INDEX
    errors = (pred != gt) & valid
    total_errors = np.sum(errors)
    if total_errors == 0:
        return 0.0

    rare_mask = np.isin(gt, rare_classes)
    rare_errors = np.sum(errors & rare_mask)
    return rare_errors / total_errors


def compute_small_object_ratio(
    gt: np.ndarray, size_threshold: int = 500
) -> float:
    """Compute the fraction of non-background objects that are small."""
    unique_classes = np.unique(gt)
    unique_classes = unique_classes[(unique_classes != 0) & (unique_classes != IGNORE_INDEX)]

    if len(unique_classes) == 0:
        return 0.0

    small_count = 0
    for c in unique_classes:
        class_pixels = np.sum(gt == c)
        if class_pixels < size_threshold:
            small_count += 1

    return small_count / len(unique_classes)


def categorize_failure(
    boundary_ratio: float,
    rare_ratio: float,
    small_ratio: float,
) -> str:
    """Categorize the primary failure mode of an image."""
    scores = {
        "boundary": boundary_ratio,
        "rare_class": rare_ratio,
        "scale": small_ratio,
    }
    primary = max(scores, key=scores.get)

    if scores[primary] < 0.2:
        return "mixed"

    category_labels = {
        "boundary": "Boundary Error",
        "rare_class": "Rare Class Confusion",
        "scale": "Scale Issue",
    }
    return category_labels[primary]


def analyze_failures(
    pred_dir: str,
    gt_dir: str,
    image_dir: Optional[str],
    num_classes: int,
    method_name: str,
    top_k: int = 20,
) -> List[Dict]:
    """Analyze all predictions and identify worst failures."""
    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)
    pred_files = sorted(pred_path.glob("*.png"))

    image_results: List[Dict] = []

    for pf in tqdm(pred_files, desc=f"  Analyzing {method_name}"):
        gf = gt_path / pf.name
        if not gf.exists():
            continue

        pred = load_mask(str(pf))
        gt = load_mask(str(gf))
        miou = compute_image_iou(pred, gt, num_classes)
        boundary_ratio = compute_boundary_error_ratio(pred, gt)
        rare_ratio = compute_rare_class_error_ratio(pred, gt, RARE_CLASSES)
        small_ratio = compute_small_object_ratio(gt)
        category = categorize_failure(boundary_ratio, rare_ratio, small_ratio)

        image_results.append({
            "filename": pf.name,
            "stem": pf.stem,
            "miou": miou,
            "boundary_error_ratio": boundary_ratio,
            "rare_class_error_ratio": rare_ratio,
            "small_object_ratio": small_ratio,
            "failure_category": category,
            "method": method_name,
        })

    # Sort by mIoU (ascending = worst first)
    image_results.sort(key=lambda x: x["miou"])
    return image_results[:top_k]


def plot_failure_case(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    info: Dict,
    palette: np.ndarray,
    save_path: str,
) -> None:
    """Create annotated failure case visualization."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Input image
    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=11, fontweight="bold")

    # Ground truth
    axes[1].imshow(colorize_mask(gt, palette))
    axes[1].set_title("Ground Truth", fontsize=11, fontweight="bold")

    # Prediction
    axes[2].imshow(colorize_mask(pred, palette))
    axes[2].set_title(f"{info['method']} (mIoU: {info['miou']:.1%})",
                       fontsize=11, fontweight="bold")

    # Error map
    valid = gt != IGNORE_INDEX
    error_map = np.zeros((*gt.shape, 3), dtype=np.uint8)
    correct = (pred == gt) & valid
    incorrect = (pred != gt) & valid
    error_map[correct] = [100, 200, 100]  # green
    error_map[incorrect] = [220, 50, 50]  # red
    error_map[~valid] = [200, 200, 200]   # gray for ignore

    axes[3].imshow(error_map)
    axes[3].set_title(f"Error Map ({info['failure_category']})",
                       fontsize=11, fontweight="bold", color="darkred")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Add annotation text
    info_text = (
        f"Boundary errors: {info['boundary_error_ratio']:.0%} | "
        f"Rare class errors: {info['rare_class_error_ratio']:.0%} | "
        f"Small objects: {info['small_object_ratio']:.0%}"
    )
    fig.text(0.5, 0.02, info_text, ha="center", fontsize=10,
             style="italic", color="dimgray")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_failure_distribution(
    failures: List[Dict],
    method_name: str,
    save_path: str,
) -> None:
    """Plot distribution of failure categories."""
    categories = [f["failure_category"] for f in failures]
    unique_cats = sorted(set(categories))
    counts = [categories.count(c) for c in unique_cats]

    cat_colors = {
        "Boundary Error": "#E57373",
        "Rare Class Confusion": "#64B5F6",
        "Scale Issue": "#FFB74D",
        "mixed": "#A5D6A7",
    }
    colors_list = [cat_colors.get(c, "#BDBDBD") for c in unique_cats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    ax1.pie(counts, labels=unique_cats, colors=colors_list, autopct="%1.0f%%",
            startangle=90, textprops={"fontsize": 10})
    ax1.set_title(f"Failure Category Distribution: {method_name}",
                  fontsize=12, fontweight="bold")

    # Scatter: mIoU vs boundary error ratio, colored by category
    for cat in unique_cats:
        cat_data = [f for f in failures if f["failure_category"] == cat]
        mious = [f["miou"] for f in cat_data]
        b_ratios = [f["boundary_error_ratio"] for f in cat_data]
        ax2.scatter(b_ratios, mious, c=cat_colors.get(cat, "#BDBDBD"),
                    label=cat, s=60, edgecolors="black", linewidths=0.5, alpha=0.8)

    ax2.set_xlabel("Boundary Error Ratio", fontsize=11)
    ax2.set_ylabel("Image mIoU", fontsize=11)
    ax2.set_title(f"Failure Examination: {method_name}", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved failure distribution to {save_path}")


def save_failures_csv(failures: List[Dict], save_path: str) -> None:
    """Save failure case results to CSV."""
    with open(save_path, "w") as f:
        f.write("method,filename,miou,boundary_error_ratio,rare_class_error_ratio,"
                "small_object_ratio,failure_category\n")
        for info in failures:
            f.write(
                f"{info['method']},{info['filename']},{info['miou']:.4f},"
                f"{info['boundary_error_ratio']:.4f},"
                f"{info['rare_class_error_ratio']:.4f},"
                f"{info['small_object_ratio']:.4f},"
                f"{info['failure_category']}\n"
            )
    print(f"  Saved failures CSV to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Failure case examination for semi-supervised segmentation baselines."
    )
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Parent dir with method subdirectories.")
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Input image directory for visualizations.")
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of worst failures to analyze.")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--visualize_top", type=int, default=5,
                        help="Number of worst cases to visualize in detail.")
    args = parser.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    tab_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    all_failures: List[Dict] = []
    methods = {"ST++": "stpp", "UniMatch": "unimatch"}

    for method_name, subdir in methods.items():
        pred_path = os.path.join(args.pred_dir, subdir, "pseudo_labels")
        if not os.path.exists(pred_path):
            print(f"  Skipping {method_name}: {pred_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Failure case examination: {method_name}")
        print(f"{'='*60}")

        failures = analyze_failures(
            pred_path, args.gt_dir, args.image_dir,
            args.num_classes, method_name, args.top_k,
        )
        all_failures.extend(failures)

        # Summary
        if failures:
            print(f"  Top {len(failures)} worst images:")
            for i, f in enumerate(failures[:5]):
                print(f"    {i+1}. {f['filename']}: mIoU={f['miou']:.3f} [{f['failure_category']}]")

            plot_failure_distribution(
                failures, method_name,
                os.path.join(fig_dir, f"failure_distribution_{subdir}.png"),
            )

            # Visualize top failures
            if args.image_dir:
                for i, info in enumerate(failures[:args.visualize_top]):
                    img_path = os.path.join(args.image_dir, info["stem"] + ".jpg")
                    if not os.path.exists(img_path):
                        img_path = os.path.join(args.image_dir, info["stem"] + ".png")
                    if os.path.exists(img_path):
                        image = load_image(img_path)
                        gt = load_mask(os.path.join(args.gt_dir, info["filename"]))
                        pred = load_mask(os.path.join(pred_path, info["filename"]))
                        plot_failure_case(
                            image, gt, pred, info, VOC_PALETTE,
                            os.path.join(fig_dir, f"failure_{subdir}_{i+1}_{info['stem']}.png"),
                        )

    if all_failures:
        save_failures_csv(
            all_failures,
            os.path.join(tab_dir, "failure_cases.csv"),
        )

    print(f"\n{'='*60}")
    print("FAILURE CASE SUMMARY:")
    for method_name in methods:
        method_failures = [f for f in all_failures if f["method"] == method_name]
        if method_failures:
            cats = [f["failure_category"] for f in method_failures]
            print(f"  {method_name}: {len(method_failures)} worst cases analyzed")
            for cat in sorted(set(cats)):
                print(f"    {cat}: {cats.count(cat)} ({cats.count(cat)/len(cats):.0%})")
    print("=" * 60)
    print("\nFailure case examination complete.")


if __name__ == "__main__":
    main()
