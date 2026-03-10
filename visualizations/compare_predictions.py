"""
Side-by-Side Prediction Comparison Visualization.

Creates publication-quality comparison grids showing: Image | Ground Truth |
ST++ Prediction | UniMatch Prediction | CW-BASS Prediction. Uses the Pascal
VOC color palette for consistent segmentation visualization.

Author: Ebenezer Tarubinga, Korea University
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

# Pascal VOC color palette (21 classes)
VOC_PALETTE: np.ndarray = np.array([
    [0, 0, 0],        # background
    [128, 0, 0],      # aeroplane
    [0, 128, 0],      # bicycle
    [128, 128, 0],    # bird
    [0, 0, 128],      # boat
    [128, 0, 128],    # bottle
    [0, 128, 128],    # bus
    [128, 128, 128],  # car
    [64, 0, 0],       # cat
    [192, 0, 0],      # chair
    [64, 128, 0],     # cow
    [192, 128, 0],    # dining table
    [64, 0, 128],     # dog
    [192, 0, 128],    # horse
    [64, 128, 128],   # motorbike
    [192, 128, 128],  # person
    [0, 64, 0],       # potted plant
    [128, 64, 0],     # sheep
    [0, 192, 0],      # sofa
    [128, 192, 0],    # train
    [0, 64, 128],     # tv/monitor
], dtype=np.uint8)

VOC_CLASSES: List[str] = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "dining table", "dog", "horse",
    "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
]

IGNORE_INDEX: int = 255


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Convert an index mask to an RGB image using the given palette."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(len(palette)):
        colored[mask == cls_idx] = palette[cls_idx]
    # Mark ignore pixels as white
    colored[mask == IGNORE_INDEX] = [255, 255, 255]
    return colored


def load_image(path: str) -> np.ndarray:
    """Load an RGB image."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_mask(path: str) -> np.ndarray:
    """Load a segmentation mask."""
    return np.array(Image.open(path), dtype=np.int32)


def compute_error_overlay(
    pred: np.ndarray, gt: np.ndarray, image: np.ndarray
) -> np.ndarray:
    """Create an overlay highlighting prediction errors in red and correct in green."""
    overlay = image.copy().astype(np.float64)
    valid = gt != IGNORE_INDEX
    correct = (pred == gt) & valid
    incorrect = (pred != gt) & valid

    # Green tint for correct pixels
    overlay[correct, 1] = np.clip(overlay[correct, 1] + 60, 0, 255)

    # Red tint for incorrect pixels
    overlay[incorrect, 0] = np.clip(overlay[incorrect, 0] + 100, 0, 255)
    overlay[incorrect, 1] = overlay[incorrect, 1] * 0.5
    overlay[incorrect, 2] = overlay[incorrect, 2] * 0.5

    return overlay.astype(np.uint8)


def create_legend(
    classes_present: List[int],
    class_names: List[str],
    palette: np.ndarray,
) -> plt.Figure:
    """Create a standalone color legend for the classes present in the image."""
    patches = []
    for cls_idx in sorted(classes_present):
        if cls_idx < len(class_names) and cls_idx != IGNORE_INDEX:
            color = palette[cls_idx] / 255.0
            patches.append(mpatches.Patch(color=color, label=class_names[cls_idx]))
    return patches


def create_comparison_grid(
    image_path: str,
    gt_path: str,
    pred_paths: Dict[str, str],
    output_path: str,
    palette: np.ndarray = VOC_PALETTE,
    class_names: List[str] = VOC_CLASSES,
    show_errors: bool = True,
    figsize_per_col: float = 4.0,
) -> None:
    """
    Create a single comparison grid for one image.

    Layout (2 rows):
      Row 1: Image | GT | Method1 | Method2 | ...
      Row 2: (blank) | (blank) | Error1 | Error2 | ...  (if show_errors)
    """
    image = load_image(image_path)
    gt = load_mask(gt_path)

    methods: Dict[str, np.ndarray] = {}
    for method_name, pred_path in pred_paths.items():
        if os.path.exists(pred_path):
            methods[method_name] = load_mask(pred_path)

    num_methods = len(methods)
    num_cols = 2 + num_methods  # image, gt, methods
    num_rows = 2 if show_errors else 1

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(figsize_per_col * num_cols, figsize_per_col * num_rows + 1),
    )
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    # Row 1: Image, GT, predictions
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input Image", fontsize=11, fontweight="bold")

    gt_colored = colorize_mask(gt, palette)
    axes[0, 1].imshow(gt_colored)
    axes[0, 1].set_title("Ground Truth", fontsize=11, fontweight="bold")

    for i, (method_name, pred) in enumerate(methods.items()):
        pred_colored = colorize_mask(pred, palette)
        axes[0, 2 + i].imshow(pred_colored)

        # Compute accuracy
        valid = gt != IGNORE_INDEX
        if np.sum(valid) > 0:
            acc = np.mean(pred[valid] == gt[valid])
            axes[0, 2 + i].set_title(
                f"{method_name}\n(acc: {acc:.1%})", fontsize=11, fontweight="bold"
            )
        else:
            axes[0, 2 + i].set_title(method_name, fontsize=11, fontweight="bold")

    # Row 2: Error overlays
    if show_errors:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")
        for i, (method_name, pred) in enumerate(methods.items()):
            error_overlay = compute_error_overlay(pred, gt, image)
            axes[1, 2 + i].imshow(error_overlay)
            axes[1, 2 + i].set_title(
                f"{method_name} Errors", fontsize=10, style="italic"
            )

    # Remove axes ticks
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    # Legend
    classes_in_gt = np.unique(gt)
    classes_in_gt = classes_in_gt[classes_in_gt != IGNORE_INDEX]
    legend_patches = create_legend(classes_in_gt.tolist(), class_names, palette)
    if legend_patches:
        fig.legend(
            handles=legend_patches, loc="lower center",
            ncol=min(len(legend_patches), 7),
            fontsize=8, framealpha=0.9,
        )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_multi_image_grid(
    image_dir: str,
    gt_dir: str,
    pred_dirs: Dict[str, str],
    output_path: str,
    num_images: int = 6,
    palette: np.ndarray = VOC_PALETTE,
    class_names: List[str] = VOC_CLASSES,
    seed: int = 42,
) -> None:
    """Create a large grid with multiple images in rows, methods in columns."""
    image_files = sorted(Path(image_dir).glob("*.jpg")) + sorted(Path(image_dir).glob("*.png"))

    rng = np.random.default_rng(seed)
    if len(image_files) > num_images:
        indices = rng.choice(len(image_files), num_images, replace=False)
        image_files = [image_files[i] for i in sorted(indices)]
    else:
        image_files = image_files[:num_images]

    method_names = list(pred_dirs.keys())
    num_cols = 2 + len(method_names)  # image, gt, methods
    num_rows = len(image_files)

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(3.5 * num_cols, 3.5 * num_rows),
    )
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    # Column headers
    col_titles = ["Image", "GT"] + method_names
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for row, img_file in enumerate(image_files):
        stem = img_file.stem
        image = load_image(str(img_file))

        gt_file = Path(gt_dir) / f"{stem}.png"
        gt = load_mask(str(gt_file)) if gt_file.exists() else np.zeros(image.shape[:2], dtype=np.int32)

        axes[row, 0].imshow(image)
        axes[row, 1].imshow(colorize_mask(gt, palette))

        for j, method in enumerate(method_names):
            pred_file = Path(pred_dirs[method]) / f"{stem}.png"
            if pred_file.exists():
                pred = load_mask(str(pred_file))
                axes[row, 2 + j].imshow(colorize_mask(pred, palette))
            else:
                axes[row, 2 + j].text(0.5, 0.5, "N/A", transform=axes[row, 2 + j].transAxes,
                                       ha="center", va="center", fontsize=14, color="gray")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved multi-image grid to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create side-by-side prediction comparison visualizations."
    )
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory with input images.")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Ground truth mask directory.")
    parser.add_argument("--pred_dirs", type=str, nargs="+", required=True,
                        help="Prediction directories (one per method).")
    parser.add_argument("--method_names", type=str, nargs="+", required=True,
                        help="Method names corresponding to pred_dirs.")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    parser.add_argument("--num_images", type=int, default=6,
                        help="Number of images for multi-image grid.")
    parser.add_argument("--show_errors", action="store_true",
                        help="Show error overlay row.")
    parser.add_argument("--single_image", type=str, default=None,
                        help="If provided, create grid for this single image.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    assert len(args.pred_dirs) == len(args.method_names), \
        "Number of pred_dirs must match number of method_names."

    pred_dirs_map = dict(zip(args.method_names, args.pred_dirs))

    if args.single_image:
        stem = Path(args.single_image).stem
        gt_path = os.path.join(args.gt_dir, f"{stem}.png")
        pred_paths = {m: os.path.join(d, f"{stem}.png") for m, d in pred_dirs_map.items()}

        create_comparison_grid(
            args.single_image, gt_path, pred_paths,
            os.path.join(args.output_dir, f"comparison_{stem}.png"),
            show_errors=args.show_errors,
        )
        print(f"  Created comparison for {stem}")
    else:
        create_multi_image_grid(
            args.image_dir, args.gt_dir, pred_dirs_map,
            os.path.join(args.output_dir, "prediction_comparison_grid.png"),
            num_images=args.num_images,
        )

    print("\nPrediction comparison visualization complete.")


if __name__ == "__main__":
    main()
