"""
Prediction Entropy Map Visualization.

Computes prediction entropy from model output logits, generates entropy
heatmap overlays on input images, and compares entropy patterns between
methods. High entropy regions indicate model uncertainty and correspond
to areas where baselines fail most often.

Author: Ebenezer Tarubinga, Korea University
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

IGNORE_INDEX: int = 255


def load_logits(path: str) -> np.ndarray:
    """Load logits from a .npy file with shape (C, H, W)."""
    return np.load(path).astype(np.float64)


def load_image(path: str) -> np.ndarray:
    """Load an RGB image as a numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path: str) -> np.ndarray:
    """Load a segmentation mask."""
    return np.array(Image.open(path), dtype=np.int32)


def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)


def compute_entropy(logits: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel prediction entropy from logits (C, H, W).
    Entropy is normalized to [0, 1] by dividing by log(C).
    """
    probs = softmax(logits, axis=0)  # (C, H, W)
    num_classes = probs.shape[0]
    # Clip to avoid log(0)
    probs_clipped = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs_clipped), axis=0)  # (H, W)
    max_entropy = np.log(num_classes)
    entropy_normalized = entropy / max_entropy
    return entropy_normalized


def compute_entropy_statistics(
    entropy: np.ndarray, gt: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute summary statistics for an entropy map."""
    stats: Dict[str, float] = {
        "mean": float(np.mean(entropy)),
        "std": float(np.std(entropy)),
        "median": float(np.median(entropy)),
        "max": float(np.max(entropy)),
        "high_entropy_frac": float(np.mean(entropy > 0.5)),
    }

    if gt is not None:
        valid = gt != IGNORE_INDEX
        if np.sum(valid) > 0:
            stats["mean_valid"] = float(np.mean(entropy[valid]))

    return stats


def create_entropy_overlay(
    image: np.ndarray,
    entropy: np.ndarray,
    alpha: float = 0.5,
    cmap_name: str = "hot",
) -> np.ndarray:
    """Create an entropy heatmap overlaid on the input image."""
    cmap = plt.cm.get_cmap(cmap_name)
    entropy_colored = cmap(entropy)[:, :, :3]  # (H, W, 3) in [0, 1]
    entropy_colored = (entropy_colored * 255).astype(np.uint8)

    image_float = image.astype(np.float64)
    overlay = (1 - alpha) * image_float + alpha * entropy_colored.astype(np.float64)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_single_entropy_examination(
    image: np.ndarray,
    entropy: np.ndarray,
    gt: Optional[np.ndarray],
    pred: Optional[np.ndarray],
    title: str,
    save_path: str,
) -> None:
    """Create a comprehensive single-image entropy examination figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input Image", fontsize=12, fontweight="bold")

    # Top-right: Entropy heatmap
    im = axes[0, 1].imshow(entropy, cmap="hot", vmin=0, vmax=1)
    axes[0, 1].set_title("Prediction Entropy", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Bottom-left: Overlay
    overlay = create_entropy_overlay(image, entropy, alpha=0.6)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Entropy Overlay", fontsize=12, fontweight="bold")

    # Bottom-right: Entropy histogram
    axes[1, 1].hist(entropy.flatten(), bins=50, color="#E64A19",
                     edgecolor="black", linewidth=0.5, alpha=0.7, density=True)
    axes[1, 1].axvline(x=np.mean(entropy), color="blue", linestyle="--",
                        linewidth=2, label=f"Mean: {np.mean(entropy):.3f}")
    axes[1, 1].set_xlabel("Normalized Entropy", fontsize=11)
    axes[1, 1].set_ylabel("Density", fontsize=11)
    axes[1, 1].set_title("Entropy Distribution", fontsize=12, fontweight="bold")
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[1, 1].set_xticks(np.arange(0, 1.1, 0.2))
    axes[1, 1].tick_params(axis="both", labelsize=9)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(
    image: np.ndarray,
    entropy_dict: Dict[str, np.ndarray],
    title: str,
    save_path: str,
) -> None:
    """Create side-by-side entropy comparison across methods."""
    num_methods = len(entropy_dict)
    fig, axes = plt.subplots(2, num_methods + 1, figsize=(4 * (num_methods + 1), 8))

    # Row 1: Image + entropy maps
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input Image", fontsize=11, fontweight="bold")

    for i, (method, entropy) in enumerate(entropy_dict.items()):
        im = axes[0, i + 1].imshow(entropy, cmap="hot", vmin=0, vmax=1)
        mean_e = np.mean(entropy)
        axes[0, i + 1].set_title(f"{method}\n(mean: {mean_e:.3f})",
                                  fontsize=11, fontweight="bold")

    # Row 2: Overlays
    axes[1, 0].axis("off")
    for i, (method, entropy) in enumerate(entropy_dict.items()):
        overlay = create_entropy_overlay(image, entropy, alpha=0.55)
        axes[1, i + 1].imshow(overlay)
        axes[1, i + 1].set_title(f"{method} Overlay", fontsize=10, style="italic")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    # Shared colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02,
                 label="Normalized Entropy")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_vs_error(
    entropy_all: np.ndarray,
    correct_all: np.ndarray,
    method_name: str,
    save_path: str,
    num_bins: int = 20,
) -> None:
    """Plot relationship between entropy and prediction error rate."""
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    error_rates = np.zeros(num_bins)
    counts = np.zeros(num_bins, dtype=np.int64)

    for i in range(num_bins):
        mask = (entropy_all >= bin_edges[i]) & (entropy_all < bin_edges[i + 1])
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            error_rates[i] = 1.0 - np.mean(correct_all[mask])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    valid = counts > 0
    ax1.bar(bin_centers, counts[valid] if np.all(valid) else counts,
            width=1.0 / num_bins, alpha=0.3, color="#90CAF9",
            edgecolor="black", linewidth=0.5, label="Pixel Count")
    ax2.plot(bin_centers[valid], error_rates[valid], color="#D32F2F",
             linewidth=2.5, marker="o", markersize=5, label="Error Rate")

    ax1.set_xlabel("Normalized Entropy", fontsize=12)
    ax1.set_ylabel("Pixel Count", fontsize=12, color="#1565C0")
    ax2.set_ylabel("Error Rate", fontsize=12, color="#D32F2F")
    ax1.set_title(f"Entropy vs. Error Rate: {method_name}",
                  fontsize=13, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved entropy vs error plot to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prediction entropy map visualization and evaluation."
    )
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory with input images.")
    parser.add_argument("--logits_dir", type=str, required=True,
                        help="Directory with .npy logit files (or parent with method subdirs).")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="Ground truth mask directory (optional).")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--compare", action="store_true",
                        help="Compare methods from subdirectories.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_files = sorted(Path(args.image_dir).glob("*.jpg")) + \
                  sorted(Path(args.image_dir).glob("*.png"))
    image_files = image_files[:args.num_images]
    print(f"Processing {len(image_files)} images")

    if args.compare:
        methods = {"ST++": "stpp", "UniMatch": "unimatch"}
    else:
        methods = {"Method": ""}

    for img_file in image_files:
        stem = img_file.stem
        image = load_image(str(img_file))
        gt = load_mask(os.path.join(args.gt_dir, f"{stem}.png")) if args.gt_dir else None

        entropy_dict: Dict[str, np.ndarray] = {}
        for method_label, subdir in methods.items():
            if subdir:
                logits_file = Path(args.logits_dir) / subdir / "logits" / f"{stem}.npy"
            else:
                logits_file = Path(args.logits_dir) / f"{stem}.npy"

            if logits_file.exists():
                logits = load_logits(str(logits_file))
                entropy = compute_entropy(logits)
                entropy_dict[method_label] = entropy

                stats = compute_entropy_statistics(entropy, gt)
                print(f"  {stem} / {method_label}: mean={stats['mean']:.3f}, "
                      f"high_entropy={stats['high_entropy_frac']:.1%}")

        if len(entropy_dict) == 1:
            method_name = list(entropy_dict.keys())[0]
            entropy = list(entropy_dict.values())[0]
            plot_single_entropy_examination(
                image, entropy, gt, None,
                f"Entropy Examination: {stem}",
                os.path.join(args.output_dir, f"entropy_{stem}.png"),
            )
        elif len(entropy_dict) > 1:
            plot_method_comparison(
                image, entropy_dict,
                f"Entropy Comparison: {stem}",
                os.path.join(args.output_dir, f"entropy_comparison_{stem}.png"),
            )

    # Aggregate entropy-error evaluation if GT available
    if args.gt_dir and len(methods) > 0:
        for method_label, subdir in methods.items():
            all_entropy: List[np.ndarray] = []
            all_correct: List[np.ndarray] = []

            for img_file in image_files:
                stem = img_file.stem
                if subdir:
                    logits_file = Path(args.logits_dir) / subdir / "logits" / f"{stem}.npy"
                else:
                    logits_file = Path(args.logits_dir) / f"{stem}.npy"
                gt_file = Path(args.gt_dir) / f"{stem}.png"

                if logits_file.exists() and gt_file.exists():
                    logits = load_logits(str(logits_file))
                    gt = load_mask(str(gt_file))
                    entropy = compute_entropy(logits)
                    pred = np.argmax(softmax(logits, axis=0), axis=0)

                    valid = gt != IGNORE_INDEX
                    all_entropy.append(entropy[valid])
                    all_correct.append((pred[valid] == gt[valid]).astype(np.float64))

            if all_entropy:
                entropy_cat = np.concatenate(all_entropy)
                correct_cat = np.concatenate(all_correct)
                safe_name = method_label.lower().replace(" ", "_").replace("+", "p")
                plot_entropy_vs_error(
                    entropy_cat, correct_cat, method_label,
                    os.path.join(args.output_dir, f"entropy_vs_error_{safe_name}.png"),
                )

    print("\nEntropy map visualization complete.")


if __name__ == "__main__":
    main()
