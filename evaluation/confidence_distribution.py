"""
Confidence Distribution and Calibration Study for Semi-Supervised Segmentation.

Loads model prediction logits/probabilities from UniMatch and ST++, computes
per-pixel confidence distributions, analyzes calibration via reliability diagrams
and Expected Calibration Error (ECE), and generates comparative visualizations.

Author: Ebenezer Tarubinga, Korea University
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

IGNORE_INDEX: int = 255


def load_logits(path: str) -> np.ndarray:
    """Load logits from a .npy file with shape (C, H, W)."""
    logits = np.load(path)
    return logits.astype(np.float64)


def load_mask(path: str) -> np.ndarray:
    """Load a ground truth segmentation mask."""
    from PIL import Image
    return np.array(Image.open(path), dtype=np.int32)


def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    """Numerically stable softmax along the specified axis."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)


def compute_confidence_and_pred(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """From logits (C, H, W), compute max confidence and predicted class per pixel."""
    probs = softmax(logits, axis=0)  # (C, H, W)
    confidence = np.max(probs, axis=0)  # (H, W)
    predicted = np.argmax(probs, axis=0)  # (H, W)
    return confidence, predicted


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error and return binned statistics.

    Returns:
        ece: scalar ECE value
        bin_centers: center of each confidence bin
        bin_accuracies: mean accuracy in each bin
        bin_confidences: mean confidence in each bin
        bin_counts: number of samples in each bin
    """
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0
    bin_accuracies = np.zeros(num_bins, dtype=np.float64)
    bin_confidences = np.zeros(num_bins, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=np.int64)

    for i in range(num_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == num_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        count = np.sum(mask)
        bin_counts[i] = count
        if count > 0:
            bin_accuracies[i] = np.mean(accuracies[mask])
            bin_confidences[i] = np.mean(confidences[mask])

    total = np.sum(bin_counts)
    ece = 0.0
    if total > 0:
        for i in range(num_bins):
            weight = bin_counts[i] / total
            ece += weight * np.abs(bin_accuracies[i] - bin_confidences[i])

    return ece, bin_centers, bin_accuracies, bin_confidences, bin_counts


def compute_per_class_confidence(
    confidences: np.ndarray,
    predictions: np.ndarray,
    gt_labels: np.ndarray,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean confidence and accuracy for each predicted class."""
    class_confidence = np.zeros(num_classes, dtype=np.float64)
    class_accuracy = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        mask = predictions == c
        if np.sum(mask) > 0:
            class_confidence[c] = np.mean(confidences[mask])
            class_accuracy[c] = np.mean(gt_labels[mask] == c)

    return class_confidence, class_accuracy


def plot_reliability_diagram(
    bin_centers: np.ndarray,
    bin_accuracies: np.ndarray,
    bin_confidences: np.ndarray,
    bin_counts: np.ndarray,
    ece: float,
    title: str,
    save_path: str,
) -> None:
    """Plot reliability diagram with gap visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 1]})

    valid = bin_counts > 0
    centers = bin_centers[valid]
    accs = bin_accuracies[valid]
    confs = bin_confidences[valid]
    counts = bin_counts[valid]

    # Main reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax1.bar(centers, accs, width=1.0 / len(bin_centers), alpha=0.7,
            edgecolor="black", linewidth=0.5, color="#4CAF50", label="Accuracy")
    ax1.bar(centers, np.abs(confs - accs), bottom=np.minimum(accs, confs),
            width=1.0 / len(bin_centers), alpha=0.3, color="#F44336", label="Gap")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(f"{title}\nECE = {ece:.4f}", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(alpha=0.3)

    # Bin count histogram
    ax2.bar(centers, counts, width=1.0 / len(bin_centers), color="#2196F3",
            edgecolor="black", linewidth=0.5, alpha=0.7)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Samples per Bin", fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved reliability diagram to {save_path}")


def plot_confidence_histogram(
    conf_correct: np.ndarray,
    conf_incorrect: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """Plot overlapping histograms of confidence for correct vs incorrect predictions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0.0, 1.0, 51)
    ax.hist(conf_correct, bins=bins, alpha=0.6, density=True,
            label="Correct", color="#4CAF50", edgecolor="black", linewidth=0.5)
    ax.hist(conf_incorrect, bins=bins, alpha=0.6, density=True,
            label="Incorrect", color="#F44336", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Max Prediction Confidence", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confidence histogram to {save_path}")


def plot_per_class_confidence(
    class_conf: np.ndarray,
    class_acc: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: str,
) -> None:
    """Scatter plot of per-class mean confidence vs accuracy."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    scatter = ax.scatter(class_conf, class_acc, c=np.arange(len(class_names)),
                         cmap="tab20", s=80, edgecolors="black", linewidths=0.5, zorder=5)

    for i, name in enumerate(class_names):
        ax.annotate(name, (class_conf[i], class_acc[i]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    ax.set_xlabel("Mean Confidence", fontsize=12)
    ax.set_ylabel("Actual Accuracy", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0.3, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved per-class confidence plot to {save_path}")


def plot_ece_comparison(
    methods: List[str], ece_values: List[float], save_path: str
) -> None:
    """Bar chart comparing ECE across methods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2"]
    bars = ax.bar(methods, ece_values, color=colors[:len(methods)],
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, ece_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=12)
    ax.set_title("Calibration Comparison", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(ece_values) * 1.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ECE comparison to {save_path}")


def analyze_method(
    logits_dir: str,
    gt_dir: str,
    num_classes: int,
    method_name: str,
    class_names: List[str],
    output_dir: str,
    max_samples: int = 500,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run full confidence and calibration evaluation for one method."""
    print(f"\n{'='*60}")
    print(f"Analyzing confidence distribution: {method_name}")
    print(f"{'='*60}")

    logits_path = Path(logits_dir)
    gt_path = Path(gt_dir)
    logit_files = sorted(logits_path.glob("*.npy"))[:max_samples]
    print(f"  Found {len(logit_files)} logit files")

    all_confidences: List[np.ndarray] = []
    all_correct: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []
    all_gt_labels: List[np.ndarray] = []

    for lf in tqdm(logit_files, desc=f"  Processing {method_name}"):
        gt_file = gt_path / lf.name.replace(".npy", ".png")
        if not gt_file.exists():
            continue

        logits = load_logits(str(lf))
        gt = load_mask(str(gt_file))

        confidence, predicted = compute_confidence_and_pred(logits)

        valid = gt != IGNORE_INDEX
        conf_valid = confidence[valid]
        pred_valid = predicted[valid]
        gt_valid = gt[valid]
        correct = (pred_valid == gt_valid).astype(np.float64)

        all_confidences.append(conf_valid)
        all_correct.append(correct)
        all_predictions.append(pred_valid)
        all_gt_labels.append(gt_valid)

    if len(all_confidences) == 0:
        print("  WARNING: No data loaded. Check paths.")
        return 0.0, np.zeros(num_classes), np.zeros(num_classes)

    all_conf = np.concatenate(all_confidences)
    all_corr = np.concatenate(all_correct)
    all_pred = np.concatenate(all_predictions)
    all_gt = np.concatenate(all_gt_labels)

    print(f"  Total pixels analyzed: {len(all_conf):,}")
    print(f"  Mean confidence: {np.mean(all_conf):.4f}")
    print(f"  Overall accuracy: {np.mean(all_corr):.4f}")

    # ECE computation
    ece, bin_centers, bin_accs, bin_confs, bin_counts = compute_ece(all_conf, all_corr)
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")

    # Per-class confidence
    class_conf, class_acc = compute_per_class_confidence(
        all_conf, all_pred, all_gt, num_classes
    )

    # Plotting
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    safe_name = method_name.lower().replace(" ", "_").replace("+", "p")

    plot_reliability_diagram(
        bin_centers, bin_accs, bin_confs, bin_counts, ece,
        f"Reliability Diagram: {method_name}",
        os.path.join(fig_dir, f"reliability_{safe_name}.png"),
    )

    correct_mask = all_corr == 1.0
    subsample_size = min(500_000, len(all_conf))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(all_conf), size=subsample_size, replace=False)

    plot_confidence_histogram(
        all_conf[idx][correct_mask[idx]],
        all_conf[idx][~correct_mask[idx]],
        f"Confidence Distribution: {method_name}",
        os.path.join(fig_dir, f"confidence_hist_{safe_name}.png"),
    )

    plot_per_class_confidence(
        class_conf, class_acc, class_names,
        f"Per-Class Calibration: {method_name}",
        os.path.join(fig_dir, f"class_calibration_{safe_name}.png"),
    )

    return ece, class_conf, class_acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confidence distribution and calibration evaluation."
    )
    parser.add_argument("--logits_dir", type=str, required=True,
                        help="Directory with .npy logit files (or parent with method subdirs).")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory with ground truth segmentation masks.")
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--compare", action="store_true",
                        help="Compare ST++ and UniMatch (expects subdirectories).")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Maximum number of images to analyze.")
    args = parser.parse_args()

    voc_classes = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "dining table", "dog", "horse",
        "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
    ]
    class_names = voc_classes[:args.num_classes]

    if args.compare:
        methods_data: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}
        for method, subdir in [("ST++", "stpp"), ("UniMatch", "unimatch")]:
            method_logits = os.path.join(args.logits_dir, subdir, "logits")
            ece, cc, ca = analyze_method(
                method_logits, args.gt_dir, args.num_classes,
                method, class_names, args.output_dir, args.max_samples,
            )
            methods_data[method] = (ece, cc, ca)

        fig_dir = os.path.join(args.output_dir, "figures")
        plot_ece_comparison(
            list(methods_data.keys()),
            [v[0] for v in methods_data.values()],
            os.path.join(fig_dir, "ece_comparison.png"),
        )
    else:
        analyze_method(
            args.logits_dir, args.gt_dir, args.num_classes,
            "Method", class_names, args.output_dir, args.max_samples,
        )

    print("\nCalibration evaluation complete.")


if __name__ == "__main__":
    main()
