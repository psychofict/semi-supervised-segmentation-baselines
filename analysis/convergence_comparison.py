"""
Convergence Comparison for Semi-Supervised Segmentation Baselines.

Loads training logs (CSV or TensorBoard format) for UniMatch and ST++, plots
training curves (epoch vs mIoU, epoch vs loss), and compares convergence speed.
Analyzes when pseudo-label quality saturates for each method.

Author: Ebenezer Tarubinga, Korea University
"""

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def load_csv_log(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Load training metrics from a CSV log file.

    Expected columns: epoch, train_loss, val_loss, val_miou, pseudo_label_acc
    Missing columns are silently ignored.
    """
    data: Dict[str, List[float]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                key = key.strip()
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(value))
                except (ValueError, TypeError):
                    data[key].append(float("nan"))

    return {k: np.array(v) for k, v in data.items()}


def load_tensorboard_log(
    log_dir: str, tags: Optional[List[str]] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load training metrics from TensorBoard event files.

    Returns dict mapping tag -> (steps, values).
    """
    if not HAS_TENSORBOARD:
        print("  WARNING: tensorboard not available, cannot load event files.")
        return {}

    ea = EventAccumulator(log_dir)
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    if tags is None:
        tags = available_tags

    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tag in tags:
        if tag in available_tags:
            events = ea.Scalars(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            data[tag] = (steps, values)

    return data


def detect_log_format(log_path: str) -> str:
    """Detect whether a log path is CSV, TensorBoard directory, or unknown."""
    if os.path.isfile(log_path) and log_path.endswith(".csv"):
        return "csv"
    elif os.path.isdir(log_path):
        events = glob.glob(os.path.join(log_path, "events.out.tfevents.*"))
        if events:
            return "tensorboard"
        # Check for CSV inside directory
        csvs = glob.glob(os.path.join(log_path, "*.csv"))
        if csvs:
            return "csv_dir"
    return "unknown"


def load_method_logs(log_path: str) -> Dict[str, np.ndarray]:
    """Load training logs from either CSV or TensorBoard format."""
    fmt = detect_log_format(log_path)

    if fmt == "csv":
        return load_csv_log(log_path)
    elif fmt == "csv_dir":
        csvs = sorted(glob.glob(os.path.join(log_path, "*.csv")))
        if csvs:
            return load_csv_log(csvs[0])
    elif fmt == "tensorboard":
        tb_data = load_tensorboard_log(log_path)
        result: Dict[str, np.ndarray] = {}
        for tag, (steps, values) in tb_data.items():
            safe_tag = tag.replace("/", "_")
            result[f"{safe_tag}_step"] = steps
            result[safe_tag] = values
        return result

    print(f"  WARNING: Could not detect log format for {log_path}")
    return {}


def smooth_curve(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply moving average smoothing to a curve."""
    if len(values) < window:
        return values
    return uniform_filter1d(values.astype(np.float64), size=window, mode="nearest")


def find_saturation_epoch(
    values: np.ndarray,
    patience: int = 10,
    min_improvement: float = 0.001,
) -> int:
    """
    Find the epoch where a metric saturates (stops improving significantly).
    Uses a patience-based approach: saturation occurs when no improvement
    greater than min_improvement is observed for `patience` epochs.
    """
    best = -np.inf
    patience_counter = 0
    for i, v in enumerate(values):
        if v > best + min_improvement:
            best = v
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            return max(0, i - patience)
    return len(values) - 1


def plot_training_curves(
    methods_data: Dict[str, Dict[str, np.ndarray]],
    metric: str,
    ylabel: str,
    title: str,
    save_path: str,
    smooth_window: int = 5,
    epoch_key: str = "epoch",
) -> None:
    """Plot training curves for a given metric across methods."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2"]
    linestyles = ["-", "--", "-.", ":"]

    for idx, (method, data) in enumerate(methods_data.items()):
        if metric not in data:
            print(f"  WARNING: {metric} not found in {method} logs, skipping.")
            continue

        values = data[metric]
        if epoch_key in data:
            epochs = data[epoch_key]
        else:
            epochs = np.arange(len(values))

        smoothed = smooth_curve(values, smooth_window)

        ax.plot(epochs, values, color=colors[idx % len(colors)],
                alpha=0.2, linewidth=0.8)
        ax.plot(epochs, smoothed, color=colors[idx % len(colors)],
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2.5, label=f"{method} (smoothed)")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved training curve to {save_path}")


def plot_convergence_speed(
    methods_data: Dict[str, Dict[str, np.ndarray]],
    metric: str,
    thresholds: List[float],
    save_path: str,
    epoch_key: str = "epoch",
) -> None:
    """Plot epochs to reach certain metric thresholds for each method."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2"]

    method_names = list(methods_data.keys())
    epochs_to_threshold: Dict[str, List[Optional[int]]] = {m: [] for m in method_names}

    for method in method_names:
        data = methods_data[method]
        if metric not in data:
            epochs_to_threshold[method] = [None] * len(thresholds)
            continue

        values = data[metric]
        epochs = data.get(epoch_key, np.arange(len(values)))

        for thresh in thresholds:
            reached = np.where(values >= thresh)[0]
            if len(reached) > 0:
                epochs_to_threshold[method].append(int(epochs[reached[0]]))
            else:
                epochs_to_threshold[method].append(None)

    x = np.arange(len(thresholds))
    width = 0.8 / len(method_names)

    for idx, method in enumerate(method_names):
        offsets = (idx - len(method_names) / 2 + 0.5) * width
        vals = [e if e is not None else 0 for e in epochs_to_threshold[method]]
        bars = ax.bar(x + offsets, vals, width,
                      color=colors[idx % len(colors)],
                      edgecolor="black", linewidth=0.5,
                      label=method)

        for bar, val, original in zip(bars, vals, epochs_to_threshold[method]):
            if original is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(original), ha="center", va="bottom", fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 1,
                        "N/A", ha="center", va="bottom", fontsize=8, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.1%}" for t in thresholds], fontsize=10)
    ax.set_xlabel(f"{metric} Threshold", fontsize=12)
    ax.set_ylabel("Epochs to Reach", fontsize=12)
    ax.set_title("Convergence Speed Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved convergence speed plot to {save_path}")


def plot_saturation_analysis(
    methods_data: Dict[str, Dict[str, np.ndarray]],
    metric: str,
    save_path: str,
    epoch_key: str = "epoch",
) -> None:
    """Annotate saturation points on training curves."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2"]

    for idx, (method, data) in enumerate(methods_data.items()):
        if metric not in data:
            continue

        values = data[metric]
        epochs = data.get(epoch_key, np.arange(len(values)))
        smoothed = smooth_curve(values, 10)

        sat_epoch_idx = find_saturation_epoch(smoothed)
        sat_epoch = epochs[sat_epoch_idx] if sat_epoch_idx < len(epochs) else epochs[-1]
        sat_value = smoothed[sat_epoch_idx] if sat_epoch_idx < len(smoothed) else smoothed[-1]

        color = colors[idx % len(colors)]
        ax.plot(epochs, smoothed, color=color, linewidth=2.5, label=method)
        ax.axvline(x=sat_epoch, color=color, linestyle=":", linewidth=1.5, alpha=0.7)
        ax.scatter([sat_epoch], [sat_value], color=color, s=100,
                   zorder=5, edgecolors="black", linewidths=1.5)
        ax.annotate(
            f"Sat. @ epoch {int(sat_epoch)}",
            (sat_epoch, sat_value),
            xytext=(15, 10), textcoords="offset points",
            fontsize=9, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color),
        )

        print(f"  {method}: saturation at epoch {int(sat_epoch)}, {metric}={sat_value:.4f}")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"Pseudo-Label Quality Saturation ({metric})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved saturation analysis to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convergence comparison for semi-supervised segmentation baselines."
    )
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Parent directory containing method log subdirectories.")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--smooth_window", type=int, default=5,
                        help="Moving average window for curve smoothing.")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["stpp", "unimatch"],
                        help="Method subdirectory names.")
    parser.add_argument("--method_labels", type=str, nargs="+",
                        default=["ST++", "UniMatch"],
                        help="Display labels for methods.")
    args = parser.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Load logs for each method
    methods_data: Dict[str, Dict[str, np.ndarray]] = {}
    for subdir, label in zip(args.methods, args.method_labels):
        method_log = os.path.join(args.log_dir, subdir)
        if os.path.exists(method_log):
            print(f"\nLoading logs for {label} from {method_log}")
            data = load_method_logs(method_log)
            if data:
                methods_data[label] = data
                print(f"  Available metrics: {list(data.keys())}")
            else:
                print(f"  WARNING: No data loaded for {label}")
        else:
            print(f"  WARNING: {method_log} does not exist, skipping {label}")

    if not methods_data:
        print("\nNo training logs found. Generating example plots with synthetic data.")
        # Generate synthetic data for demonstration
        np.random.seed(42)
        epochs = np.arange(200)

        stpp_miou = 0.65 * (1 - np.exp(-epochs / 30)) + np.random.normal(0, 0.01, 200)
        stpp_loss = 1.0 * np.exp(-epochs / 40) + 0.3 + np.random.normal(0, 0.02, 200)
        stpp_pl_acc = 0.72 * (1 - np.exp(-epochs / 25)) + np.random.normal(0, 0.01, 200)

        uni_miou = 0.75 * (1 - np.exp(-epochs / 50)) + np.random.normal(0, 0.01, 200)
        uni_loss = 0.9 * np.exp(-epochs / 50) + 0.2 + np.random.normal(0, 0.02, 200)
        uni_pl_acc = 0.82 * (1 - np.exp(-epochs / 40)) + np.random.normal(0, 0.01, 200)

        methods_data = {
            "ST++": {
                "epoch": epochs.astype(np.float64),
                "val_miou": np.clip(stpp_miou, 0, 1),
                "train_loss": np.clip(stpp_loss, 0, None),
                "pseudo_label_acc": np.clip(stpp_pl_acc, 0, 1),
            },
            "UniMatch": {
                "epoch": epochs.astype(np.float64),
                "val_miou": np.clip(uni_miou, 0, 1),
                "train_loss": np.clip(uni_loss, 0, None),
                "pseudo_label_acc": np.clip(uni_pl_acc, 0, 1),
            },
        }
        print("  Generated synthetic training data for demonstration.")

    # Plot training curves
    for metric, ylabel, title in [
        ("val_miou", "Validation mIoU", "Training Convergence: Validation mIoU"),
        ("train_loss", "Training Loss", "Training Convergence: Loss"),
        ("pseudo_label_acc", "Pseudo-Label Accuracy", "Pseudo-Label Quality Over Training"),
    ]:
        any_has_metric = any(metric in d for d in methods_data.values())
        if any_has_metric:
            plot_training_curves(
                methods_data, metric, ylabel, title,
                os.path.join(fig_dir, f"convergence_{metric}.png"),
                smooth_window=args.smooth_window,
            )

    # Convergence speed
    if any("val_miou" in d for d in methods_data.values()):
        plot_convergence_speed(
            methods_data, "val_miou",
            thresholds=[0.50, 0.55, 0.60, 0.65],
            save_path=os.path.join(fig_dir, "convergence_speed.png"),
        )

    # Saturation analysis
    if any("pseudo_label_acc" in d for d in methods_data.values()):
        plot_saturation_analysis(
            methods_data, "pseudo_label_acc",
            os.path.join(fig_dir, "pseudo_label_saturation.png"),
        )

    print("\nConvergence analysis complete.")


if __name__ == "__main__":
    main()
