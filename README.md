# Semi-Supervised Semantic Segmentation: Baseline Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Systematic Study of UniMatch (CVPR 2023) and ST++ (CVPR 2022)**

*Ebenezer Tarubinga, M.Sc. Artificial Intelligence, Korea University*
*Supervised by Prof. Seong-Whan Lee*

---

## Context

This repository contains a comprehensive analysis of two leading semi-supervised semantic segmentation baselines: **UniMatch** (Yang et al., CVPR 2023) and **ST++** (Yang et al., CVPR 2022). The systematic investigation of their strengths and limitations directly informed the development of our own methods:

- **CW-BASS**: Class-Wise Boundary-Aware Semi-Supervised Segmentation (IJCNN 2025)
- **FARCLUSS**: Frequency-Adaptive Rare-Class Learning for Unified Semi-Supervised Segmentation (Neural Networks 2025)

By rigorously analyzing where and why these baselines fail, we identified the precise gaps that motivated the design of CW-BASS's boundary-aware modules and FARCLUSS's frequency-adaptive rebalancing strategy.

---

## Research Questions

1. **Where do pseudo-label errors concentrate?** We find that errors are not uniformly distributed but cluster heavily near object boundaries and on rare classes, suggesting that uniform confidence thresholding is suboptimal.

2. **How does confidence calibration differ between methods?** UniMatch produces better-calibrated probabilities than ST++ due to its dual-stream perturbation, but both methods remain over-confident on boundary pixels, motivating CW-BASS's spatially-aware calibration.

3. **What causes boundary degradation in semi-supervised settings?** Pseudo-labels near boundaries are noisy, and standard cross-entropy training amplifies this noise over self-training rounds. This is the core motivation for CW-BASS's boundary-aware loss.

4. **How do rare classes suffer disproportionately?** Rare classes receive fewer confident pseudo-labels, creating a feedback loop where under-represented classes degrade further. FARCLUSS directly addresses this with frequency-adaptive sampling and class-balanced contrastive learning.

---

## Key Findings

1. **Boundary pseudo-label accuracy drops by 15--25%** compared to interior pixels (within a 5-pixel trimap), confirming the need for boundary-aware training.
2. **Rare classes (< 2% of pixels) show 20--30% lower IoU** than frequent classes, with the gap widening under lower label ratios.
3. **UniMatch's dual perturbation improves calibration** (ECE reduced by ~35% vs. ST++), but calibration degrades on boundary pixels for both.
4. **Convergence analysis** reveals that ST++ pseudo-label quality saturates earlier (~60 epochs) than UniMatch (~100 epochs), suggesting diminishing returns from simple self-training.
5. **Confidence threshold sensitivity** analysis shows a narrow optimal range (0.90--0.95) for both methods; CW-BASS's dynamic thresholding removes this sensitivity.
6. **Failure cases cluster** around small objects, thin structures, and class-ambiguous boundaries -- exactly the scenarios our methods target.

---

## Results Comparison

### Pascal VOC 2012 (mIoU %)

| Method       | 1/16  | 1/8   | 1/4   | 1/2   |
|:-------------|:-----:|:-----:|:-----:|:-----:|
| ST++ (CVPR 2022) | 65.2 | 71.0 | 74.6 | 77.3 |
| UniMatch (CVPR 2023) | 75.2 | 76.6 | 77.2 | 78.8 |
| **CW-BASS (IJCNN 2025)** | **76.1** | **77.5** | **78.0** | **79.4** |
| **FARCLUSS (Neural Networks 2025)** | **78.2** | **79.0** | **79.8** | **80.3** |

### Cityscapes (mIoU %)

| Method       | 1/16  | 1/8   | 1/4   | 1/2   |
|:-------------|:-----:|:-----:|:-----:|:-----:|
| ST++ (CVPR 2022) | 67.4 | 72.2 | 74.4 | 77.0 |
| UniMatch (CVPR 2023) | 76.6 | 77.2 | 78.6 | 79.5 |
| **CW-BASS (IJCNN 2025)** | **77.3** | **78.1** | **79.2** | **80.1** |
| **FARCLUSS (Neural Networks 2025)** | **78.8** | **79.5** | **80.5** | **81.0** |

---

## Repository Structure

```
semi-supervised-segmentation-baselines/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── analysis/
│   ├── pseudo_label_quality.py       # Pseudo-label accuracy analysis
│   ├── confidence_distribution.py    # Calibration and confidence study
│   ├── boundary_accuracy.py          # Boundary mIoU evaluation
│   ├── class_imbalance_study.py      # Class frequency vs. IoU analysis
│   ├── convergence_comparison.py     # Training curve comparison
│   └── ablation_threshold_sensitivity.py  # Threshold sweep analysis
├── visualizations/
│   ├── compare_predictions.py        # Side-by-side prediction grids
│   ├── entropy_maps.py              # Entropy heatmap overlays
│   ├── per_class_iou_radar.py       # Radar charts for per-class IoU
│   └── failure_case_analysis.py     # Failure mode categorization
├── configs/
│   ├── unimatch_pascal_1_8.yaml
│   ├── unimatch_cityscapes_1_8.yaml
│   ├── stpp_pascal_1_8.yaml
│   └── stpp_cityscapes_1_8.yaml
├── notebooks/
│   └── baseline_deep_dive.py        # Notebook-style analysis walkthrough
├── docs/
│   ├── UNIMATCH_NOTES.md
│   ├── STPP_NOTES.md
│   └── MOTIVATION_FOR_CWBASS.md
└── results/
    ├── tables/.gitkeep
    └── figures/.gitkeep
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/etarubinga/semi-supervised-segmentation-baselines.git
cd semi-supervised-segmentation-baselines

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. **Pascal VOC 2012**: Download from [the official site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and place under `data/VOCdevkit/VOC2012/`.
2. **Cityscapes**: Download from [the Cityscapes website](https://www.cityscapes-dataset.com/) and place under `data/cityscapes/`.
3. **Model predictions**: Run the baseline models (UniMatch, ST++) and save predictions/logits under `data/predictions/{unimatch,stpp}/`.

### Running Analyses

```bash
# Pseudo-label quality analysis
python analysis/pseudo_label_quality.py \
    --pred_dir data/predictions/unimatch/pseudo_labels \
    --gt_dir data/VOCdevkit/VOC2012/SegmentationClass \
    --num_classes 21 --output_dir results

# Confidence distribution analysis
python analysis/confidence_distribution.py \
    --logits_dir data/predictions/unimatch/logits \
    --gt_dir data/VOCdevkit/VOC2012/SegmentationClass \
    --num_classes 21 --output_dir results

# Boundary accuracy analysis
python analysis/boundary_accuracy.py \
    --pred_dir data/predictions/unimatch/pseudo_labels \
    --gt_dir data/VOCdevkit/VOC2012/SegmentationClass \
    --num_classes 21 --trimap_width 5 --output_dir results

# Class imbalance study
python analysis/class_imbalance_study.py \
    --pred_dir data/predictions \
    --gt_dir data/VOCdevkit/VOC2012/SegmentationClass \
    --num_classes 21 --output_dir results

# Convergence comparison
python analysis/convergence_comparison.py \
    --log_dir data/training_logs \
    --output_dir results

# Threshold sensitivity ablation
python analysis/ablation_threshold_sensitivity.py \
    --logits_dir data/predictions/unimatch/logits \
    --gt_dir data/VOCdevkit/VOC2012/SegmentationClass \
    --num_classes 21 --output_dir results
```

---

## Acknowledgments

We gratefully acknowledge the original authors of the baseline methods:

- **UniMatch**: Lihe Yang, Lei Qi, Litong Feng, Wayne Zhang, Yinghuan Shi (CVPR 2023) -- for pioneering the unified dual-stream perturbation framework.
- **ST++**: Lihe Yang, Wei Zhuo, Lei Qi, Yinghuan Shi, Yang Gao (CVPR 2022) -- for the selective re-training strategy with prioritized pseudo-labels.

This work was conducted at the Department of Artificial Intelligence, Korea University, under the supervision of Prof. Seong-Whan Lee.

---

## Citations

```bibtex
@inproceedings{yang2023revisiting,
  title={Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation},
  author={Yang, Lihe and Qi, Lei and Feng, Litong and Zhang, Wayne and Shi, Yinghuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={7236--7246},
  year={2023}
}

@inproceedings{yang2022st++,
  title={ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation},
  author={Yang, Lihe and Zhuo, Wei and Qi, Lei and Shi, Yinghuan and Gao, Yang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={4268--4277},
  year={2022}
}

@inproceedings{tarubinga2025cwbass,
  title={CW-BASS: Class-Wise Boundary-Aware Semi-Supervised Segmentation},
  author={Tarubinga, Ebenezer and Lee, Seong-Whan},
  booktitle={Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year={2025}
}

@article{tarubinga2025farcluss,
  title={FARCLUSS: Frequency-Adaptive Rare-Class Learning for Unified Semi-Supervised Segmentation},
  author={Tarubinga, Ebenezer and Lee, Seong-Whan},
  journal={Neural Networks},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
