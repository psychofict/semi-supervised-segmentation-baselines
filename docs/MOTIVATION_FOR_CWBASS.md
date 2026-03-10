# Motivation for CW-BASS: From Baseline Evaluation to Boundary-Aware Design

**CW-BASS:** Class-Wise Boundary-Aware Semi-Supervised Segmentation (IJCNN 2025)
**Authors:** Ebenezer Tarubinga, Seong-Whan Lee

---

## How Baseline Evaluation Informed CW-BASS

This document details how our systematic evaluation of UniMatch and ST++ revealed specific failure patterns that directly motivated each component of CW-BASS.

---

## Gap 1: Boundary Degradation

### Observation

Using `evaluation/boundary_accuracy.py`, we measured prediction accuracy as a function of distance from object boundaries. The results were striking:

| Distance from Boundary | ST++ Accuracy | UniMatch Accuracy |
|:----------------------:|:------------:|:-----------------:|
| 0 px (on boundary)     | 0.52         | 0.61              |
| 1-3 px                 | 0.63         | 0.71              |
| 4-7 px                 | 0.78         | 0.83              |
| 8+ px (interior)       | 0.89         | 0.92              |

Both methods show a 25-30% accuracy drop at boundaries compared to object interiors. Even UniMatch, which is significantly better than ST++ overall, suffers from this boundary degradation.

### Root Cause Examination

1. **Pseudo-label noise at boundaries**: Object boundaries are inherently ambiguous at the pixel level. Both methods generate noisy pseudo-labels in these regions.

2. **Standard cross-entropy is boundary-agnostic**: The loss treats all pixels equally, so the majority of interior pixels dominate the gradient signal, and boundary pixels receive insufficient attention.

3. **Augmentation disrupts boundaries**: Strong augmentations (especially spatial augmentations like CutMix) can disrupt boundary structure, making it harder to learn boundary-specific features.

### CW-BASS Design Response

**Boundary-Aware Loss Weighting:**
- We introduce a boundary-aware weighting function that assigns higher loss weight to pixels near object boundaries
- Weight is computed from the distance transform of the current pseudo-label
- Formula: `w(p) = 1 + alpha * exp(-d(p) / sigma)` where `d(p)` is the distance from pixel p to the nearest boundary
- This ensures boundary pixels contribute more to the gradient, counteracting the numerical dominance of interior pixels

**Boundary Prediction Head:**
- An auxiliary task that predicts object boundaries from intermediate features
- This forces the encoder to maintain boundary-relevant features throughout the network
- The boundary prediction is also used to refine pseudo-labels by down-weighting high-confidence pseudo-labels that contradict the predicted boundary map

**Spatially-Adaptive Thresholding:**
- Instead of a uniform confidence threshold, we lower the threshold near predicted boundaries
- This retains more pseudo-label training signal in boundary regions, where the model needs it most
- Interior threshold: 0.95, Boundary threshold: 0.80 (adaptively interpolated)

---

## Gap 2: Class-Wise Accuracy Disparity

### Observation

Using `evaluation/class_imbalance_study.py`, we found that per-class IoU is strongly correlated with class frequency:

- **Frequent classes** (person, background): IoU > 85%
- **Medium classes** (car, cat, dog): IoU 70-80%
- **Rare classes** (bicycle, potted plant, dining table): IoU 40-60%

The gap between frequent and rare classes is 25-35%, and this gap widens under lower label ratios.

### Root Cause Examination

1. **Fewer training signals for rare classes**: With fewer pixels, rare classes produce fewer pseudo-labels above the confidence threshold.

2. **Confirmation bias**: The model becomes increasingly confident about common classes, which biases pseudo-labels away from rare classes.

3. **Class-agnostic thresholding**: A uniform threshold (0.95) is too aggressive for rare classes, where the model rarely reaches high confidence.

### CW-BASS Design Response

**Class-Wise Adaptive Thresholding:**
- Each class maintains its own confidence threshold, adapted based on the class-wise confidence distribution
- Threshold for class c: `tau_c = percentile_p(conf_c)` where p is chosen to retain a target fraction of pseudo-labels
- This ensures rare classes always contribute to training, even when overall model confidence is lower

**Class-Wise Loss Balancing:**
- Inverse-frequency weighting of the pseudo-label loss
- Weight for class c: `w_c = (median_freq / freq_c)^gamma` where gamma controls the strength of rebalancing
- Combined with boundary-aware weighting for a spatially and class-aware loss

---

## Gap 3: Confidence Calibration at Boundaries

### Observation

Using `evaluation/confidence_distribution.py`, we found that both methods are over-confident at boundaries:

- The model assigns confidence > 0.9 to 65% of boundary pixels
- But only 55-60% of those boundary pixels are actually correct
- This means standard confidence thresholding cannot distinguish good boundary pseudo-labels from bad ones

### CW-BASS Design Response

**Boundary-Conditioned Calibration:**
- We apply a calibration correction factor that reduces effective confidence near boundaries
- Corrected confidence: `conf_corrected(p) = conf(p) * (1 - beta * boundary_prob(p))`
- This prevents over-confident boundary pseudo-labels from corrupting training

---

## Gap 4: Convergence Behavior

### Observation

Using `evaluation/convergence_comparison.py`, we found:
- ST++ pseudo-label quality saturates after ~60 epochs
- UniMatch continues improving until ~100 epochs but then also plateaus
- Boundary accuracy specifically stops improving even as interior accuracy continues to grow

### CW-BASS Design Response

**Curriculum-Based Boundary Training:**
- In early training, use relaxed boundary criteria (wider trimap, lower threshold)
- Gradually tighten boundary criteria as the model improves
- This prevents early-stage noisy boundary pseudo-labels from creating persistent errors

---

## Summary: Baseline Gap to CW-BASS Component Mapping

| Baseline Gap | CW-BASS Component | Improvement |
|:---|:---|:---|
| Boundary accuracy drops 25-30% | Boundary-aware loss weighting | +3-5% boundary mIoU |
| Boundaries are over-confident | Boundary-conditioned calibration | Reduces false positives by 20% |
| No boundary feature learning | Auxiliary boundary prediction head | +1-2% overall mIoU |
| Uniform confidence threshold | Spatially-adaptive thresholding | +15% boundary pseudo-label retention |
| Class-agnostic threshold | Class-wise adaptive thresholding | +5-10% rare class IoU |
| Convergence plateau | Curriculum boundary training | Extends improvement by 30 epochs |

---

## Experimental Validation

The combination of these components in CW-BASS yields:
- **Pascal VOC 1/8**: 77.5 mIoU (vs. 76.6 UniMatch, +0.9)
- **Cityscapes 1/8**: 78.1 mIoU (vs. 77.2 UniMatch, +0.9)
- **Boundary mIoU improvement**: +4.2% on Pascal VOC
- **Rare class mIoU improvement**: +6.1% on average across rare classes
