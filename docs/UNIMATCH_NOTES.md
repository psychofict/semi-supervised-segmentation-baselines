# UniMatch Technical Notes

**Paper:** "Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation"
**Authors:** Lihe Yang, Lei Qi, Litong Feng, Wayne Zhang, Yinghuan Shi
**Venue:** CVPR 2023

---

## Core Idea

UniMatch unifies image-level and feature-level perturbations under a single weak-to-strong consistency framework. The key insight is that enforcing consistency between a weakly-augmented view and multiple strongly-augmented views (at both image and feature levels) provides richer supervisory signals than either perturbation type alone.

## Architecture

### Dual-Stream Perturbation Framework

The framework maintains three views of each unlabeled image:

1. **Weakly-augmented view** (anchor): Standard augmentation (random crop + horizontal flip). This view generates the pseudo-labels via the teacher (EMA model or the model itself).

2. **Strongly-augmented view (image-level)**: Aggressive color jittering, random grayscale conversion, Gaussian blur, and CutMix. The model's prediction on this view should match the pseudo-label from the weak view.

3. **Strongly-augmented view (feature-level)**: The features from the weakly-augmented image are perturbed by applying dropout and additive Gaussian noise at the feature level, then decoded. This creates a different "strong" view without altering the input image.

### Loss Formulation

```
L_total = L_supervised + lambda * (L_image_consistency + L_feature_consistency)
```

Where:
- `L_supervised`: Standard cross-entropy on labeled data
- `L_image_consistency`: Cross-entropy between pseudo-labels and predictions on image-perturbed view
- `L_feature_consistency`: Cross-entropy between pseudo-labels and predictions on feature-perturbed view
- `lambda`: Consistency weight (ramped up during training)

### Pseudo-Label Generation

Pseudo-labels are generated from the weakly-augmented view:
1. Forward pass on weakly-augmented image
2. Apply softmax to get class probabilities
3. Take argmax for pseudo-label, max probability for confidence
4. Threshold: only pixels with confidence >= tau (typically 0.95) contribute to the loss

## Key Technical Details

### Feature Perturbation
- Applied after the encoder, before the decoder
- Channel-wise dropout with p=0.5
- Additive Gaussian noise with std=0.1
- These are computationally cheap (no extra forward pass through encoder)

### Image Perturbation
- ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)
- RandomGrayscale(p=0.2)
- GaussianBlur(kernel_size=[3,7], sigma=[0.1, 2.0])
- The perturbations are chosen to be strong enough to create meaningful consistency targets

### Training Strategy
- SGD optimizer with momentum 0.9
- Poly learning rate schedule with power 0.9
- Batch size: typically 4 labeled + 4 unlabeled
- Consistency weight: ramped up from 0 to 1.0 over first 30 epochs
- No EMA teacher by default (the model generates its own pseudo-labels from the weak view)

## Strengths (Observed in Our Evaluation)

1. **Better calibration**: The dual-stream consistency regularization produces better-calibrated probabilities compared to ST++ (ECE reduced by ~35%).

2. **More stable training**: Unlike ST++'s discrete self-training rounds, UniMatch's online consistency training is smoother and less prone to error accumulation.

3. **Feature perturbation is complementary**: Image-level and feature-level perturbations capture different aspects of invariance. Feature perturbation is especially effective for capturing semantic invariances.

4. **Single-stage training**: No need for multiple training phases or pseudo-label regeneration rounds.

## Weaknesses (Identified in Our Evaluation)

1. **Boundary-agnostic**: The confidence threshold is uniform across all pixel locations. Our boundary evaluation shows that boundary pixels have systematically lower accuracy even when confidence is high (over-confident at boundaries).

2. **Class-agnostic thresholding**: The 0.95 threshold treats all classes equally. Rare classes rarely exceed this threshold, leading to fewer training signals for already under-represented categories.

3. **No explicit boundary handling**: The perturbation framework does not specifically target boundary regions, where pseudo-label noise is highest.

4. **Confidence saturation**: At higher label ratios, the model quickly becomes overconfident, and the consistency losses provide diminishing returns.

## Relevance to CW-BASS

Our boundary accuracy evaluation showed that UniMatch's predictions degrade by 15-20% within a 5-pixel trimap of object boundaries. This directly motivated CW-BASS's:
- **Boundary-aware loss weighting**: Higher loss weight for boundary pixels
- **Spatially-adaptive thresholding**: Lower confidence threshold near boundaries to retain more training signals
- **Boundary prediction head**: Auxiliary boundary detection task to improve boundary representation

## Relevance to FARCLUSS

The per-class confidence evaluation revealed that rare classes (bicycle, potted plant, dining table) have 20-30% fewer pseudo-labels retained after thresholding. This motivated FARCLUSS's:
- **Frequency-adaptive sampling**: Over-sample images containing rare classes
- **Class-balanced confidence thresholds**: Lower thresholds for rare classes
- **Contrastive learning**: Class-balanced contrastive loss to improve feature discrimination for rare classes
