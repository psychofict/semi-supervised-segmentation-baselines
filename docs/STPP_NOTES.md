# ST++ Technical Notes

**Paper:** "ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation"
**Authors:** Lihe Yang, Wei Zhuo, Lei Qi, Yinghuan Shi, Yang Gao
**Venue:** CVPR 2022

---

## Core Idea

ST++ improves upon vanilla self-training by introducing two key mechanisms: (1) selective re-training, which prioritizes reliable unlabeled samples, and (2) prioritized pseudo-labeling, which assigns pseudo-labels in an easy-to-hard curriculum. The method operates in discrete self-training rounds, each generating new pseudo-labels from the previous round's model.

## Architecture

### Multi-Phase Training Pipeline

**Phase 1: Supervised Pre-Training**
- Train model on labeled data only
- Standard cross-entropy loss with DeepLabV3+ on ResNet-101
- This provides the initial model for generating first-round pseudo-labels

**Phase 2: Self-Training Rounds (3 rounds typically)**
Each round consists of:
1. Generate pseudo-labels for all unlabeled images using current model
2. Estimate reliability of each pseudo-label
3. Selectively add reliable pseudo-labeled images to training set
4. Re-train from scratch (or fine-tune) on labeled + pseudo-labeled data

### Reliability Estimation

ST++ estimates pseudo-label reliability at the image level using prediction stability:

1. Run the model on the same unlabeled image multiple times with different augmentations
2. Measure consistency of predictions across augmentations
3. Images with more stable predictions are considered more reliable

**Stability Score:**
```
S(x) = 1/T * sum_{t=1}^{T} IoU(f(aug_t(x)), pseudo_label(x))
```
where T is the number of augmentation rounds and `f(aug_t(x))` is the prediction under augmentation t.

### Prioritized Pseudo-Labeling

Instead of using all unlabeled data at once:
1. Sort unlabeled images by reliability score (highest first)
2. In round 1: use top 50% most reliable images
3. In round 2: use top 75%
4. In round 3: use all images
This creates an easy-to-hard curriculum that reduces error accumulation.

### Loss Function

```
L = L_ce(labeled) + L_ce(pseudo-labeled, filtered)
```

Where pseudo-labeled loss only includes pixels with confidence above threshold (0.9 by default) from images selected by the reliability criterion.

## Key Technical Details

### Pseudo-Label Generation
- Forward pass with test-time augmentation (horizontal flip)
- Softmax + argmax for class prediction
- Per-pixel confidence thresholding at 0.9
- Generated once per self-training round (not online)

### Selective Re-Training Strategy
- Not all unlabeled images are used in each round
- Reliability is measured at the image level (not pixel level)
- This is a key difference from UniMatch, which operates at the pixel level
- Image-level selection prevents images with many noisy labels from corrupting training

### Training Schedule
- Phase 1: 80 epochs (Pascal VOC) / 100 epochs (Cityscapes)
- Phase 2: 60 epochs per round (Pascal VOC) / 80 epochs per round (Cityscapes)
- 3 self-training rounds total
- Each round restarts the learning rate schedule

## Strengths (Observed in Our Evaluation)

1. **Conceptual simplicity**: The self-training framework is straightforward and modular. Each component (pseudo-labeling, reliability estimation, curriculum) can be analyzed independently.

2. **Image-level reliability**: By filtering at the image level, ST++ avoids training on images where pseudo-labels are globally poor. This is a practical advantage over pixel-level methods that might include some correct pixels from a mostly-wrong pseudo-label.

3. **Progressive pseudo-label inclusion**: The easy-to-hard curriculum reduces error accumulation compared to using all pseudo-labels from the start.

4. **Decoupled pseudo-label generation**: Because pseudo-labels are generated offline (not online), they can be analyzed, visualized, and debugged separately from training.

## Weaknesses (Identified in Our Evaluation)

1. **Coarse reliability estimation**: Image-level reliability misses pixel-level variation. An image might be classified as "reliable" overall but have very poor pseudo-labels near object boundaries.

2. **Discrete rounds cause information loss**: The model is retrained from scratch (or fine-tuned from Phase 1) each round. Knowledge from intermediate stages can be lost. This contrasts with UniMatch's continuous online training.

3. **Pseudo-label staleness**: Within a round, pseudo-labels are fixed. As the model improves during training, the pseudo-labels become increasingly stale and potentially limit learning.

4. **Slower convergence**: Our convergence evaluation shows ST++ pseudo-label quality saturates earlier (~60 epochs) than UniMatch (~100 epochs), suggesting the discrete rounds create a ceiling effect.

5. **Fixed confidence threshold**: The 0.9 threshold is applied uniformly across all classes and spatial locations, leading to:
   - Rare classes losing most of their pseudo-labels
   - Boundary regions being over-confidently included (false sense of reliability)

6. **No perturbation diversity**: Unlike UniMatch's dual-stream perturbation, ST++ uses only standard augmentation. This limits the model's ability to learn perturbation-invariant representations.

## Relevance to CW-BASS

The discrete pseudo-labeling rounds in ST++ provided a clear experimental framework for studying how pseudo-label quality evolves over training. Our evaluation of ST++ pseudo-labels across rounds revealed:

- **Round 1 -> Round 2**: Boundary accuracy improves slightly (+2-3%), but rare class accuracy often degrades (-5-8%) due to confirmation bias
- **Round 2 -> Round 3**: Marginal improvements overall, suggesting diminishing returns from additional rounds
- **Boundary pixels**: Even after 3 rounds, boundary accuracy remains 20-25% lower than interior accuracy

These observations motivated CW-BASS's:
- **Online boundary refinement** instead of discrete rounds
- **Class-wise monitoring** to detect and prevent confirmation bias on rare classes
- **Boundary-conditioned pseudo-label filtering** that uses stricter criteria near boundaries

## Relevance to FARCLUSS

The class-level evaluation of ST++ revealed severe frequency bias:
- After reliability filtering, rare classes had 40-60% fewer pseudo-labeled pixels than frequent classes
- This bias compounds across rounds: fewer pseudo-labels -> worse model for that class -> even fewer pseudo-labels next round
- The easy-to-hard curriculum exacerbates this because rare-class images tend to have lower reliability scores

These observations motivated FARCLUSS's:
- **Frequency-adaptive pseudo-label weighting**: Higher weight for rare-class pixels
- **Class-balanced reliability scoring**: Reliability measured per-class, not per-image
- **Anti-confirmation-bias regularization**: Prevents the model from becoming increasingly certain about incorrect rare-class predictions
