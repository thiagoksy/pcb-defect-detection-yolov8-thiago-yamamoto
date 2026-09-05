# Author: Thiago Yamamoto

# 🔬 Guide: How to Evaluate and Analyze PCB Defect Detection Models

This guide outlines a structured engineering approach to interpreting YOLOv8 training outputs, diagnosing model bottlenecks, and applying data-driven improvements to computer vision inspection systems.

---

## 📊 1. Interpreting Training Curves (`results.png`)

When training completes, YOLOv8 generates a `results.png` file mapping performance across epochs. Use this checklist for analysis:

* **Loss Convergence (Box, Class, DFL Loss):** Look for steady downward slopes. If validation loss plateaus early while training loss keeps dropping, the model is beginning to overfit to the training set.
* **mAP Metrics (mAP50 & mAP50-95):** 
  * `mAP50`: Evaluates performance at a standard 50% Intersection over Union (IoU) threshold. Ideal for checking general localization capability.
  * `mAP50-95`: A stricter metric averaging performance across IoU thresholds from 0.50 to 0.95. High values here indicate precise bounding box alignment.
* **Epoch Tuning Strategy:** If metrics are still improving linearly at epoch 15 (as seen in baseline hardware tests), double the training duration (e.g., to 30 or 50 epochs) while monitoring GPU thermal limits.

---

## 🗂️ 2. Diagnosing Misclassifications via Confusion Matrix

The `confusion_matrix.png` file is your primary tool for tracking cross-talk between similar classes:

* **The Diagonal Rule:** High values along the normalized diagonal represent correct classifications. Off-diagonal numbers reveal where the model gets confused.
* **Addressing Feature Overlap:** If classes like `Slug` and `Spillover` blend together, examine whether the bounding box labels overlap in the source dataset. Often, data re-labeling or refining class boundary definitions solves cross-talk faster than adding more layers.

---

## 🚀 3. Actionable Improvements & Next Steps

To push accuracy higher beyond initial baseline constraints, consider implementing these production-level enhancements:

* **Test-Time Augmentation (TTA):** Enable TTA during inference (`augment=True`) to predict images at multiple scales and flips, reducing false positives caused by orientation or minor shadows.
* **Data Balancing:** If classes like `slug` have significantly fewer training samples than `damaged`, apply targeted data augmentation (rotation, brightness shifts, cutout) specifically to minority classes.
* **Confidence Threshold Tuning:** Adjust the inference confidence parameter (`conf=0.5`) dynamically based on production needs—raising it to reduce false alarms in high-speed lines, or lowering it during initial sorting stages to ensure zero defective boards escape.
