# 📈 Model Evaluation & Engineering Insights

This document breaks down the performance metrics, quantitative results, and real-world edge cases observed during the training and validation of the PCB defect detection model.

---

## 📊 Quantitative Metrics (After 15 Epochs)

The model achieved an overall **mAP50 of 0.734** after 15 epochs under hardware-constrained training (NVIDIA GTX 1650). Analyzing individual classes reveals high confidence in distinct structural defects, while highlighting boundaries where visual overlap impacts classification:

* **Critical Defect Classes (High Confidence):**
  * `miss_welding`: `0.966` mAP50
  * `damaged`: `0.952` mAP50
  * `lack_of_part`: `0.951` mAP50
  * `Short_circuit`: `0.942` mAP50

---

## 🔍 Real-World Error Analysis & Edge Cases

Deploying and testing the model in real scenarios uncovered valuable computer vision challenges that provide key lessons for future iterations:

### 1. The "Through-Hole" Confusion (False Positives)
* **The Issue:** The model occasionally flags PCB through-holes (vias and mounting holes) as `damaged`.
* **The Engineering Cause:** Because YOLOv8 analyzes 2D optical images, circular empty holes cast dark shadows and create distinct boundary edges. Lacking 3D depth context, the network maps these optical patterns to features it learned as physical board flaws (`damaged`).
* **Mitigation Strategy:** Future updates require incorporating multi-angle lighting, polarized lenses to reduce shadows, or training with expanded negative samples of healthy vias.

### 2. Solder Jumpers vs. Short Circuits
* **The Issue:** Intentional solder jumpers (created manually for hardware patches or circuit modifications) are consistently flagged as `Short_circuit`.
* **The Engineering Cause:** Visually, an intentional solder bridge has the exact same pixel density, metallic gloss, and spatial topology as an accidental short-circuit bridge. 
* **Mitigation Strategy:** Context-aware classification or metadata integration (comparing against the original Gerber/CAD design layout file rather than relying purely on visual inference).

---

## 🛠️ How to Analyze Your Own Training Runs

If you run `train.py`, YOLOv8 automatically generates graphs and a Confusion Matrix inside the `runs/detect/pcb_defect_model_15_epochs/` folder. Here is how to interpret them:

1. **`/15_epochs/results.png`**: Check the training and validation loss curves. If the curves are still steadily sloping downward at epoch 15, it indicates the model hasn't fully converged yet, and increasing epochs (e.g., to 30 or 50) will yield better accuracy.
2. **`/15_epochs/confusion_matrix.png`**: Look at the diagonal cells (correct predictions) versus off-diagonal values (confusions). Pay close attention to classes that cross-talk (such as `Short_circuit` being misclassified as `damaged` with ~0.80 confidence, as observed in our tests).
