# pcb-defect-detection-yolov8-thiago-yamamoto
A computer vision system powered by a YOLOv8 deep learning neural network


# 🔍 Automated PCB Defect Detection using YOLOv8

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Framework](https://img.shields.io/badge/PyTorch-CUDA-red.svg)](https://pytorch.org/)

Computer vision pipeline developed to detect and classify manufacturing defects on Printed Circuit Boards (PCBs) using **YOLOv8**. This project was built as a practical implementation to bridge industrial automated optical inspection (AOI) concepts with edge-device machine learning constraints.

---

## 🚀 Key Features & Engineering Challenges

* **Model Architecture:** Trained using YOLOv8 on custom PCB defect classes.
* **Hardware:** Developed and executed iteratively under local hardware constraints (**NVIDIA GeForce GTX 1650, 4GB VRAM**), highlighting the trade-offs between training time, batch sizing, and accuracy.
* **Class Analysis:** Capable of detecting multiple defect types:
  * `Short_circuit` / Bridging
  * `Damaged` traces and physical board flaws
  * `Lack_of_part` (missing components)
  * `Miss_welding`, `Slug`, `Spillover`, and `Redundant` elements.
* **Real-World Edge Cases Identified:** 
  * *False Positives on Vias:* The 2D model occasionally confuses through-hole pads/vias with physical damage (`damaged`) due to lighting shadows and lack of 3D depth context.
  * *Intentional Jumpers vs. Defects:* Solder bridges created manually for hardware patches/bypasses are naturally flagged as `Short_circuit` by the model due to visual pattern matching.

---

## 📊 Performance Metrics (15 Epochs) (I will update the results with different epochs and parameters)

* **Overall mAP50:** `0.734`
* **Critical Defect Classes mAP50:**
  * `miss_welding`: `0.966`
  * `damaged`: `0.952`
  * `lack_of_part`: `0.951`
  * `Short_circuit`: `0.942`

---

## 🛠️ Project Structure

The repository is organized into modular scripts for training and inference:
I'm using only the dataset from Roboflow

1. **`train.py`**: Configures and executes the YOLOv8 training loop over the dataset.
2. **`predict_folder.py`**: Runs batch inference on a given test dataset folder, saving bounded-box visualizations.
3. **`predict_webcam.py`**: Opens a live video feed to test real-time inference using the trained weights.

---

## ⚙️ How to Run

### 1. Prerequisites
Install the required dependencies:
I'm using Python 3.11.0
```bash
pip install ultralytics opencv-python torch
```

### 2. Training the Model
To train the model from scratch or fine-tune it further, run:
```bash
python train.py
```
Or depending on how it was installed
```bash
py train.py
```

### 3. Running Batch Inference
To test the model on a folder of unseen PCB images:
```bash
python predict_folder.py
```
Or depending on how it was installed
```bash
py predict_folder.py
```

### 4. Real-Time Webcam Detection
To test live inference using your webcam:
```bash
python predict_webcam.py
```
Or depending on how it was installed
```bash
py predict_webcam.py
```

### 📈 Future Improvements

[ ] Expand dataset size and balance underrepresented classes (slug, spillover).

[ ] Increase training epochs to 30+ to allow convergence on recall curves.

[ ] Implement test-time augmentation (TTA) to reduce false positives on through-holes.
