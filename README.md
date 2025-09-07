# Behavior Detection with Multi-Sensor Data

## 📌 Project Goal
The goal of this project is to develop a predictive model that distinguishes **BFRB-like (Body-Focused Repetitive Behaviors)** and **non-BFRB-like** activity using data collected from a wrist-worn device.  
By successfully disentangling these behaviors, this work supports the design of more accurate wearable BFRB-detection devices, relevant to a wide range of mental illnesses, ultimately strengthening the tools available for treatment.

---

## 📊 Dataset
We used the **CMI Detect Behavior with Sensor Data** competition datasets.  
Key modalities:
- **IMU (Inertial Measurement Unit)**
- **Thermopiles**
- **Time-of-Flight (ToF)**

Half of the test set includes only IMU data, while the other half includes all three sensor modalities, enabling evaluation of whether additional sensors justify their cost and complexity.

---

## 🎭 Model Architecture
The final model is a **multi-branch deep learning architecture** with CNNs, Squeeze-and-Excitation modules, and a BERT encoder for temporal modeling.

### 🔹 Key Components
- **SEBlock**: Learns channel-wise attention weights to emphasize important feature channels.  
- **ResNetSEBlock**: Residual block with SE module for improved gradient flow and feature recalibration.  
- **CMIModel**:
  - Separate branches for IMU, Thermopile, and ToF data.  
  - CNN + SE-based feature extraction in each branch.  
  - Concatenation of features → passed to a **BERT encoder** with trainable `[CLS]` token.  
  - Final **fully connected classifier** outputs gesture class scores.  

This design combines:
- CNNs → local feature extraction  
- SE modules → channel attention  
- BERT → long-range temporal dependency modeling  

---

## 📏 Evaluation
The official competition metric is a **hybrid macro F1 score** that equally weights:
1. **Binary F1**: Target vs. non-target gestures.  
2. **Macro F1**: Across gesture classes, with all non-targets collapsed into one class.  

The final score = **average of Binary F1 and Macro F1**.

---

## 🔧 Usage

### 🔹 Installation
```bash
pip install -r requirements.txt
```

### 🔹 Training
Run the training notebook:
```bash
jupyter notebook cmi-model-5-6-1-training-model.ipynb
```

### 🔹 Inference / Submission
To generate predictions and submission file:
```bash
jupyter notebook cmi-model-5-6-1-submission-inference-model.ipynb
```

### 🔹 Data Preprocessing
For preparing features and scaling:
```bash
jupyter notebook cmi-model-5-6-1-data-preprocessing.ipynb
```

### 🔹 Artifacts
- `feature_cols.npy` – selected feature indices  
- `gesture_classes.npy` – list of gesture classes  
- `label_encoder.pkl` – label encoder for class mapping  
- `scaler.pkl` – fitted feature scaler  
- `sequence_maxlen.npy` – max sequence length for inputs  
- `submission_1.parquet` – example submission file  

---

## 🎯 Outputs & Real-World Impact
This competition challenges participants to:
1. Detect whether a gesture is BFRB-like or not.  
2. Classify the **specific type** of BFRB gesture.  

Importantly, the evaluation design helps determine whether **thermopiles and ToF sensors** significantly improve detection accuracy compared to IMU-only setups.  
Insights from this project directly inform **design choices in wearable mental health devices**.  

Relevant articles:  
- Garey, J. (2025). *What Is Excoriation, or Skin-Picking?* [Child Mind Institute](https://childmind.org/article/excoriation-or-skin-picking/)  
- Martinelli, K. (2025). *What is Trichotillomania?* [Child Mind Institute](https://childmind.org/article/what-is-trichotillomania/)  

---

## 🚀 Future Work
- Explore **self-supervised pretraining** on raw IMU sequences.  
- Test **larger transformer-based backbones** (e.g., Longformer, Performer) for long sequence modeling.  
- Investigate **multi-modal attention fusion** instead of simple concatenation.  
- Perform **model compression / quantization** for deployment on edge devices.  

---

## 📚 References
- [Competition: CMI – Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data)  
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)  
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)  

---

## 👨‍💼 Author
Developed by [Priyansh Keshari](https://github.com/priyanshkeshari) as part of the CMI Behavior Detection challenge.

