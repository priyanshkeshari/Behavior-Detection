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

## Features
The model leverages multi-sensor data collected from a wrist-worn device:

- **IMU (Inertial Measurement Unit):** accelerometer and gyroscope data.
- **Thermopiles (THM):** thermal readings from multiple sensors.
- **Time-of-Flight (TOF):** distance and depth measurements.

Preprocessing generates features like:

- Normalized sensor readings
- Sequence padding for uniform length
- Feature scaling
- Label encoding for gestures

Files in this project:

- `feature_cols.npy` — Selected feature columns
- `gesture_classes.npy` — Gesture class labels
- `sequence_maxlen.npy` — Maximum sequence length
- `label_encoder.pkl` — Encodes gesture labels
- `scaler.pkl` — StandardScaler for normalization

---

## 📏 Evaluation
The official competition metric is a **hybrid macro F1 score** that equally weights:
1. **Binary F1**: Target vs. non-target gestures.  
2. **Macro F1**: Across gesture classes, with all non-targets collapsed into one class.  

The final score = **average of Binary F1 and Macro F1**.

---

## Workflow

### Data Preprocessing
1. Load raw sensor data
2. Normalize and scale features
3. Encode labels
4. Save features and metadata (`.npy` and `.pkl` files)

### Training
1. Use `cmi-model-5-6-1-training-model.ipynb`
2. Configure model hyperparameters for each branch
3. Train CMIModel on IMU + THM + TOF or IMU-only data
4. Save trained weights

### Inference & Submission
1. Use `cmi-model-5-6-1-submission-inference-model.ipynb`
2. Load trained model
3. Prepare test sequences
4. Generate predictions
5. Save output to `.parquet` or CSV for submission

<br>

```mermaid
flowchart TD
    %% === Raw Data & Preprocessing ===
    A[**Raw Sensor Data**] --> B[**Data Preprocessing**]
    B --> B1[Load IMU, THM, TOF Data]
    B1 --> B2[Normalize / Scale Features]
    B2 --> B3[Sequence Padding / Max Length Adjustment]

    %% === Parallel Feature Extraction Lanes ===
    subgraph IMU_Lane[**IMU Features**]
        style IMU_Lane fill:#1B2631,stroke:#1B4F72,stroke-width:1px,color:#FFFFFF
        C1[ResNetSE Blocks + SE Attention + Dropout]
    end

    subgraph THM_Lane[**THM Features**]
        style THM_Lane fill:#4A235A,stroke:#7D6608,stroke-width:1px,color:#FFFFFF
        C2[Conv1D + BN + ReLU + MaxPool + Dropout]
    end

    subgraph TOF_Lane[**TOF Features**]
        style TOF_Lane fill:#78281F,stroke:#78281F,stroke-width:1px,color:#FFFFFF
        C3[Conv1D + BN + ReLU + MaxPool + Dropout]
    end

    B3 --> C1
    B3 --> C2
    B3 --> C3

    %% === Fusion & Transformer ===
    C1 --> D[**Feature Concatenation / Fusion**]
    C2 --> D
    C3 --> D

    D --> E[**Transformer / BERT Encoder**]
    E --> F[CLS Token Extraction]

    %% === Classifier ===
    F --> G[**Fully Connected Classifier**]
    G --> H[**Softmax / Gesture Prediction**]

    %% === Train/Test Split & Evaluation ===
    B3 --> I[**Train/Test Split / Cross-Validation**]
    I --> J[**CMIModel Training**]
    J --> K[**Model Evaluation**]
    K --> L{**Evaluation Metric**}
    L -->|Binary F1 + Macro F1| M[**Final Score**]

    %% === Model Saving & Inference ===
    J --> N[Save Model Weights]
    N --> O[Model Download for Inference]
    O --> P[Inference on Test Data]
    P --> Q[Submission Generation]
    Q --> R[Submit to Kaggle / Competition]

    %% === Optional Query / Analysis ===
    U[Optional Query / Analysis] --> L

 ```   
  
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

