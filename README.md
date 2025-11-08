# 🩺 Chest X-Ray Classification: DenseNet-121 vs ResNet-50

## 📘 Overview
This project compares two deep convolutional neural networks — **ResNet-50** and **DenseNet-121** — for **automatic chest X-ray image classification**.  
The models were evaluated on a dataset consisting of two classes: **Normal** and **Pneumonia**.  
Both networks were fine-tuned using transfer learning with ImageNet pre-trained weights.

---

## 🧠 Model Architectures

### 🔹 ResNet-50
ResNet-50 is composed of 50 layers and employs **residual (skip) connections** to prevent vanishing gradients and enable the training of deeper networks.

**Architecture Summary:**
- **Total parameters:** 29,885,317 (114.00 MB)  
- **Trainable parameters:** 2,099,201 (8.01 MB)  
- **Non-trainable parameters:** 23,587,712 (89.98 MB)  
- **Optimizer parameters:** 4,198,404 (16.02 MB)

**Figure 1. ResNet-50 Training History**  
![ResNet-50 Training History](https://raw.githubusercontent.com/doctorcode9/ChestXray-Classifier-ResNet50-Vs-DenseNet121/refs/heads/main/images/resnet_train.png)

**Figure 2. ResNet-50 Confusion Matrix**  
![ResNet-50 Confusion Matrix](https://raw.githubusercontent.com/doctorcode9/ChestXray-Classifier-ResNet50-Vs-DenseNet121/refs/heads/main/images/resnet_matrix.png)


**Figure 3. ResNet-50 Validation Results**  
![ResNet-50 Validation Results](https://raw.githubusercontent.com/doctorcode9/ChestXray-Classifier-ResNet50-Vs-DenseNet121/refs/heads/main/images/resnet_val.png)

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|:------|:----------:|:------:|:---------:|:--------:|
| NORMAL | 0.92 | 0.81 | 0.86 | 234 |
| PNEUMONIA | 0.89 | 0.96 | 0.92 | 390 |
| **Accuracy** |  |  | **0.90** | **624** |
| **Macro Avg** | 0.91 | 0.88 | 0.89 | 624 |
| **Weighted Avg** | 0.90 | 0.90 | 0.90 | 624 |

---

### 🔹 DenseNet-121
DenseNet-121 establishes **dense connections** between all layers within a block, enhancing feature reuse and gradient propagation.  
It is more parameter-efficient compared to ResNet-50.

**Architecture Summary:**
- **Total parameters:** 8,088,129 (30.85 MB)  
- **Trainable parameters:** 1,050,625 (4.01 MB)  
- **Non-trainable parameters:** 7,037,504 (26.85 MB)

**Figure 4. DenseNet-121 Training History**  
![DenseNet-121 Training History](https://raw.githubusercontent.com/doctorcode9/ChestXray-Classifier-ResNet50-Vs-DenseNet121/refs/heads/main/images/dense_train.png)

**Figure 5. DenseNet-121 Confusion Matrix**  
![DenseNet-121 Confusion Matrix](https://raw.githubusercontent.com/doctorcode9/ChestXray-Classifier-ResNet50-Vs-DenseNet121/refs/heads/main/images/dense_matrix.png)

**Figure 6. DenseNet-121 Validation Results**  
![DenseNet-121 Validation Results](https://raw.githubusercontent.com/doctorcode9/ChestXray-Classifier-ResNet50-Vs-DenseNet121/refs/heads/main/images/dense_val.png)

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|:------|:----------:|:------:|:---------:|:--------:|
| NORMAL | 0.94 | 0.73 | 0.82 | 234 |
| PNEUMONIA | 0.86 | 0.97 | 0.91 | 390 |
| **Accuracy** |  |  | **0.88** | **624** |
| **Macro Avg** | 0.90 | 0.85 | 0.87 | 624 |
| **Weighted Avg** | 0.89 | 0.88 | 0.88 | 624 |

---

## 📊 Performance Comparison

| Metric | ResNet-50 | DenseNet-121 |
|:-------|:-----------:|:-------------:|
| Accuracy | 0.90 | 0.88 |
| Precision (Macro) | 0.91 | 0.90 |
| Recall (Macro) | 0.88 | 0.85 |
| F1-Score (Macro) | 0.89 | 0.87 |
| Total Parameters | 29.9M | 8.1M |
| Trainable Parameters | 2.1M | 1.0M |

---

## 💡 Discussion
- **ResNet-50** achieved slightly higher accuracy and balanced precision-recall values.  
- **DenseNet-121** demonstrated competitive performance with fewer parameters, confirming its parameter efficiency and feature reuse advantage.  
- Both models show strong potential for automated chest X-ray diagnosis, with minor trade-offs between computational cost and sensitivity.

---

## 🧭 Conclusion
Both architectures are effective for chest X-ray classification:
- **ResNet-50** delivers robust and consistent performance.  
- **DenseNet-121** achieves comparable accuracy while being lighter and faster to train.

Further work could involve model ensembling or fine-tuning with larger and more diverse datasets to enhance generalization.

---
