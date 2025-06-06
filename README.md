# 🫁 Pneumonia Detection using CNN, ResNet, DenseNet, and ViT with Grad-CAM

## 📌 Overview

This project builds an AI system to detect **pneumonia from chest X-ray images** using several deep learning models. To enhance interpretability, it also integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to highlight the image regions most influential in the model's decision-making.

---

## 📚 Models Used

- ✅ **Custom CNN** – A basic convolutional neural network
- ✅ **ResNet-18** – Residual Network for deep feature extraction
- ✅ **DenseNet-121** – Efficient and compact CNN with dense connections
- ✅ **ViT (Vision Transformer)** – Transformer-based model (`vit-base-patch16-224`)

Each model is trained or fine-tuned for binary classification: **Pneumonia vs Normal**.

---

## 🔍 Explainability with Grad-CAM

Grad-CAM is used to visualize the regions of X-ray images that the model focuses on when making predictions. This is particularly helpful in medical imaging for trust and verification.

---

## 🧪 Model Evaluation (Example)

| Model     | Accuracy  | Precision  | Recall  | F1-score |
|-----------|-----------|------------|---------|----------|
| CNN       | 93.22%    | 92.82%     | 93.17%  | 92.22%    |
| ResNet-18 | 99.56%    | 99.89%     | 99.16%  | 99.87%    |
| DenseNet  | 99.11%    | 99.65%     | 98.42%  | 98.67%    |
| ViT       | 93.01%    | 92.82%     | 94.11%  | 93.05%    |

---

## 🛠️ Installation & Usage

### 📦 Clone the repository

```bash
git clone https://github.com/hoangit03/Pneumonia_app.git
```
