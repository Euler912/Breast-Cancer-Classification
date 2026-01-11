# 🎗️ Breast Cancer Classification (PyTorch & Scikit-Learn)

This repository contains a comprehensive implementation of classification algorithms to diagnose breast cancer tumors using the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

The project demonstrates a rigorous Data Science lifecycle: from Exploratory Data Analysis (EDA) and feature selection to the implementation of a custom **Neural Network in PyTorch** to capture non-linear patterns that classical linear models might miss.

## 🚀 Project Overview

The objective is to build and compare robust classification models to predict whether a tumor is **Malignant** (Cancerous) or **Benign** (Non-cancerous).

The analysis focuses on **minimizing False Negatives** (predicting Benign when it's actually Malignant), which is the most critical metric in medical diagnosis contexts.

### Key Features
* **Deep Learning Implementation:** Built a custom Feed-Forward Neural Network (MLP) using `torch.nn.Module` with manual training loops.
* **Data Preprocessing:** Tensor conversion, feature scaling (StandardScaler) for convergence stability, and encoding.
* **Feature Engineering:** Analysis of feature importance to reduce dimensionality and noise.
* **Evaluation Metrics:** Comparison using Accuracy, Recall (Sensitivity), and ROC-AUC curves.

## 🛠️ Tech Stack

* **Deep Learning:** `PyTorch` (torch, torch.nn, torch.optim)
* **Machine Learning:** `scikit-learn`
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`

## 🧠 Methodology & Models

The notebook follows a structured pipeline:

1.  **Data Loading & Tensor Conversion:**
    * Loading the Wisconsin dataset and converting generic arrays to PyTorch Tensors.
2.  **Feature Selection:**
    * Analyzing correlation to remove multicollinear features that affect model weights.
3.  **Model Training & Comparison:**
    * **Logistic Regression:** Implemented as a baseline to test linear separability.
    * **PyTorch Neural Network:** A custom Multi-Layer Perceptron (MLP) designed to optimize the decision boundary for non-linear feature interactions.
4.  **Performance Analysis:**
    * Tracking Loss convergence over epochs.
    * Confusion Matrix visualization to audit False Negatives.
 
