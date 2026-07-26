# Echocardiogram Analysis & Survival Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)

A comprehensive machine learning analysis of echocardiogram data combining **Exploratory Data Analysis**, **K-Means Clustering** for patient risk stratification, and **Logistic Regression** for survival prediction.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Features](#-project-features)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [License](#-license)
- [Author](#-author)

---

## 📊 Overview

This project performs end-to-end analysis on echocardiogram measurements from heart attack patients to:

- **Explore & Visualize** clinical patterns through comprehensive EDA
- **Segment Patients** into 3 risk categories using K-Means clustering
- **Predict Survival** outcomes with Logistic Regression
- **Handle Class Imbalance** using SMOTE oversampling
- **Evaluate Model** with precision, recall, F1-score and confusion matrix

---

## 🏥 Dataset

The echocardiogram dataset contains clinical measurements from heart attack patients.

| Feature | Description |
|---------|-------------|
| `Survival` | Number of months patient survived |
| `Still-alive` | Survival status (0 = deceased, 1 = survived) |
| `Age-at-heart-attack` | Patient age at time of heart attack |
| `Pericardial-effusion` | Pericardial effusion presence |
| `Fractional-shortening` | Heart contraction efficiency |
| `Epss` | E-point septal separation |
| `Lvdd` | Left ventricular diastolic dimension |
| `Wall-motion-score` | Heart wall motion score |
| `Wall-motion-index` | Wall motion index (ratio) |

---

## ✨ Project Features

### 🔍 Exploratory Data Analysis (EDA)
- Comprehensive statistical summaries (mean, std, skewness, kurtosis)
- Missing value detection and imputation
- Age group survival analysis with dual-axis visualizations
- Mortality distribution by age group (stacked bar charts)
- Feature correlation heatmap
- Distribution histograms for all features

### 🎯 Clustering Analysis
- Feature selection: 4 key clinical measurements
- Optimal cluster determination using:
  - **Elbow Method** (WCSS)
  - **Silhouette Score**
- K-Means clustering (k = 3)
- Patient risk categorization:
  - 🔴 **High Risk** — Lowest survival rates
  - 🟠 **Medium Risk** — Moderate survival rates
  - 🟢 **Low Risk** — Highest survival rates
- Interactive Plotly bar charts for feature comparison across clusters

### 🤖 Survival Prediction
- **Logistic Regression** classifier
- **SMOTE** for handling imbalanced target classes
- **RobustScaler** for outlier-resistant feature scaling
- Comprehensive model evaluation:
  - Accuracy Score
  - Classification Report (Precision, Recall, F1-Score)
  - Confusion Matrix

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.8+ |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-learn (KMeans, LogisticRegression, StandardScaler, RobustScaler) |
| **Imbalanced Data** | Imbalanced-learn (SMOTE) |
| **Development** | Git, GitHub |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

**1. Clone the repository**
```bash
git clone https://github.com/Alirezanaghdi-dev/echocardiogram-survival-analysis-prediction.git
cd echocardiogram-survival-analysis-prediction
