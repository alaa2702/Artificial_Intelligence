# 🚴 Bike Rental Demand Prediction

A machine learning project that predicts hourly bike rental demand using environmental and temporal features. This project implements and compares three regression models: Linear Regression, K-Nearest Neighbors, and Random Forest.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Feature Engineering](#feature-engineering)
- [Results](#results)
- [Visualizations](#visualizations)

## 🎯 Overview

This project analyzes bike-sharing data to predict rental demand based on various factors including weather conditions, time of day, and seasonal patterns. The goal is to help bike-sharing systems optimize their fleet distribution and improve service availability.

## ✨ Features

- **Advanced Feature Engineering**: Cyclical encoding for temporal features, one-hot encoding for categories, and interaction terms
- **Multiple ML Models**: Comparison of Linear Regression, KNN, and Random Forest
- **Comprehensive Evaluation**: R² score, MAE, and RMSE metrics
- **Rich Visualizations**: Distribution plots, actual vs predicted scatter plots, and feature importance analysis

## 📊 Dataset

The project expects a CSV file containing bike rental data with the following features:

**Temporal Features:**
- `instant`: Record index
- `dteday`: Date string
- `hr`: Hour of the day (0-23)
- `mnth`: Month (1-12)
- `season`: Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)
- `yr`: Year
- `weekday`: Day of week
- `workingday`: Working day (0/1)
- `holiday`: Holiday (0/1)

**Weather Features:**
- `weathersit`: Weather situation (1-4, from clear to heavy rain/snow)
- `temp`: Normalized temperature
- `atemp`: Normalized feeling temperature
- `hum`: Normalized humidity
- `windspeed`: Normalized wind speed

**Target Variable:**
- `cnt`: Total bike rentals (casual + registered)

**Note:** The columns `casual` and `registered` are dropped to prevent data leakage.

## 🔧 Installation

### Requirements

```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```

### For Google Colab

All required libraries are pre-installed. Simply run the notebook directly.

## 🚀 Usage

1. **Upload Your Dataset**: When prompted, upload your CSV file containing bike rental data
2. **Run the Notebook**: Execute all cells sequentially
3. **View Results**: Examine the printed metrics and visualizations

```python
# The script will automatically:
# 1. Load and preprocess your data
# 2. Engineer features
# 3. Train three models
# 4. Display performance metrics
# 5. Generate visualizations
```

## 🤖 Models

### 1. Linear Regression
- **Purpose**: Baseline model
- **Strengths**: Fast, interpretable
- **Use Case**: Checking for linear relationships

### 2. K-Nearest Neighbors (k=5)
- **Purpose**: Pattern matching
- **Strengths**: Captures local patterns
- **Note**: Requires feature scaling

### 3. Random Forest (100 trees)
- **Purpose**: Complex pattern detection
- **Strengths**: Handles non-linearity, provides feature importance
- **Best For**: Capturing interactions between features

## 🔨 Feature Engineering

### 1. Cyclical Encoding
Converts circular features (hour, month) into sine/cosine pairs to properly represent their cyclical nature:

```python
hr_sin = sin(2π × hour / 24)
hr_cos = cos(2π × hour / 24)
```

**Why?** Hour 23 and Hour 0 are adjacent on a clock, but numerically they're 23 units apart.

### 2. One-Hot Encoding
Converts categorical variables (season, weather) into binary features:

```python
season_1 → [1, 0, 0, 0]
season_2 → [0, 1, 0, 0]
```

**Why?** Prevents the model from assuming ordinal relationships where none exist.

### 3. Interaction Terms
Creates composite features to capture combined effects:

```python
temp_hum_interaction = temperature × humidity
```

**Why?** High temperature alone is comfortable, but high temp + high humidity is uncomfortable.

## 📈 Results

The models are evaluated using three metrics:

- **R² Score**: Proportion of variance explained (0 to 1, higher is better)
- **MAE** (Mean Absolute Error): Average prediction error in number of bikes
- **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily

Example output:
```
--- Random Forest ---
R^2 Score: 0.9234 (Closer to 1.0 is better)
MAE:       45.23 (Average error in bikes)
RMSE:      67.89
```

## 📊 Visualizations

The project generates several visualizations:

1. **Feature Distributions**: Histograms showing the spread of each feature
2. **Target Distribution**: Distribution of bike rental counts
3. **Actual vs Predicted**: Scatter plots comparing predictions to ground truth
4. **Feature Importance**: Bar chart showing the top 10 drivers of demand
5. **Model Comparison**: Bar chart comparing R² scores across models

## 🏆 Key Insights

- **Hour of day** is typically the strongest predictor of demand
- **Temperature** and **working day status** significantly impact rentals
- **Random Forest** usually achieves the best performance for this type of data
- **Feature engineering** (especially cyclical encoding) dramatically improves model accuracy

## 📝 Notes

- The script automatically removes leakage columns (`casual`, `registered`)
- Feature scaling is applied for KNN but not Random Forest (RF handles raw features well)
- The train/test split is 80/20 with `random_state=42` for reproducibility

## 🤝 Contributing

Feel free to fork this project and experiment