# Linear Regression with Stochastic Gradient Descent (SGD)

## Project Overview
This project implements **multiple linear regression** using **Stochastic Gradient Descent (SGD)** in C++. The model trains on CSV data, normalizes features, optimizes weights through gradient descent, and outputs performance metrics including MSE and R² scores.

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [How It Works](#how-it-works)
4. [Compilation & Execution](#compilation--execution)
5. [Input Format](#input-format)
6. [Output Files](#output-files)
7. [Parameter Tuning](#parameter-tuning)
8. [Mathematical Details](#mathematical-details)
9. [Limitations & Future Work](#limitations--future-work)

---

## Features

✓ **CSV Data Loading** — Read training data directly from CSV files  
✓ **Feature Normalization** — Standardize features (z-score normalization)  
✓ **SGD Training** — Stochastic gradient descent with random shuffling  
✓ **Model Evaluation** — Compute MSE and R² metrics  
✓ **Loss Tracking** — Record loss history for monitoring convergence  
✓ **Configurable Parameters** — Adjust learning rate and epochs

---

## Project Structure

```
/home/alaa2/AI/assignment/
├── linear_regression_sgd.cpp    # Main C++ implementation
├── MultipleLR.csv               # Input dataset
├── loss_history.txt             # Output: loss values per epoch
├── README.md                    # This documentation
└── plot_loss.gnuplot           # Optional: Gnuplot script for visualization
```

---

## How It Works

### 1. Data Loading (`readCSV`)
Reads a CSV file and stores it in a 2D vector:
- **Columns 0 to m-1:** Features (X)
- **Column m:** Target variable (y)

### 2. Feature Normalization
**Z-score normalization** is applied to each feature:
```
X_normalized[i][j] = (X[i][j] - mean[j]) / std[j]
```

**Benefits:**
- Prevents features with large scales from dominating
- Improves convergence speed
- Makes learning rate less sensitive

### 3. Model Training (SGD)

```cpp
class LinearRegressionSGD {
    double lr;                      // Learning rate
    int epochs;                     // Number of iterations
    double intercept;               // Bias term
    vector<double> weights;         // Feature weights
    vector<double> loss_history;    // MSE per epoch
};
```

**Training Loop:**
```
for each epoch:
    shuffle training samples randomly
    for each sample (x, y):
        y_pred = intercept + Σ(weight[j] * x[j])
        error = y_pred - y
        
        // Update weights
        for each feature j:
            weight[j] -= lr * 2 * error * x[j]
        
        // Update intercept
        intercept -= lr * 2 * error
    
    // Compute epoch MSE
    mse = (1/n) * Σ(error²)
    loss_history.append(mse)
```

### 4. Model Evaluation
After training, the model computes:
- **MSE (Mean Squared Error):** Average of squared prediction errors
- **R² Score:** Proportion of variance explained (0 to 1, higher is better)

---

## Compilation & Execution

### Prerequisites
- **g++ compiler** (C++17 or later)
- **Linux/Unix environment**

### Compile
```bash
g++ -std=c++17 linear_regression_sgd.cpp -O2 -o lr
```

### Run
```bash
./lr
```

### Sample Output
```
Epoch 0 MSE = 125.456
Epoch 20 MSE = 45.123
Epoch 40 MSE = 28.567
...
Epoch 200 MSE = 12.345

Final training MSE: 12.345
R^2: 0.8765

Intercept: 2.3456
Weights: 1.2345 -0.4567 0.8901
```

---

## Input Format

**File:** `MultipleLR.csv`

CSV format with comma-separated values:
- **First n-1 columns:** Features
- **Last column:** Target variable (y)

### Example
```
feature1,feature2,feature3,target
1.5,2.3,-0.8,10.5
2.1,3.2,1.2,15.3
0.9,1.5,0.5,8.2
3.2,4.1,-0.3,18.9
```

**Requirements:**
- No header row (all numeric data)
- No missing values
- Consistent number of columns in each row

---

## Output Files

### 1. Console Output
- **Epoch-wise MSE:** Printed every 10% of total epochs
- **Final Metrics:** MSE and R² score after training
- **Model Parameters:** Intercept and weights

### 2. `loss_history.txt`
Text file with one loss (MSE) value per line, in order of epochs:
```
125.456
115.234
105.123
...
12.345
```

**Use for:** Plotting loss curves, analyzing convergence

---

## Parameter Tuning

### Learning Rate (`lr`)
```cpp
LinearRegressionSGD model(0.01, 200);  // lr = 0.01
```

| Value | Behavior |
|-------|----------|
| **Too high** (e.g., 0.1) | Fast but unstable; loss may diverge |
| **Optimal** (e.g., 0.01) | Steady convergence |
| **Too low** (e.g., 0.001) | Slow convergence; may not reach minimum |

**Recommendation:** Start with 0.01; adjust if loss doesn't decrease smoothly.

### Epochs
```cpp
LinearRegressionSGD model(0.01, 200);  // epochs = 200
```

| Value | Effect |
|-------|--------|
| **Low** (50–100) | Faster training but may underfit |
| **Medium** (200–500) | Good balance for most datasets |
| **High** (1000+) | Risk of overfitting; diminishing returns |

**Recommendation:** Plot `loss_history.txt` to find where loss plateaus.

---

## Mathematical Details

### Linear Regression Model
```
y_pred = intercept + Σ(weight[j] * x[j])
       = b₀ + w₁x₁ + w₂x₂ + ... + wₘxₘ
```

### Loss Function (MSE)
```
MSE = (1/n) * Σᵢ(yᵢ - ŷᵢ)²
```

### Gradient Descent Update
For each sample, the gradient of MSE with respect to weight j:
```
∂MSE/∂wⱼ = 2 * error * xⱼ
```

Weight update rule (gradient descent):
```
wⱼ ← wⱼ - lr * ∂MSE/∂wⱼ
   = wⱼ - lr * 2 * error * xⱼ
```

### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)

where:
  SS_res = Σ(y - ŷ)²           (residual sum of squares)
  SS_tot = Σ(y - ȳ)²           (total sum of squares)
  ȳ = mean(y)
```

**Interpretation:**
- R² = 1: Perfect fit
- R² = 0.8: Model explains 80% of variance (good)
- R² < 0: Model worse than mean baseline

---

## Example Walkthrough

Given dataset:
```
x1, x2, y
1,  2,  5
2,  3,  8
3,  4,  11
```

1. **Load & Normalize:** Features are standardized
2. **Initialize:** weights = [0, 0], intercept = 0
3. **Train (200 epochs):**
   - Epoch 0: MSE ≈ 40.0
   - Epoch 50: MSE ≈ 2.5
   - Epoch 200: MSE ≈ 0.1
4. **Output:** 
   - Intercept ≈ 2.0
   - Weights ≈ [1.5, 1.5]
   - R² ≈ 0.99

---

## Limitations

1. **No Train/Test Split:** Uses same data for training and evaluation
   - *Fix:* Add data splitting logic
   
2. **No Regularization:** No L1/L2 penalty to prevent overfitting
   - *Fix:* Implement Ridge (L2) or Lasso (L1) regularization
   
3. **Linear Only:** Cannot capture non-linear relationships
   - *Fix:* Use polynomial features or neural networks
   
4. **Full Epoch Evaluation:** Computes MSE on entire dataset every epoch
   - *Impact:* Slower for large datasets
   - *Fix:* Use mini-batch SGD
   
5. **No Convergence Check:** Trains for fixed epochs regardless of convergence
   - *Fix:* Add early stopping when loss improvement < threshold

6. **Single-threaded:** No parallelization
   - *Fix:* Use OpenMP for parallel updates

---

## Future Enhancements

### High Priority
- [ ] Train/validation/test split
- [ ] Early stopping based on validation loss
- [ ] Ridge regression (L2 regularization)
- [ ] Plotting integration (gnuplot or matplotlib)

### Medium Priority
- [ ] Mini-batch SGD
- [ ] Learning rate scheduling (decay)
- [ ] Feature selection
- [ ] Cross-validation

### Low Priority
- [ ] Lasso regression (L1)
- [ ] Elastic Net (L1+L2)
- [ ] Polynomial feature expansion
- [ ] Parallel SGD

---

## Troubleshooting

### Issue: Loss diverges (increases over time)
**Cause:** Learning rate too high  
**Solution:** Reduce `lr` to 0.001 or 0.005

### Issue: Loss decreases very slowly
**Cause:** Learning rate too low  
**Solution:** Increase `lr` to 0.05 or 0.1

### Issue: Poor R² score
**Cause:** Features don't linearly relate to target  
**Solution:** Check feature-target correlation; add polynomial features

### Issue: Compilation error
**Ensure:** g++ version supports C++17
```bash
g++ --version  # Should show >= 7.0
```

---

## References

- [Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/sgd.html)
- [Linear Regression Mathematics](https://en.wikipedia.org/wiki/Linear_regression)
- [Feature Normalization](https://en.wikipedia.org/wiki/Standardization_(statistics))
- [R² Coefficient](https://en.wikipedia.org/wiki/Coefficient_of_determination)

---

## Author Notes

This implementation prioritizes **clarity and correctness** over performance. For production use, consider:
- Using optimized libraries (Eigen, Armadillo, TensorFlow)
- Adding GPU acceleration
- Implementing distributed training for large-scale data

---

**Last Updated:** December 2025  
**License:** MIT (or as specified by your institution)
