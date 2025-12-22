# Artificial Intelligence Portfolio

A comprehensive collection of machine learning and data science projects demonstrating various AI techniques and algorithms.

## 📚 Project Overview

This repository contains multiple projects showcasing different aspects of artificial intelligence:

- **Linear Regression with SGD** (C++ Implementation)
- **C++ Machine Learning Assignment** (Python Notebook)
- **Bike Rental Demand Prediction** (ML Project)

---

## 📁 Repository Structure

```
Artificial_Intelligence/
├── assignment_Py/                    # Linear Regression with SGD (C++)
│   ├── LinearRegressionSGD.cpp      # Main C++ implementation
│   ├── LinearRegressionSGD.h        # Header file
│   ├── main.cpp                     # Entry point
│   ├── MultipleLR.csv               # Input dataset
│   ├── loss_history.txt             # Training loss history
│   ├── plot_loss.gnuplot            # Gnuplot visualization script
│   └── README.md                    # Project documentation
│
├── assingnment_cpp/                  # C++ Assignment (Python Notebook)
│   ├── assignment.ipynb             # Jupyter notebook with analysis
│   └── MultipleLR.csv               # Dataset file
│
├── project/                          # Bike Rental Demand Prediction
│   ├── project.ipynb                # Main ML project notebook
│   ├── Readme.md                    # Project documentation
│   └── Readme.txt:Zone.Identifier   # Windows metadata
│
└── README.md                         # This file
```

---

## 🚀 Projects

### 1. Linear Regression with Stochastic Gradient Descent (SGD)

**Location:** `assignment_Py/`

A C++ implementation of multiple linear regression using SGD algorithm.

**Features:**
- CSV data loading and preprocessing
- Feature normalization (z-score)
- Stochastic Gradient Descent optimization
- MSE and R² performance metrics
- Loss history tracking
- Gnuplot visualization support

**Key Files:**
- `LinearRegressionSGD.cpp` - Core algorithm implementation
- `LinearRegressionSGD.h` - Class definition and interfaces
- `main.cpp` - Training pipeline
- `MultipleLR.csv` - Training dataset
- `loss_history.txt` - Output metrics

**Compilation & Execution:**
```bash
cd assignment_Py
g++ -o LinearRegressionSGD main.cpp LinearRegressionSGD.cpp -std=c++17
./LinearRegressionSGD
```

---

### 2. C++ Machine Learning Assignment

**Location:** `assingnment_cpp/`

An interactive Jupyter notebook demonstrating linear regression with SGD in Python.

**Contents:**
- Step-by-step linear regression implementation
- SGD optimization
- Model evaluation and validation
- Visualization of results

**Technologies:** Python, NumPy, Pandas, Matplotlib

---

### 3. Bike Rental Demand Prediction

**Location:** `project/`

A comprehensive machine learning project predicting hourly bike rental demand.

**Features:**
- Advanced feature engineering (cyclical encoding, interaction terms)
- Multiple model comparison:
  - Linear Regression
  - K-Nearest Neighbors (KNN)
  - Random Forest
- Comprehensive evaluation metrics (R², MAE, RMSE)
- Rich data visualizations

**Dataset Features:**
- **Temporal:** Hour, month, season, day of week, working day, holiday
- **Weather:** Temperature, humidity, wind speed, weather conditions
- **Target:** Hourly bike rental count

**Technologies:** Python, Scikit-learn, Pandas, Matplotlib, Seaborn

---

## 💻 Requirements

### For C++ Projects (assignment_Py/)
- C++17 or higher
- Standard C++ compiler (g++, clang)

### For Python Projects (assingnment_cpp/, project/)
- Python 3.7+
- Jupyter Notebook / JupyterLab
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn

Install Python dependencies:
```bash
pip install jupyter numpy pandas matplotlib seaborn scikit-learn
```

---

## 🔧 Getting Started

### Clone the Repository
```bash
git clone https://github.com/alaa2702/Artificial_Intelligence.git
cd Artificial_Intelligence
```

### Running Each Project

**Linear Regression with SGD (C++):**
```bash
cd assignment_Py
g++ -o LinearRegressionSGD main.cpp LinearRegressionSGD.cpp -std=c++17
./LinearRegressionSGD
```

**Python Notebooks:**
```bash
cd assingnment_cpp
jupyter notebook assignment.ipynb

# Or for the Bike Rental project:
cd ../project
jupyter notebook project.ipynb
```

---

## 📊 Learning Outcomes

This repository demonstrates:

1. **Machine Learning Fundamentals**
   - Linear regression concepts
   - Optimization algorithms (SGD)
   - Feature engineering and normalization

2. **Algorithm Implementation**
   - Building ML models from scratch in C++
   - Implementing gradient descent variants
   - Model evaluation and validation

3. **Data Analysis**
   - CSV data processing
   - Feature scaling and transformation
   - Performance metrics (MSE, R², MAE, RMSE)

4. **Practical ML Development**
   - End-to-end ML pipeline
   - Model comparison and selection
   - Results visualization

---

## 📈 Key Concepts Covered

- **Regression Analysis**
- **Stochastic Gradient Descent (SGD)**
- **Feature Normalization & Scaling**
- **Loss Functions & Error Metrics**
- **Train-Test Evaluation**
- **Hyperparameter Tuning**
- **Ensemble Methods (Random Forest)**
- **K-Nearest Neighbors (KNN)**

---

## 📝 Notes

- The `assingnment_cpp/` folder appears to be mislabeled; it contains Python notebooks despite the name.
- All projects use the same dataset structure (CSV format) for consistency.
- Detailed documentation for each project is available in their respective README files.

---

## 👤 Author

**Alaa** (@alaa2702)

---

## 📄 License

This repository is part of an AI learning portfolio. Feel free to reference or adapt for educational purposes.

---

## 🤝 Contributing

Suggestions and improvements are welcome! Feel free to:
- Report issues
- Suggest enhancements
- Submit pull requests

---

## 📧 Contact

For questions or collaboration, please reach out through the repository.

---

**Last Updated:** December 2025
