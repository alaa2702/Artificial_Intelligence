#ifndef LINEAR_REGRESSION_SGD_H
#define LINEAR_REGRESSION_SGD_H

#include <vector>
#include <iostream>

using namespace std;

class LinearRegressionSGD {
public:
    double lr;            // learning rate
    int epochs;         // number of epochs
    double intercept; // bias term
    vector<double> weights; // model weights
    vector<double> loss_history; // history of loss values during training
    // Constructor
    LinearRegressionSGD(double lr = 0.01, int epochs = 100);

    // Compute prediction for a sample
    double predictOne(const vector<double>& x);

    // Train model
    void fit(const vector<vector<double>>& X, const vector<double>& y);

    // Evaluate R^2 and MSE
    void evaluate(const vector<vector<double>>& X, const vector<double>& y);
};

#endif // LINEAR_REGRESSION_SGD_H
