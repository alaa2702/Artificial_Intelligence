#include "LinearRegressionSGD.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>

// Constructor
LinearRegressionSGD::LinearRegressionSGD(double lr, int epochs)
// Initialize learning rate and epochs 
    : lr(lr), epochs(epochs), intercept(0.0) {}

// Compute prediction for a sample
double LinearRegressionSGD::predictOne(const vector<double>& x) {
    double y_pred = intercept;
    for (size_t i = 0; i < weights.size(); i++)
        y_pred += weights[i] * x[i];
    return y_pred;
}

// Train model
void LinearRegressionSGD::fit(const vector<vector<double>>& X, const vector<double>& y) {
    int n = X.size();
    int m = X[0].size();
    weights.assign(m, 0.0);

    // Random shuffle
    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);// Fill with 0,1,...,n-1
    default_random_engine rng(42);

    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle(indices.begin(), indices.end(), rng);

        for (int idx : indices) {
            double y_pred = predictOne(X[idx]);
            double error = y_pred - y[idx];

            // Update weights
            for (int j = 0; j < m; j++)
                weights[j] -= lr * 2 * error * X[idx][j];

            // Update bias
            intercept -= lr * 2 * error;
        }

        // Compute MSE for monitoring
        double mse = 0.0;
        for (int i = 0; i < n; i++) {
            double err = predictOne(X[i]) - y[i];
            mse += err * err;
        }
        mse /= n;
        loss_history.push_back(mse);

        if (epoch % max(1, epochs / 10) == 0)
            cout << "Epoch " << epoch << " MSE = " << mse << endl;
    }
}

// Evaluate R^2 and MSE
void LinearRegressionSGD::evaluate(const vector<vector<double>>& X, const vector<double>& y) {
    int n = X.size();
    double mse = 0.0, ss_res = 0.0, ss_tot = 0.0;
    double y_mean = accumulate(y.begin(), y.end(), 0.0) / n;

    for (int i = 0; i < n; i++) {
        double y_pred = predictOne(X[i]);
        double err = y[i] - y_pred;
        mse += err * err;
        ss_res += err * err;
        ss_tot += pow(y[i] - y_mean, 2);
    }
    mse /= n;
    double r2 = 1 - ss_res / ss_tot;

    cout << "\nFinal training MSE: " << mse << endl;
    cout << "R^2: " << r2 << endl;
}
