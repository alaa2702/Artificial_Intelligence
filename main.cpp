#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "LinearRegressionSGD.h"

using namespace std;

//Utility: read CSV into a 2D vector 
vector<vector<double>> readCSV(const string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }
        data.push_back(row);
    }
    return data;
}

// ---------- Compute mean of each column ----------
vector<double> meanCols(const vector<vector<double>>& X) {
    int n = X.size();
    int m = X[0].size();
    vector<double> mean(m, 0.0);

    for (auto& row : X)
        for (int j = 0; j < m; j++)
            mean[j] += row[j];
    for (int j = 0; j < m; j++)
        mean[j] /= n;
    return mean;
}

// ---------- Compute std of each column ----------
vector<double> stdCols(const vector<vector<double>>& X, const vector<double>& mean) {
    int n = X.size();
    int m = X[0].size();
    vector<double> std(m, 0.0);

    for (auto& row : X)
        for (int j = 0; j < m; j++)
            std[j] += pow(row[j] - mean[j], 2);
    for (int j = 0; j < m; j++)
        std[j] = sqrt(std[j] / n);
    return std;
}

int main() {
    string filename = "MultipleLR.csv"; // change if needed
    auto data = readCSV(filename);

    // Assume last column = target
    int n = data.size();
    int m = data[0].size() - 1;

    vector<vector<double>> X(n, vector<double>(m));
    vector<double> y(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            X[i][j] = data[i][j];
        y[i] = data[i][m];
    }

    // Normalize features
    auto mean = meanCols(X);
    auto stddev = stdCols(X, mean);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            if (stddev[j] != 0)
                X[i][j] = (X[i][j] - mean[j]) / stddev[j];

    // Train model
    LinearRegressionSGD model(0.01, 200);
    model.fit(X, y);
    model.evaluate(X, y);
    
    // Save loss history to file
    ofstream loss_file("loss_history.txt");
    for (const auto& loss : model.loss_history) {
        loss_file << loss << endl;
    }
    loss_file.close();  
    
    cout << "\nIntercept: " << model.intercept << endl;
    cout << "Weights: ";
    for (double w : model.weights) cout << w << " ";
    cout << endl;
    return 0;
}
