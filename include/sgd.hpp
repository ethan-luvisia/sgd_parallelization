#ifndef SGD_HPP
#define SGD_HPP

#include "dataset.hpp"
#include "model.hpp"

// ------------------------------------------------------------
// Tutorial overview:
// SGD hyperparameters control how training behaves:
// - learning_rate: step size
// - lambda: regularization amount
// - epochs: number of full passes over data
// - shuffle_each_epoch: randomize sample order each pass
// ------------------------------------------------------------

// Hyperparameters and options for stochastic gradient descent (SGD).
struct SGDConfig {
    // Step size for each gradient update.
    double learning_rate = 0.01;
    // L2 regularization strength.
    double lambda = 0.0;
    // Number of full passes through the dataset.
    int epochs = 20;
    // Whether to reshuffle sample order before each epoch.
    bool shuffle_each_epoch = true;
    // Random seed used for shuffling.
    unsigned seed = 42;
};

// Train logistic regression weights and bias with SGD.
void train_sgd(LogisticModel& model, Dataset& data, const SGDConfig& config);

#endif
