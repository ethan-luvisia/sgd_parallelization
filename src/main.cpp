#include "dataset.hpp"
#include "model.hpp"
#include "sgd.hpp"
#include "metrics.hpp"

#include <iostream>

int main() {
    // ------------------------------------------------------------
    // Tutorial overview:
    // 1) Choose dataset size and feature count.
    // 2) Generate synthetic binary classification data.
    // 3) Create a logistic regression model (weights + bias).
    // 4) Configure SGD training hyperparameters.
    // 5) Train model and print final loss/accuracy.
    // ------------------------------------------------------------

    // Number of training examples to generate.
    std::size_t n_samples = 1000;
    // Number of input features per example.
    std::size_t n_features = 20;

    // Create synthetic data for a binary classification task.
    // Seed is fixed so runs are reproducible.
    Dataset data = make_synthetic_dataset(n_samples, n_features, 42);

    // Create logistic regression model with one weight per feature.
    LogisticModel model(n_features);

    // Configure SGD training hyperparameters.
    SGDConfig config;
    // Learning rate controls update step size.
    config.learning_rate = 0.01;
    // L2 regularization strength.
    config.lambda = 0.001;
    // Number of full passes over all samples.
    config.epochs = 25;
    // Shuffle each epoch so SGD does not always see samples in same order.
    config.shuffle_each_epoch = true;
    // Seed for shuffling, again for reproducibility.
    config.seed = 42;

    std::cout << "Starting training...\n";
    // Fit model parameters (weights and bias) using SGD.
    train_sgd(model, data, config);

    // Report final dataset-wide metrics after training.
    // Loss measures "how wrong" probabilities are (lower is better).
    // Accuracy measures fraction of exact class matches (higher is better).
    std::cout << "\nFinal metrics:\n";
    std::cout << "Loss: " << compute_loss(model, data, config.lambda) << "\n";
    std::cout << "Accuracy: " << compute_accuracy(model, data) << "\n";

    return 0;
}
