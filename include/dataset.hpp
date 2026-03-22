#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>

// ------------------------------------------------------------
// Tutorial overview:
// These types define how training data is stored.
// - A `Sample` is one row: features + one binary label.
// - A `Dataset` is a collection of many samples.
// ------------------------------------------------------------

// One training example used by logistic regression.
struct Sample {
    // Feature vector (input values) for one example.
    // Example: if there are 20 features, this vector has 20 doubles.
    std::vector<double> x;
    // Binary class label:
    // 0 means class 0, 1 means class 1.
    int y;
};

// Full dataset container.
struct Dataset {
    // All examples used for training/evaluation.
    std::vector<Sample> samples;
    // Number of features per sample.
    // This is kept separately for convenience and consistency checks.
    std::size_t num_features = 0;
};

// Build a synthetic binary-classification dataset.
// - n_samples: how many examples to generate.
// - n_features: how many input features per example.
// - seed: fixed random seed for reproducible generation.
Dataset make_synthetic_dataset(std::size_t n_samples, std::size_t n_features, unsigned seed = 42);
// Randomly reorder dataset samples (used to improve SGD training behavior).
void shuffle_dataset(Dataset& data, unsigned seed);

#endif
