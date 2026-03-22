#include "sgd.hpp"
#include "metrics.hpp"

#include <random>
#include <algorithm>
#include <iostream>

void train_sgd(LogisticModel& model, Dataset& data, const SGDConfig& config) {
    // ------------------------------------------------------------
    // Tutorial overview:
    // This is stochastic gradient descent (SGD) training.
    //
    // Key idea:
    // - Read one sample.
    // - Compute prediction error.
    // - Move each parameter a small step in the direction that reduces loss.
    //
    // Repeat this over all samples for multiple epochs.
    // ------------------------------------------------------------

    // RNG used for sample shuffling.
    std::mt19937 rng(config.seed);

    // Outer loop: one epoch = one full pass over the training dataset.
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        if (config.shuffle_each_epoch) {
            // Shuffle to reduce ordering bias in SGD updates.
            std::shuffle(data.samples.begin(), data.samples.end(), rng);
        }

        // SGD processes one sample at a time (stochastic updates).
        for (const auto& sample : data.samples) {
            // --- Forward pass ---
            // Forward pass: predicted P(y=1|x).
            double p = model.predict_prob(sample.x);
            // --- Error signal ---
            // For logistic loss, derivative wrt linear score z is (p - y).
            double err = p - sample.y;

            for (std::size_t j = 0; j < model.w.size(); ++j) {
                // --- Per-weight gradient ---
                // Gradient wrt weight j:
                // dL/dw_j = (p - y) * x_j + lambda * w_j
                // First part is data term, second is L2 regularization gradient.
                double grad = err * sample.x[j] + config.lambda * model.w[j];
                // --- Gradient descent step ---
                // Gradient descent update:
                // w_j = w_j - learning_rate * grad
                model.w[j] -= config.learning_rate * grad;
            }

            // --- Bias update ---
            // Bias gradient has no L2 penalty:
            // dL/db = (p - y)
            model.b -= config.learning_rate * err;
        }

        // After processing all samples once, report epoch summary.
        // Track training progress after each epoch.
        double loss = compute_loss(model, data, config.lambda);
        double acc = compute_accuracy(model, data);

        std::cout << "Epoch " << (epoch + 1)
                  << " | Loss: " << loss
                  << " | Accuracy: " << acc
                  << "\n";
    }
}
