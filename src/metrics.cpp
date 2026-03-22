#include "metrics.hpp"

#include <cmath>
#include <algorithm>

double compute_loss(const LogisticModel& model, const Dataset& data, double lambda) {
    // ------------------------------------------------------------
    // Tutorial overview:
    // We compute objective value used in logistic regression training:
    // average cross-entropy loss + L2 regularization penalty.
    //
    // Cross-entropy for one sample:
    //   L = -(y*log(p) + (1-y)*log(1-p))
    // where y is true label (0 or 1), and p is predicted P(y=1).
    //
    // L2 penalty:
    //   (lambda/2) * sum_j w_j^2
    // This discourages very large weights and helps generalization.
    // ------------------------------------------------------------

    // Accumulator for total (un-averaged) data loss across all samples.
    double loss = 0.0;

    for (const auto& sample : data.samples) {
        // For each sample, compute predicted probability p.
        // Predicted probability that y=1 for this sample.
        double p = model.predict_prob(sample.x);
        // Clamp probabilities away from exactly 0 and 1.
        // This avoids log(0), which is undefined and would produce -inf.
        p = std::max(1e-12, std::min(1.0 - 1e-12, p));

        // Binary cross-entropy for one sample:
        // -(y * log(p) + (1 - y) * log(1 - p))
        // If y=1, only -log(p) contributes.
        // If y=0, only -log(1-p) contributes.
        loss += -sample.y * std::log(p) - (1 - sample.y) * std::log(1.0 - p);
    }

    // L2 regularization term: (lambda/2) * ||w||^2
    // where ||w||^2 = sum_j (w_j^2).
    double reg = 0.0;
    for (double wi : model.w) {
        // Add square of each weight to build ||w||^2.
        reg += wi * wi;
    }
    reg *= 0.5 * lambda;

    // Return average loss per sample plus regularization penalty.
    return loss / static_cast<double>(data.samples.size()) + reg;
}

double compute_accuracy(const LogisticModel& model, const Dataset& data) {
    // ------------------------------------------------------------
    // Tutorial overview:
    // Accuracy answers: "How many labels did we get exactly right?"
    // It does not care about confidence, only final class 0/1 match.
    // ------------------------------------------------------------

    // Count of correctly classified samples.
    int correct = 0;

    for (const auto& sample : data.samples) {
        // Compare predicted class to true label.
        if (model.predict_label(sample.x) == sample.y) {
            ++correct;
        }
    }

    // Accuracy = correct predictions / total predictions.
    return static_cast<double>(correct) / static_cast<double>(data.samples.size());
}
