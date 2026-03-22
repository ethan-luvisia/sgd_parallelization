#include "model.hpp"
#include "utils.hpp"

LogisticModel::LogisticModel(std::size_t num_features)
    // ------------------------------------------------------------
    // Tutorial overview:
    // `w` is the learned weight vector. It has one value per feature.
    // `b` is the bias (intercept), a constant offset.
    // Initializing to zeros means the model starts "neutral" and learns from data.
    // ------------------------------------------------------------
    
    // Initialize all weights to 0 and bias to 0.
    // Starting at zero is acceptable for logistic regression because optimization
    // is convex (there is a single global optimum for this objective).
    : w(num_features, 0.0), b(0.0) {}

double LogisticModel::predict_prob(const std::vector<double>& x) const {
    // ------------------------------------------------------------
    // Tutorial overview:
    // Step 1: Compute linear score z = w.x + b (dot product + bias).
    // Step 2: Pass z through sigmoid to map any real value to [0, 1].
    // Result is interpreted as probability that class is 1.
    // ------------------------------------------------------------

    // Linear score z = w.x + b, then map to probability with sigmoid.
    return sigmoid(dot(w, x) + b);
}

int LogisticModel::predict_label(const std::vector<double>& x) const {
    // ------------------------------------------------------------
    // Tutorial overview:
    // Convert soft probability into hard class label.
    // 0.5 threshold means:
    // - >= 50% chance of class 1 -> predict 1
    // - otherwise -> predict 0
    // ------------------------------------------------------------

    // Standard binary threshold: probability >= 0.5 -> class 1, else class 0.
    return predict_prob(x) >= 0.5 ? 1 : 0;
}
