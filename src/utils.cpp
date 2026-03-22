#include "utils.hpp"

#include <cmath>

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    // ------------------------------------------------------------
    // Tutorial overview:
    // Dot product multiplies matching entries and adds them:
    //   a.b = a0*b0 + a1*b1 + ... + aN*bN
    // In logistic regression, this combines features with their weights.
    // ------------------------------------------------------------

    // Running sum for element-wise products.
    double sum = 0.0;
    // Compute sum_i a[i] * b[i].
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double sigmoid(double z) {
    // ------------------------------------------------------------
    // Tutorial overview:
    // Sigmoid squashes any real number into (0, 1), so it can be a probability.
    //   sigmoid(z) = 1 / (1 + exp(-z))
    //
    // Large positive z -> output near 1
    // Large negative z -> output near 0
    //
    // We use two algebraically equivalent formulas for numerical stability.
    // ------------------------------------------------------------

    // Numerically stable sigmoid implementation:
    // We branch to avoid overflow in exp() for large |z|.
    if (z >= 0.0) {
        // When z is non-negative, exp(-z) is safe and small.
        double ez = std::exp(-z);
        return 1.0 / (1.0 + ez);
    } else {
        // When z is negative, exp(z) is safe and small.
        // Algebraically equivalent to 1/(1+exp(-z)).
        double ez = std::exp(z);
        return ez / (1.0 + ez);
    }
}
