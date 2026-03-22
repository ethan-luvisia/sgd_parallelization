#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>

// ------------------------------------------------------------
// Tutorial overview:
// Logistic regression learns parameters to model:
//   P(y=1 | x) = sigmoid(w.x + b)
// where:
// - x is input feature vector
// - w is learned weight vector
// - b is learned bias
// ------------------------------------------------------------

// Logistic regression model:
// probability = sigmoid(w . x + b)
class LogisticModel {
public:
    // Initialize a model with `num_features` weights.
    // Weights and bias start at 0.0.
    explicit LogisticModel(std::size_t num_features);

    // Return predicted probability P(y=1 | x) in [0, 1].
    double predict_prob(const std::vector<double>& x) const;
    // Convert probability into a hard class prediction (0 or 1) using 0.5 cutoff.
    int predict_label(const std::vector<double>& x) const;

    // Weight vector (one coefficient per input feature).
    std::vector<double> w;
    // Bias/intercept term (constant added to w.x).
    double b;
};

#endif
