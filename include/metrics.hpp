#ifndef METRICS_HPP
#define METRICS_HPP

#include "dataset.hpp"
#include "model.hpp"

// ------------------------------------------------------------
// Tutorial overview:
// Metrics are evaluation tools:
// - Loss: optimization objective (how wrong probabilities are).
// - Accuracy: classification success rate (how often class is correct).
// ------------------------------------------------------------

// Average logistic loss (cross-entropy) + L2 regularization term.
// `lambda` controls regularization strength.
double compute_loss(const LogisticModel& model, const Dataset& data, double lambda);
// Fraction of correct predictions in [0, 1].
double compute_accuracy(const LogisticModel& model, const Dataset& data);

#endif
