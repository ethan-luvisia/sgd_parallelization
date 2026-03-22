#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>

// ------------------------------------------------------------
// Tutorial overview:
// Utility math functions reused by model/training code.
// ------------------------------------------------------------

// Dot product between two vectors:
// a . b = sum_i (a[i] * b[i]).
double dot(const std::vector<double>& a, const std::vector<double>& b);
// Logistic sigmoid function:
// sigmoid(z) = 1 / (1 + exp(-z)).
double sigmoid(double z);

#endif
