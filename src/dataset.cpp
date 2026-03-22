#include "dataset.hpp"

#include <random>
#include <algorithm>

Dataset make_synthetic_dataset(std::size_t n_samples, std::size_t n_features, unsigned seed) {
    // ------------------------------------------------------------
    // Tutorial overview:
    // This function makes fake training data for two classes.
    // - Class 0 features are sampled around -1.
    // - Class 1 features are sampled around +1.
    // Because centers are different, logistic regression can learn to separate them.
    // ------------------------------------------------------------

    // Dataset object that will hold all generated samples.
    Dataset data;
    // Store feature count so the rest of the code knows expected vector length.
    data.num_features = n_features;

    // Mersenne Twister pseudo-random generator initialized with user seed.
    std::mt19937 rng(seed);
    // Distribution for class 0 features: mean -1, stddev 1.
    std::normal_distribution<double> dist0(-1.0, 1.0);
    // Distribution for class 1 features: mean +1, stddev 1.
    // Using different means creates two separable classes.
    std::normal_distribution<double> dist1(1.0, 1.0);

    // Reserve memory up front to avoid repeated reallocations while pushing samples.
    data.samples.reserve(n_samples);

    for (std::size_t i = 0; i < n_samples; ++i) {
        // Each loop iteration creates exactly one training example.
        // Build one sample at a time.
        Sample s;
        // Allocate space for all features in this sample.
        s.x.resize(n_features);

        // First half of samples are label 0, second half are label 1.
        int label = (i < n_samples / 2) ? 0 : 1;
        s.y = label;

        for (std::size_t j = 0; j < n_features; ++j) {
            // Each feature dimension is sampled independently.
            // Draw each feature from the class-specific Gaussian.
            // This means class 0 tends to have smaller values than class 1.
            s.x[j] = (label == 0) ? dist0(rng) : dist1(rng);
        }

        // Append completed sample to dataset.
        data.samples.push_back(s);
    }

    // Return a fully generated dataset by value.
    return data;
}

void shuffle_dataset(Dataset& data, unsigned seed) {
    // ------------------------------------------------------------
    // Tutorial overview:
    // SGD often works better when sample order is randomized.
    // Shuffling does not change the dataset contents, only their order.
    // ------------------------------------------------------------

    // Deterministic RNG for reproducible shuffling.
    std::mt19937 rng(seed);
    // In-place random permutation of sample order.
    std::shuffle(data.samples.begin(), data.samples.end(), rng);
}
