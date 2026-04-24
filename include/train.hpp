#pragma once

#include "types.hpp"

namespace hogwild {

RunResult train_logistic(const LogisticDataset& data, const RunConfig& cfg);
RunResult train_matrix_factorization(const MFDataset& data, const RunConfig& cfg);

}  // namespace hogwild
