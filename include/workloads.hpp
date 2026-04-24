#pragma once

#include "types.hpp"

namespace hogwild {

LogisticDataset make_synthetic_logistic_dataset(const RunConfig& cfg);
MFDataset make_synthetic_mf_dataset(const RunConfig& cfg);

}  // namespace hogwild
