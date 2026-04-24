#pragma once

#include "types.hpp"

namespace hogwild {

double learning_rate_for_epoch(const RunConfig& cfg, int epoch);

}  // namespace hogwild
