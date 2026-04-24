#pragma once

#include "types.hpp"

namespace hogwild {

void append_summary_csv(const RunConfig& cfg, const RunResult& result);
void append_trace_csv(const RunConfig& cfg, const RunResult& result);

}  // namespace hogwild
