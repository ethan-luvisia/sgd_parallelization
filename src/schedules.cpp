#include "schedules.hpp"

#include <cmath>

namespace hogwild {

double learning_rate_for_epoch(const RunConfig& cfg, int epoch) {
  switch (cfg.schedule) {
    case LRSchedule::Constant:
      return cfg.lr;
    case LRSchedule::EpochDecay:
      return cfg.lr / (1.0 + cfg.decay * static_cast<double>(epoch));
    case LRSchedule::Backoff:
      return cfg.lr * std::pow(cfg.backoff_gamma, static_cast<double>(epoch));
  }
  return cfg.lr;
}

}  // namespace hogwild
