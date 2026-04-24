#include "types.hpp"

#include <stdexcept>

namespace hogwild {

std::string to_string(Algorithm a) {
  switch (a) {
    case Algorithm::Serial:
      return "serial";
    case Algorithm::CoarseLock:
      return "coarse_lock";
    case Algorithm::StripedLock:
      return "striped_lock";
    case Algorithm::Hogwild:
      return "hogwild";
    case Algorithm::LocalBatchReduce:
      return "local_batch_reduce";
  }
  return "unknown";
}

std::string to_string(WorkloadType w) {
  switch (w) {
    case WorkloadType::Logistic:
      return "logistic";
    case WorkloadType::MatrixFactorization:
      return "matrix_factorization";
  }
  return "unknown";
}

std::string to_string(LRSchedule s) {
  switch (s) {
    case LRSchedule::Constant:
      return "constant";
    case LRSchedule::EpochDecay:
      return "epoch_decay";
    case LRSchedule::Backoff:
      return "backoff";
  }
  return "unknown";
}

Algorithm parse_algorithm(const std::string& s) {
  if (s == "serial") return Algorithm::Serial;
  if (s == "coarse_lock") return Algorithm::CoarseLock;
  if (s == "striped_lock") return Algorithm::StripedLock;
  if (s == "hogwild") return Algorithm::Hogwild;
  if (s == "local_batch_reduce") return Algorithm::LocalBatchReduce;
  throw std::invalid_argument("Unknown algorithm: " + s);
}

WorkloadType parse_workload(const std::string& s) {
  if (s == "logistic") return WorkloadType::Logistic;
  if (s == "matrix_factorization" || s == "mf") return WorkloadType::MatrixFactorization;
  throw std::invalid_argument("Unknown workload: " + s);
}

LRSchedule parse_schedule(const std::string& s) {
  if (s == "constant") return LRSchedule::Constant;
  if (s == "epoch_decay") return LRSchedule::EpochDecay;
  if (s == "backoff") return LRSchedule::Backoff;
  throw std::invalid_argument("Unknown schedule: " + s);
}

}  // namespace hogwild
