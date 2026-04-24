#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hogwild {

enum class Algorithm {
  Serial,
  CoarseLock,
  StripedLock,
  Hogwild,
  LocalBatchReduce
};

enum class WorkloadType {
  Logistic,
  MatrixFactorization
};

enum class LRSchedule {
  Constant,
  EpochDecay,
  Backoff
};

struct SparseSample {
  std::vector<int> indices;
  std::vector<double> values;
  int label = 0;
};

struct LogisticDataset {
  std::vector<SparseSample> samples;
  int dim = 0;
  double omega = 0.0;
  double delta = 0.0;
  double rho = 0.0;
};

struct MFObservation {
  int user = 0;
  int item = 0;
  double rating = 0.0;
};

struct MFDataset {
  std::vector<MFObservation> observations;
  int users = 0;
  int items = 0;
  int rank = 0;
  double omega = 0.0;
  double delta = 0.0;
  double rho = 0.0;
};

struct RunConfig {
  WorkloadType workload = WorkloadType::Logistic;
  Algorithm algorithm = Algorithm::Serial;
  LRSchedule schedule = LRSchedule::Constant;

  int threads = 1;
  int epochs = 5;
  int seed = 42;
  int init_seed = 7;
  int trial = 0;

  double lr = 0.01;
  double decay = 0.0;
  double backoff_gamma = 0.9;

  int eval_every_updates = 0;
  bool shuffle_each_epoch = true;

  // logistic
  int num_samples = 10000;
  int dim = 1000;
  int active_k = 10;
  double label_noise = 0.1;
  double hotspot_skew = 0.0;
  double l2 = 1e-4;

  // matrix factorization
  int mf_users = 2000;
  int mf_items = 2000;
  int mf_rank = 32;
  int mf_observations = 200000;
  double mf_noise = 0.05;
  double mf_reg = 1e-4;
  int mf_local_batch = 64;

  // output
  std::string out_summary_csv;
  std::string out_trace_csv;
  std::string run_id;

  // metric target
  bool has_target_loss = false;
  double target_loss = 0.0;
};

struct TracePoint {
  double elapsed_s = 0.0;
  std::uint64_t updates = 0;
  int epoch = 0;
  double loss = 0.0;
};

struct RunResult {
  double runtime_s = 0.0;
  double final_loss = 0.0;
  double throughput_updates_per_s = 0.0;
  bool reached_target = false;
  double time_to_target_s = -1.0;
  std::uint64_t total_updates = 0;
  std::vector<TracePoint> trace;

  // dataset proxies
  double omega = 0.0;
  double delta = 0.0;
  double rho = 0.0;
};

std::string to_string(Algorithm a);
std::string to_string(WorkloadType w);
std::string to_string(LRSchedule s);

Algorithm parse_algorithm(const std::string& s);
WorkloadType parse_workload(const std::string& s);
LRSchedule parse_schedule(const std::string& s);

}  // namespace hogwild
