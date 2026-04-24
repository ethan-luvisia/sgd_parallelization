#include "train.hpp"
#include "types.hpp"
#include "workloads.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace hogwild;

namespace {

void assert_true(bool cond, const char* msg) {
  if (!cond) throw std::runtime_error(msg);
}

void test_logistic_generator() {
  RunConfig cfg;
  cfg.workload = WorkloadType::Logistic;
  cfg.num_samples = 500;
  cfg.dim = 200;
  cfg.active_k = 8;
  cfg.seed = 11;

  const auto data = make_synthetic_logistic_dataset(cfg);
  assert_true(static_cast<int>(data.samples.size()) == cfg.num_samples, "bad sample count");
  assert_true(data.dim == cfg.dim, "bad dim");
  for (const auto& s : data.samples) {
    assert_true(static_cast<int>(s.indices.size()) == cfg.active_k, "bad active k");
    for (int idx : s.indices) {
      assert_true(idx >= 0 && idx < cfg.dim, "index out of range");
    }
  }
}

void test_serial_determinism() {
  RunConfig cfg;
  cfg.workload = WorkloadType::Logistic;
  cfg.algorithm = Algorithm::Serial;
  cfg.num_samples = 2000;
  cfg.dim = 500;
  cfg.active_k = 6;
  cfg.epochs = 3;
  cfg.seed = 123;
  cfg.init_seed = 999;
  cfg.lr = 0.05;

  const auto data = make_synthetic_logistic_dataset(cfg);
  const auto r1 = train_logistic(data, cfg);
  const auto r2 = train_logistic(data, cfg);
  assert_true(std::abs(r1.final_loss - r2.final_loss) < 1e-12, "serial run not deterministic");
}

void test_coarse_lock_smoke() {
  RunConfig cfg;
  cfg.workload = WorkloadType::Logistic;
  cfg.algorithm = Algorithm::CoarseLock;
  cfg.num_samples = 2000;
  cfg.dim = 500;
  cfg.active_k = 6;
  cfg.epochs = 2;
  cfg.seed = 12;
  cfg.init_seed = 4;
  cfg.lr = 0.05;
  cfg.threads = 4;

  const auto data = make_synthetic_logistic_dataset(cfg);
  const auto r = train_logistic(data, cfg);
  assert_true(std::isfinite(r.final_loss), "coarse lock loss not finite");
}

void test_mf_shapes() {
  RunConfig cfg;
  cfg.workload = WorkloadType::MatrixFactorization;
  cfg.mf_users = 100;
  cfg.mf_items = 150;
  cfg.mf_rank = 8;
  cfg.mf_observations = 3000;
  cfg.seed = 19;

  const auto data = make_synthetic_mf_dataset(cfg);
  assert_true(data.users == 100, "mf users mismatch");
  assert_true(data.items == 150, "mf items mismatch");
  assert_true(data.rank == 8, "mf rank mismatch");
  assert_true(static_cast<int>(data.observations.size()) == 3000, "mf observations mismatch");
}

}  // namespace

int main() {
  test_logistic_generator();
  test_serial_determinism();
  test_coarse_lock_smoke();
  test_mf_shapes();
  std::cout << "All tests passed\n";
  return 0;
}
