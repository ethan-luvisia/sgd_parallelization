#include "train.hpp"

#include "schedules.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <memory>
#include <vector>

#if HOGWILD_HAS_OPENMP
#include <omp.h>
#endif

namespace hogwild {
namespace {

double sigmoid(double x) {
  if (x >= 0.0) {
    const double z = std::exp(-x);
    return 1.0 / (1.0 + z);
  }
  const double z = std::exp(x);
  return z / (1.0 + z);
}

double logistic_loss_dense(const LogisticDataset& data, const std::vector<double>& w, double b,
                           double l2) {
  double total = 0.0;
  for (const auto& s : data.samples) {
    double z = b;
    for (std::size_t t = 0; t < s.indices.size(); ++t) {
      z += w[static_cast<std::size_t>(s.indices[t])] * s.values[t];
    }
    double p = sigmoid(z);
    p = std::clamp(p, 1e-12, 1.0 - 1e-12);
    total += (s.label == 1) ? -std::log(p) : -std::log(1.0 - p);
  }
  double reg = 0.0;
  for (double wi : w) reg += wi * wi;
  return total / static_cast<double>(data.samples.size()) + 0.5 * l2 * reg;
}

double logistic_loss_atomic(const LogisticDataset& data, const std::atomic<double>* w,
                            const std::atomic<double>& b, int dim, double l2) {
  double total = 0.0;
  for (const auto& s : data.samples) {
    double z = b.load(std::memory_order_relaxed);
    for (std::size_t t = 0; t < s.indices.size(); ++t) {
      z += w[static_cast<std::size_t>(s.indices[t])].load(std::memory_order_relaxed) * s.values[t];
    }
    double p = sigmoid(z);
    p = std::clamp(p, 1e-12, 1.0 - 1e-12);
    total += (s.label == 1) ? -std::log(p) : -std::log(1.0 - p);
  }
  double reg = 0.0;
  for (int i = 0; i < dim; ++i) {
    const double v = w[static_cast<std::size_t>(i)].load(std::memory_order_relaxed);
    reg += v * v;
  }
  return total / static_cast<double>(data.samples.size()) + 0.5 * l2 * reg;
}

inline void atomic_add(std::atomic<double>& target, double delta) {
  double old = target.load(std::memory_order_relaxed);
  while (!target.compare_exchange_weak(old, old + delta, std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
  }
}

void update_dense(const SparseSample& s, std::vector<double>& w, double& b, double lr, double l2) {
  double z = b;
  for (std::size_t t = 0; t < s.indices.size(); ++t) {
    z += w[static_cast<std::size_t>(s.indices[t])] * s.values[t];
  }
  const double grad_common = sigmoid(z) - static_cast<double>(s.label);
  for (std::size_t t = 0; t < s.indices.size(); ++t) {
    const std::size_t idx = static_cast<std::size_t>(s.indices[t]);
    const double grad = grad_common * s.values[t] + l2 * w[idx];
    w[idx] -= lr * grad;
  }
  b -= lr * grad_common;
}

void update_hogwild(const SparseSample& s, std::atomic<double>* w, std::atomic<double>& b,
                    double lr, double l2) {
  double z = b.load(std::memory_order_relaxed);
  for (std::size_t t = 0; t < s.indices.size(); ++t) {
    z += w[static_cast<std::size_t>(s.indices[t])].load(std::memory_order_relaxed) * s.values[t];
  }
  const double grad_common = sigmoid(z) - static_cast<double>(s.label);
  for (std::size_t t = 0; t < s.indices.size(); ++t) {
    const std::size_t idx = static_cast<std::size_t>(s.indices[t]);
    const double cur = w[idx].load(std::memory_order_relaxed);
    const double grad = grad_common * s.values[t] + l2 * cur;
    atomic_add(w[idx], -lr * grad);
  }
  atomic_add(b, -lr * grad_common);
}

std::vector<int> make_order(int n) {
  std::vector<int> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  return order;
}

}  // namespace

RunResult train_logistic(const LogisticDataset& data, const RunConfig& cfg) {
  if (data.samples.empty()) {
    throw std::invalid_argument("empty logistic dataset");
  }

  RunResult out;
  out.omega = data.omega;
  out.delta = data.delta;
  out.rho = data.rho;

  std::mt19937_64 init_rng(static_cast<std::uint64_t>(cfg.init_seed));
  std::normal_distribution<double> init_dist(0.0, 0.01);

  std::vector<double> w(static_cast<std::size_t>(data.dim), 0.0);
  for (double& wi : w) wi = init_dist(init_rng);
  double b = init_dist(init_rng);

  std::unique_ptr<std::atomic<double>[]> w_atomic;
  std::atomic<double> b_atomic{0.0};
  if (cfg.algorithm == Algorithm::Hogwild) {
    w_atomic = std::make_unique<std::atomic<double>[]>(static_cast<std::size_t>(data.dim));
    for (std::size_t i = 0; i < w.size(); ++i) {
      w_atomic[i].store(w[i], std::memory_order_relaxed);
    }
    b_atomic.store(b, std::memory_order_relaxed);
  }

  std::vector<int> order = make_order(static_cast<int>(data.samples.size()));
  std::mt19937_64 shuffle_rng(static_cast<std::uint64_t>(cfg.seed + 1337));

  auto t0 = std::chrono::steady_clock::now();
  std::uint64_t total_updates = 0;

  auto record_eval = [&](int epoch) {
    TracePoint p;
    p.epoch = epoch;
    p.updates = total_updates;
    const auto now = std::chrono::steady_clock::now();
    p.elapsed_s = std::chrono::duration<double>(now - t0).count();
    if (cfg.algorithm == Algorithm::Hogwild) {
      p.loss = logistic_loss_atomic(data, w_atomic.get(), b_atomic, data.dim, cfg.l2);
    } else {
      p.loss = logistic_loss_dense(data, w, b, cfg.l2);
    }
    out.trace.push_back(p);
    out.final_loss = p.loss;
    if (cfg.has_target_loss && !out.reached_target && p.loss <= cfg.target_loss) {
      out.reached_target = true;
      out.time_to_target_s = p.elapsed_s;
    }
  };

  record_eval(0);

  const int stripes = 256;
  std::vector<std::mutex> striped_locks(static_cast<std::size_t>(stripes));
  std::mutex coarse_mutex;

  for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
    if (cfg.shuffle_each_epoch) {
      std::shuffle(order.begin(), order.end(), shuffle_rng);
    }

    const double lr = learning_rate_for_epoch(cfg, epoch - 1);

    if (cfg.algorithm == Algorithm::Serial || cfg.threads <= 1 || !HOGWILD_HAS_OPENMP) {
      for (int pos = 0; pos < static_cast<int>(order.size()); ++pos) {
        const auto& s = data.samples[static_cast<std::size_t>(order[static_cast<std::size_t>(pos)])];
        if (cfg.algorithm == Algorithm::Hogwild) {
          update_hogwild(s, w_atomic.get(), b_atomic, lr, cfg.l2);
        } else {
          update_dense(s, w, b, lr, cfg.l2);
        }
        ++total_updates;
      }
    } else if (cfg.algorithm == Algorithm::CoarseLock) {
#if HOGWILD_HAS_OPENMP
      std::atomic<int> next{0};
#pragma omp parallel num_threads(cfg.threads)
      {
        while (true) {
          const int pos = next.fetch_add(1, std::memory_order_relaxed);
          if (pos >= static_cast<int>(order.size())) break;
          const auto& s = data.samples[static_cast<std::size_t>(order[static_cast<std::size_t>(pos)])];
          {
            std::lock_guard<std::mutex> lk(coarse_mutex);
            update_dense(s, w, b, lr, cfg.l2);
          }
        }
      }
      total_updates += static_cast<std::uint64_t>(order.size());
#endif
    } else if (cfg.algorithm == Algorithm::StripedLock) {
#if HOGWILD_HAS_OPENMP
      std::atomic<int> next{0};
#pragma omp parallel num_threads(cfg.threads)
      {
        std::vector<int> lock_ids;
        while (true) {
          const int pos = next.fetch_add(1, std::memory_order_relaxed);
          if (pos >= static_cast<int>(order.size())) break;
          const auto& s = data.samples[static_cast<std::size_t>(order[static_cast<std::size_t>(pos)])];

          lock_ids.clear();
          lock_ids.reserve(s.indices.size() + 1U);
          lock_ids.push_back(0);
          for (const int idx : s.indices) {
            lock_ids.push_back(1 + (idx % (stripes - 1)));
          }
          std::sort(lock_ids.begin(), lock_ids.end());
          lock_ids.erase(std::unique(lock_ids.begin(), lock_ids.end()), lock_ids.end());

          for (int lid : lock_ids) striped_locks[static_cast<std::size_t>(lid)].lock();
          update_dense(s, w, b, lr, cfg.l2);
          for (auto it = lock_ids.rbegin(); it != lock_ids.rend(); ++it) {
            striped_locks[static_cast<std::size_t>(*it)].unlock();
          }
        }
      }
      total_updates += static_cast<std::uint64_t>(order.size());
#endif
    } else if (cfg.algorithm == Algorithm::Hogwild) {
#if HOGWILD_HAS_OPENMP
      std::atomic<int> next{0};
#pragma omp parallel num_threads(cfg.threads)
      {
        while (true) {
          const int pos = next.fetch_add(1, std::memory_order_relaxed);
          if (pos >= static_cast<int>(order.size())) break;
          const auto& s = data.samples[static_cast<std::size_t>(order[static_cast<std::size_t>(pos)])];
          update_hogwild(s, w_atomic.get(), b_atomic, lr, cfg.l2);
        }
      }
      total_updates += static_cast<std::uint64_t>(order.size());
#endif
    } else {
      throw std::invalid_argument("unsupported algorithm for logistic");
    }

    record_eval(epoch);
  }

  const auto t1 = std::chrono::steady_clock::now();
  out.runtime_s = std::chrono::duration<double>(t1 - t0).count();
  out.total_updates = total_updates;
  out.throughput_updates_per_s =
      (out.runtime_s > 0.0) ? static_cast<double>(total_updates) / out.runtime_s : 0.0;
  return out;
}

}  // namespace hogwild
