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

inline std::size_t uidx(int u, int f, int rank) {
  return static_cast<std::size_t>(u * rank + f);
}

inline std::size_t vidx(int i, int f, int rank, int users) {
  return static_cast<std::size_t>(users * rank + i * rank + f);
}

inline void atomic_add(std::atomic<double>& target, double delta) {
  double old = target.load(std::memory_order_relaxed);
  while (!target.compare_exchange_weak(old, old + delta, std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
  }
}

double mf_loss_dense(const MFDataset& data, const std::vector<double>& params, double reg) {
  const int rank = data.rank;
  const int users = data.users;
  double mse = 0.0;
  for (const auto& o : data.observations) {
    double pred = 0.0;
    for (int f = 0; f < rank; ++f) {
      pred += params[uidx(o.user, f, rank)] * params[vidx(o.item, f, rank, users)];
    }
    const double err = pred - o.rating;
    mse += err * err;
  }
  mse /= static_cast<double>(data.observations.size());

  double l2 = 0.0;
  for (double v : params) l2 += v * v;
  return mse + 0.5 * reg * l2;
}

double mf_loss_atomic(const MFDataset& data, const std::atomic<double>* params, int total_params,
                      double reg) {
  const int rank = data.rank;
  const int users = data.users;
  double mse = 0.0;
  for (const auto& o : data.observations) {
    double pred = 0.0;
    for (int f = 0; f < rank; ++f) {
      pred += params[uidx(o.user, f, rank)].load(std::memory_order_relaxed) *
              params[vidx(o.item, f, rank, users)].load(std::memory_order_relaxed);
    }
    const double err = pred - o.rating;
    mse += err * err;
  }
  mse /= static_cast<double>(data.observations.size());

  double l2 = 0.0;
  for (int i = 0; i < total_params; ++i) {
    const double x = params[static_cast<std::size_t>(i)].load(std::memory_order_relaxed);
    l2 += x * x;
  }
  return mse + 0.5 * reg * l2;
}

void mf_update_dense(const MFObservation& o, std::vector<double>& params, int rank, int users,
                     double lr, double reg) {
  std::vector<double> u_old(static_cast<std::size_t>(rank));
  std::vector<double> v_old(static_cast<std::size_t>(rank));

  double pred = 0.0;
  for (int f = 0; f < rank; ++f) {
    const std::size_t ui = uidx(o.user, f, rank);
    const std::size_t vi = vidx(o.item, f, rank, users);
    u_old[static_cast<std::size_t>(f)] = params[ui];
    v_old[static_cast<std::size_t>(f)] = params[vi];
    pred += u_old[static_cast<std::size_t>(f)] * v_old[static_cast<std::size_t>(f)];
  }

  const double err = pred - o.rating;
  for (int f = 0; f < rank; ++f) {
    const std::size_t ui = uidx(o.user, f, rank);
    const std::size_t vi = vidx(o.item, f, rank, users);
    const double gu = err * v_old[static_cast<std::size_t>(f)] + reg * u_old[static_cast<std::size_t>(f)];
    const double gv = err * u_old[static_cast<std::size_t>(f)] + reg * v_old[static_cast<std::size_t>(f)];
    params[ui] -= lr * gu;
    params[vi] -= lr * gv;
  }
}

void mf_update_hogwild(const MFObservation& o, std::atomic<double>* params, int rank,
                       int users, double lr, double reg) {
  std::vector<double> u_old(static_cast<std::size_t>(rank));
  std::vector<double> v_old(static_cast<std::size_t>(rank));

  double pred = 0.0;
  for (int f = 0; f < rank; ++f) {
    const std::size_t ui = uidx(o.user, f, rank);
    const std::size_t vi = vidx(o.item, f, rank, users);
    u_old[static_cast<std::size_t>(f)] = params[ui].load(std::memory_order_relaxed);
    v_old[static_cast<std::size_t>(f)] = params[vi].load(std::memory_order_relaxed);
    pred += u_old[static_cast<std::size_t>(f)] * v_old[static_cast<std::size_t>(f)];
  }

  const double err = pred - o.rating;
  for (int f = 0; f < rank; ++f) {
    const std::size_t ui = uidx(o.user, f, rank);
    const std::size_t vi = vidx(o.item, f, rank, users);
    const double gu = err * v_old[static_cast<std::size_t>(f)] + reg * u_old[static_cast<std::size_t>(f)];
    const double gv = err * u_old[static_cast<std::size_t>(f)] + reg * v_old[static_cast<std::size_t>(f)];
    atomic_add(params[ui], -lr * gu);
    atomic_add(params[vi], -lr * gv);
  }
}

}  // namespace

RunResult train_matrix_factorization(const MFDataset& data, const RunConfig& cfg) {
  if (data.observations.empty()) {
    throw std::invalid_argument("empty matrix factorization dataset");
  }

  RunResult out;
  out.omega = data.omega;
  out.delta = data.delta;
  out.rho = data.rho;

  const int rank = data.rank;
  const int users = data.users;
  const int items = data.items;
  const std::size_t total_params = static_cast<std::size_t>((users + items) * rank);

  std::mt19937_64 init_rng(static_cast<std::uint64_t>(cfg.init_seed));
  std::normal_distribution<double> init_dist(0.0, 0.05);

  std::vector<double> params(total_params, 0.0);
  for (double& v : params) v = init_dist(init_rng);

  std::unique_ptr<std::atomic<double>[]> params_atomic;
  if (cfg.algorithm == Algorithm::Hogwild) {
    params_atomic = std::make_unique<std::atomic<double>[]>(total_params);
    for (std::size_t i = 0; i < total_params; ++i) params_atomic[i].store(params[i]);
  }

  std::vector<int> order(data.observations.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 shuffle_rng(static_cast<std::uint64_t>(cfg.seed + 4242));

  auto t0 = std::chrono::steady_clock::now();
  std::uint64_t total_updates = 0;

  auto record_eval = [&](int epoch) {
    TracePoint p;
    p.epoch = epoch;
    p.updates = total_updates;
    p.elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (cfg.algorithm == Algorithm::Hogwild) {
      p.loss = mf_loss_atomic(data, params_atomic.get(), static_cast<int>(total_params), cfg.mf_reg);
    } else {
      p.loss = mf_loss_dense(data, params, cfg.mf_reg);
    }
    out.trace.push_back(p);
    out.final_loss = p.loss;
    if (cfg.has_target_loss && !out.reached_target && p.loss <= cfg.target_loss) {
      out.reached_target = true;
      out.time_to_target_s = p.elapsed_s;
    }
  };

  record_eval(0);

  std::mutex coarse_lock;
  const int stripes = 1024;
  std::vector<std::mutex> striped(static_cast<std::size_t>(stripes));

  for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
    if (cfg.shuffle_each_epoch) {
      std::shuffle(order.begin(), order.end(), shuffle_rng);
    }
    const double lr = learning_rate_for_epoch(cfg, epoch - 1);

    if (cfg.algorithm == Algorithm::Serial || cfg.threads <= 1 || !HOGWILD_HAS_OPENMP) {
      for (int pos : order) {
        const auto& o = data.observations[static_cast<std::size_t>(pos)];
        if (cfg.algorithm == Algorithm::Hogwild) {
          mf_update_hogwild(o, params_atomic.get(), rank, users, lr, cfg.mf_reg);
        } else {
          mf_update_dense(o, params, rank, users, lr, cfg.mf_reg);
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
          const auto& o = data.observations[static_cast<std::size_t>(order[static_cast<std::size_t>(pos)])];
          {
            std::lock_guard<std::mutex> lk(coarse_lock);
            mf_update_dense(o, params, rank, users, lr, cfg.mf_reg);
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
        while (true) {
          const int pos = next.fetch_add(1, std::memory_order_relaxed);
          if (pos >= static_cast<int>(order.size())) break;
          const auto& o = data.observations[static_cast<std::size_t>(order[static_cast<std::size_t>(pos)])];

          const int user_lock = o.user % (stripes / 2);
          const int item_lock = (o.item % (stripes / 2)) + stripes / 2;
          const int a = std::min(user_lock, item_lock);
          const int b = std::max(user_lock, item_lock);
          striped[static_cast<std::size_t>(a)].lock();
          striped[static_cast<std::size_t>(b)].lock();
          mf_update_dense(o, params, rank, users, lr, cfg.mf_reg);
          striped[static_cast<std::size_t>(b)].unlock();
          striped[static_cast<std::size_t>(a)].unlock();
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
          const auto& o = data.observations[static_cast<std::size_t>(order[static_cast<std::size_t>(pos)])];
          mf_update_hogwild(o, params_atomic.get(), rank, users, lr, cfg.mf_reg);
        }
      }
      total_updates += static_cast<std::uint64_t>(order.size());
#endif
    } else if (cfg.algorithm == Algorithm::LocalBatchReduce) {
#if HOGWILD_HAS_OPENMP
      std::atomic<int> next{0};
#pragma omp parallel num_threads(cfg.threads)
      {
        std::vector<double> local(total_params, 0.0);
        int local_count = 0;
        while (true) {
          const int pos = next.fetch_add(1, std::memory_order_relaxed);
          if (pos >= static_cast<int>(order.size())) break;
          const auto& o = data.observations[static_cast<std::size_t>(order[static_cast<std::size_t>(pos)])];

          std::vector<double> u_old(static_cast<std::size_t>(rank));
          std::vector<double> v_old(static_cast<std::size_t>(rank));
          double pred = 0.0;
          for (int f = 0; f < rank; ++f) {
            const std::size_t ui = uidx(o.user, f, rank);
            const std::size_t vi = vidx(o.item, f, rank, users);
            u_old[static_cast<std::size_t>(f)] = params[ui];
            v_old[static_cast<std::size_t>(f)] = params[vi];
            pred += u_old[static_cast<std::size_t>(f)] * v_old[static_cast<std::size_t>(f)];
          }
          const double err = pred - o.rating;
          for (int f = 0; f < rank; ++f) {
            const std::size_t ui = uidx(o.user, f, rank);
            const std::size_t vi = vidx(o.item, f, rank, users);
            local[ui] += -lr * (err * v_old[static_cast<std::size_t>(f)] +
                                cfg.mf_reg * u_old[static_cast<std::size_t>(f)]);
            local[vi] += -lr * (err * u_old[static_cast<std::size_t>(f)] +
                                cfg.mf_reg * v_old[static_cast<std::size_t>(f)]);
          }
          ++local_count;
          if (local_count >= cfg.mf_local_batch) {
#pragma omp critical
            {
              for (std::size_t i = 0; i < local.size(); ++i) {
                params[i] += local[i];
                local[i] = 0.0;
              }
            }
            local_count = 0;
          }
        }
#pragma omp critical
        {
          for (std::size_t i = 0; i < local.size(); ++i) {
            params[i] += local[i];
          }
        }
      }
      total_updates += static_cast<std::uint64_t>(order.size());
#endif
    } else {
      throw std::invalid_argument("unsupported algorithm for matrix factorization");
    }

    record_eval(epoch);
  }

  out.runtime_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  out.total_updates = total_updates;
  out.throughput_updates_per_s =
      (out.runtime_s > 0.0) ? static_cast<double>(total_updates) / out.runtime_s : 0.0;
  return out;
}

}  // namespace hogwild
