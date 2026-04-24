#include "workloads.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <unordered_set>

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

std::vector<double> make_zipf_cdf(int n, double skew) {
  std::vector<double> cdf(static_cast<std::size_t>(n), 0.0);
  if (n <= 0) return cdf;
  if (skew <= 1e-12) {
    for (int i = 0; i < n; ++i) {
      cdf[static_cast<std::size_t>(i)] = static_cast<double>(i + 1) / static_cast<double>(n);
    }
    return cdf;
  }
  double sum = 0.0;
  for (int i = 1; i <= n; ++i) {
    sum += 1.0 / std::pow(static_cast<double>(i), skew);
  }
  double running = 0.0;
  for (int i = 1; i <= n; ++i) {
    running += (1.0 / std::pow(static_cast<double>(i), skew)) / sum;
    cdf[static_cast<std::size_t>(i - 1)] = running;
  }
  cdf.back() = 1.0;
  return cdf;
}

int sample_with_cdf(std::mt19937_64& rng, const std::vector<double>& cdf) {
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  const double r = uni(rng);
  const auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
  if (it == cdf.end()) return static_cast<int>(cdf.size()) - 1;
  return static_cast<int>(it - cdf.begin());
}

double estimate_logistic_rho(const std::vector<SparseSample>& samples, std::mt19937_64& rng) {
  if (samples.size() < 2) return 0.0;
  std::uniform_int_distribution<std::size_t> uid(0, samples.size() - 1);
  const int pairs = std::min<int>(20000, static_cast<int>(samples.size() * 4));
  int overlap = 0;
  for (int p = 0; p < pairs; ++p) {
    const auto& a = samples[uid(rng)];
    const auto& b = samples[uid(rng)];
    bool hit = false;
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < a.indices.size() && j < b.indices.size()) {
      if (a.indices[i] == b.indices[j]) {
        hit = true;
        break;
      }
      if (a.indices[i] < b.indices[j]) {
        ++i;
      } else {
        ++j;
      }
    }
    if (hit) ++overlap;
  }
  return static_cast<double>(overlap) / static_cast<double>(pairs);
}

double estimate_mf_rho(const std::vector<MFObservation>& obs, std::mt19937_64& rng) {
  if (obs.size() < 2) return 0.0;
  std::uniform_int_distribution<std::size_t> uid(0, obs.size() - 1);
  const int pairs = std::min<int>(20000, static_cast<int>(obs.size() * 4));
  int overlap = 0;
  for (int p = 0; p < pairs; ++p) {
    const auto& a = obs[uid(rng)];
    const auto& b = obs[uid(rng)];
    if (a.user == b.user || a.item == b.item) {
      ++overlap;
    }
  }
  return static_cast<double>(overlap) / static_cast<double>(pairs);
}

}  // namespace

LogisticDataset make_synthetic_logistic_dataset(const RunConfig& cfg) {
  LogisticDataset out;
  out.dim = cfg.dim;
  out.samples.reserve(static_cast<std::size_t>(cfg.num_samples));

  std::mt19937_64 rng(static_cast<std::uint64_t>(cfg.seed));
  std::normal_distribution<double> normal(0.0, 1.0);
  std::normal_distribution<double> noise(0.0, cfg.label_noise);

  std::vector<double> w_true(static_cast<std::size_t>(cfg.dim), 0.0);
  for (double& v : w_true) {
    v = normal(rng) * 0.5;
  }
  const auto cdf = make_zipf_cdf(cfg.dim, cfg.hotspot_skew);
  std::vector<std::uint64_t> incidence(static_cast<std::size_t>(cfg.dim), 0ULL);

  const int k = std::max(1, std::min(cfg.active_k, cfg.dim));
  for (int n = 0; n < cfg.num_samples; ++n) {
    SparseSample s;
    s.indices.reserve(static_cast<std::size_t>(k));
    s.values.reserve(static_cast<std::size_t>(k));

    std::unordered_set<int> seen;
    seen.reserve(static_cast<std::size_t>(k) * 2U);
    while (static_cast<int>(s.indices.size()) < k) {
      const int idx = sample_with_cdf(rng, cdf);
      if (seen.insert(idx).second) {
        s.indices.push_back(idx);
      }
    }
    std::sort(s.indices.begin(), s.indices.end());
    for (const int idx : s.indices) {
      const double val = normal(rng);
      s.values.push_back(val);
      incidence[static_cast<std::size_t>(idx)] += 1ULL;
    }

    double z = noise(rng);
    for (std::size_t t = 0; t < s.indices.size(); ++t) {
      z += w_true[static_cast<std::size_t>(s.indices[t])] * s.values[t];
    }
    const double p = sigmoid(z);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    s.label = (uni(rng) < p) ? 1 : 0;
    out.samples.push_back(std::move(s));
  }

  const double mean_inc = static_cast<double>(cfg.num_samples * k) / static_cast<double>(cfg.dim);
  const auto max_it = std::max_element(incidence.begin(), incidence.end());
  const double max_inc = (max_it == incidence.end()) ? 0.0 : static_cast<double>(*max_it);

  out.omega = static_cast<double>(k);
  out.delta = (mean_inc > 0.0) ? (max_inc / mean_inc) : 0.0;
  out.rho = estimate_logistic_rho(out.samples, rng);
  return out;
}

MFDataset make_synthetic_mf_dataset(const RunConfig& cfg) {
  MFDataset out;
  out.users = cfg.mf_users;
  out.items = cfg.mf_items;
  out.rank = cfg.mf_rank;
  out.observations.reserve(static_cast<std::size_t>(cfg.mf_observations));

  std::mt19937_64 rng(static_cast<std::uint64_t>(cfg.seed + 9137));
  std::normal_distribution<double> normal(0.0, 1.0);
  std::normal_distribution<double> obs_noise(0.0, cfg.mf_noise);

  std::vector<double> u_true(static_cast<std::size_t>(cfg.mf_users * cfg.mf_rank), 0.0);
  std::vector<double> v_true(static_cast<std::size_t>(cfg.mf_items * cfg.mf_rank), 0.0);
  for (double& x : u_true) x = normal(rng) * 0.2;
  for (double& x : v_true) x = normal(rng) * 0.2;

  const auto user_cdf = make_zipf_cdf(cfg.mf_users, cfg.hotspot_skew);
  const auto item_cdf = make_zipf_cdf(cfg.mf_items, cfg.hotspot_skew);
  std::vector<std::uint64_t> user_inc(static_cast<std::size_t>(cfg.mf_users), 0ULL);
  std::vector<std::uint64_t> item_inc(static_cast<std::size_t>(cfg.mf_items), 0ULL);

  for (int n = 0; n < cfg.mf_observations; ++n) {
    const int u = sample_with_cdf(rng, user_cdf);
    const int i = sample_with_cdf(rng, item_cdf);

    double dot = 0.0;
    for (int f = 0; f < cfg.mf_rank; ++f) {
      dot += u_true[static_cast<std::size_t>(u * cfg.mf_rank + f)] *
             v_true[static_cast<std::size_t>(i * cfg.mf_rank + f)];
    }

    MFObservation o;
    o.user = u;
    o.item = i;
    o.rating = dot + obs_noise(rng);
    out.observations.push_back(o);

    user_inc[static_cast<std::size_t>(u)] += 1ULL;
    item_inc[static_cast<std::size_t>(i)] += 1ULL;
  }

  const double user_mean = static_cast<double>(cfg.mf_observations) / static_cast<double>(cfg.mf_users);
  const double item_mean = static_cast<double>(cfg.mf_observations) / static_cast<double>(cfg.mf_items);

  const double max_user = static_cast<double>(*std::max_element(user_inc.begin(), user_inc.end()));
  const double max_item = static_cast<double>(*std::max_element(item_inc.begin(), item_inc.end()));

  out.omega = static_cast<double>(2 * cfg.mf_rank);
  const double delta_user = (user_mean > 0.0) ? max_user / user_mean : 0.0;
  const double delta_item = (item_mean > 0.0) ? max_item / item_mean : 0.0;
  out.delta = std::max(delta_user, delta_item);
  out.rho = estimate_mf_rho(out.observations, rng);
  return out;
}

}  // namespace hogwild
