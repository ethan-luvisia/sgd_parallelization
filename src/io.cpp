#include "io.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace hogwild {
namespace {

bool file_exists(const std::string& p) {
  if (p.empty()) return false;
  return std::filesystem::exists(std::filesystem::path(p));
}

void ensure_parent_dir(const std::string& p) {
  if (p.empty()) return;
  const auto parent = std::filesystem::path(p).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

}  // namespace

void append_summary_csv(const RunConfig& cfg, const RunResult& result) {
  if (cfg.out_summary_csv.empty()) return;

  ensure_parent_dir(cfg.out_summary_csv);
  const bool exists = file_exists(cfg.out_summary_csv);
  std::ofstream out(cfg.out_summary_csv, std::ios::app);

  if (!exists) {
    out << "run_id,trial,seed,init_seed,workload,algorithm,schedule,threads,epochs,"
        << "num_samples,dim,active_k,hotspot_skew,l2,mf_users,mf_items,mf_rank,mf_observations,mf_reg,"
        << "lr,decay,backoff_gamma,target_loss,reached_target,time_to_target_s,runtime_s,total_updates,"
        << "throughput_updates_per_s,final_loss,omega,delta,rho\n";
  }

  out << cfg.run_id << ',' << cfg.trial << ',' << cfg.seed << ',' << cfg.init_seed << ','
      << to_string(cfg.workload) << ',' << to_string(cfg.algorithm) << ',' << to_string(cfg.schedule)
      << ',' << cfg.threads << ',' << cfg.epochs << ',' << cfg.num_samples << ',' << cfg.dim << ','
      << cfg.active_k << ',' << cfg.hotspot_skew << ',' << cfg.l2 << ',' << cfg.mf_users << ','
      << cfg.mf_items << ',' << cfg.mf_rank << ',' << cfg.mf_observations << ',' << cfg.mf_reg << ','
      << cfg.lr << ',' << cfg.decay << ',' << cfg.backoff_gamma << ','
      << (cfg.has_target_loss ? cfg.target_loss : -1.0) << ',' << (result.reached_target ? 1 : 0) << ','
      << result.time_to_target_s << ',' << result.runtime_s << ',' << result.total_updates << ','
      << result.throughput_updates_per_s << ',' << result.final_loss << ',' << result.omega << ','
      << result.delta << ',' << result.rho << '\n';
}

void append_trace_csv(const RunConfig& cfg, const RunResult& result) {
  if (cfg.out_trace_csv.empty()) return;

  ensure_parent_dir(cfg.out_trace_csv);
  const bool exists = file_exists(cfg.out_trace_csv);
  std::ofstream out(cfg.out_trace_csv, std::ios::app);

  if (!exists) {
    out << "run_id,trial,seed,workload,algorithm,schedule,threads,epoch,elapsed_s,updates,loss\n";
  }

  for (const auto& p : result.trace) {
    out << cfg.run_id << ',' << cfg.trial << ',' << cfg.seed << ',' << to_string(cfg.workload) << ','
        << to_string(cfg.algorithm) << ',' << to_string(cfg.schedule) << ',' << cfg.threads << ','
        << p.epoch << ',' << p.elapsed_s << ',' << p.updates << ',' << p.loss << '\n';
  }
}

}  // namespace hogwild
