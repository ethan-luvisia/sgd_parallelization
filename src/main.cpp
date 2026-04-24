#include "io.hpp"
#include "system_info.hpp"
#include "train.hpp"
#include "types.hpp"
#include "workloads.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_help() {
  std::cout << "hogwild_bench options:\n"
            << "  --workload logistic|mf\n"
            << "  --algorithm serial|coarse_lock|striped_lock|hogwild|local_batch_reduce\n"
            << "  --threads INT --epochs INT --seed INT --init-seed INT --trial INT\n"
            << "  --schedule constant|epoch_decay|backoff --lr FLOAT --decay FLOAT --backoff-gamma FLOAT\n"
            << "  --target-loss FLOAT\n"
            << "\n"
            << "Logistic params:\n"
            << "  --num-samples INT --dim INT --active-k INT --label-noise FLOAT --hotspot-skew FLOAT --l2 FLOAT\n"
            << "\n"
            << "Matrix factorization params:\n"
            << "  --mf-users INT --mf-items INT --mf-rank INT --mf-observations INT --mf-noise FLOAT --mf-reg FLOAT\n"
            << "\n"
            << "Outputs:\n"
            << "  --run-id STRING --out-summary-csv PATH --out-trace-csv PATH --print-system-info\n";
}

bool has_arg(int i, int argc) { return i + 1 < argc; }

}  // namespace

int main(int argc, char** argv) {
  using namespace hogwild;

  RunConfig cfg;
  bool print_sys = false;

  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto next = [&]() -> std::string {
      if (!has_arg(i, argc)) throw std::invalid_argument("Missing value for: " + key);
      return argv[++i];
    };

    if (key == "--help" || key == "-h") {
      print_help();
      return 0;
    } else if (key == "--workload") {
      cfg.workload = parse_workload(next());
    } else if (key == "--algorithm") {
      cfg.algorithm = parse_algorithm(next());
    } else if (key == "--threads") {
      cfg.threads = std::stoi(next());
    } else if (key == "--epochs") {
      cfg.epochs = std::stoi(next());
    } else if (key == "--seed") {
      cfg.seed = std::stoi(next());
    } else if (key == "--init-seed") {
      cfg.init_seed = std::stoi(next());
    } else if (key == "--trial") {
      cfg.trial = std::stoi(next());
    } else if (key == "--schedule") {
      cfg.schedule = parse_schedule(next());
    } else if (key == "--lr") {
      cfg.lr = std::stod(next());
    } else if (key == "--decay") {
      cfg.decay = std::stod(next());
    } else if (key == "--backoff-gamma") {
      cfg.backoff_gamma = std::stod(next());
    } else if (key == "--num-samples") {
      cfg.num_samples = std::stoi(next());
    } else if (key == "--dim") {
      cfg.dim = std::stoi(next());
    } else if (key == "--active-k") {
      cfg.active_k = std::stoi(next());
    } else if (key == "--label-noise") {
      cfg.label_noise = std::stod(next());
    } else if (key == "--hotspot-skew") {
      cfg.hotspot_skew = std::stod(next());
    } else if (key == "--l2") {
      cfg.l2 = std::stod(next());
    } else if (key == "--mf-users") {
      cfg.mf_users = std::stoi(next());
    } else if (key == "--mf-items") {
      cfg.mf_items = std::stoi(next());
    } else if (key == "--mf-rank") {
      cfg.mf_rank = std::stoi(next());
    } else if (key == "--mf-observations") {
      cfg.mf_observations = std::stoi(next());
    } else if (key == "--mf-noise") {
      cfg.mf_noise = std::stod(next());
    } else if (key == "--mf-reg") {
      cfg.mf_reg = std::stod(next());
    } else if (key == "--run-id") {
      cfg.run_id = next();
    } else if (key == "--out-summary-csv") {
      cfg.out_summary_csv = next();
    } else if (key == "--out-trace-csv") {
      cfg.out_trace_csv = next();
    } else if (key == "--target-loss") {
      cfg.has_target_loss = true;
      cfg.target_loss = std::stod(next());
    } else if (key == "--print-system-info") {
      print_sys = true;
    } else {
      throw std::invalid_argument("Unknown arg: " + key);
    }
  }

  if (cfg.run_id.empty()) {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    cfg.run_id = "run_" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
  }

  try {
    if (print_sys) {
      std::cout << "system_info=" << collect_system_info() << "\n";
    }

    RunResult result;
    if (cfg.workload == WorkloadType::Logistic) {
      const auto data = make_synthetic_logistic_dataset(cfg);
      result = train_logistic(data, cfg);
    } else {
      const auto data = make_synthetic_mf_dataset(cfg);
      result = train_matrix_factorization(data, cfg);
    }

    append_summary_csv(cfg, result);
    append_trace_csv(cfg, result);

    std::cout << std::fixed << std::setprecision(6)
              << "run_id=" << cfg.run_id << " workload=" << to_string(cfg.workload)
              << " algorithm=" << to_string(cfg.algorithm) << " threads=" << cfg.threads
              << " runtime_s=" << result.runtime_s
              << " throughput=" << result.throughput_updates_per_s
              << " final_loss=" << result.final_loss
              << " reached_target=" << (result.reached_target ? 1 : 0)
              << " ttt_s=" << result.time_to_target_s
              << " omega=" << result.omega << " delta=" << result.delta << " rho=" << result.rho
              << "\n";

  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
