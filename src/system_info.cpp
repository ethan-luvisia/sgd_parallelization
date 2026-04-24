#include "system_info.hpp"

#include <array>
#include <cstdio>
#include <sstream>
#include <string>
#include <thread>

namespace hogwild {
namespace {

std::string run_cmd(const char* cmd) {
  std::array<char, 256> buffer{};
  std::string out;
  FILE* pipe = popen(cmd, "r");
  if (!pipe) return "unknown";
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    out += buffer.data();
  }
  pclose(pipe);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
  return out.empty() ? "unknown" : out;
}

}  // namespace

std::string collect_system_info() {
  std::ostringstream oss;
  oss << "hardware_threads=" << std::thread::hardware_concurrency() << ";";
  oss << "compiler=" << run_cmd("c++ --version | head -n 1") << ";";
  oss << "os=" << run_cmd("uname -a") << ";";
  oss << "cpu=" << run_cmd("sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu | grep 'Model name' | head -n 1");
  return oss.str();
}

}  // namespace hogwild
