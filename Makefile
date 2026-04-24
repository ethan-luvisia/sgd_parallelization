.PHONY: build test quick full plots report clean

build:
	cmake -S . -B build_gcc -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15
	cmake --build build_gcc -j

test: build
	ctest --test-dir build_gcc --output-on-failure

quick: build
	python3 scripts/run_benchmarks.py --mode quick --exe build_gcc/hogwild_bench --results-dir results

full: build
	python3 scripts/run_benchmarks.py --mode full --exe build_gcc/hogwild_bench --results-dir results_full

plots:
	HOME=/tmp MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl_cache XDG_CACHE_HOME=/tmp python3 scripts/plot_results.py --summary results_full/summary.csv --trace results_full/trace.csv --out-dir plots --report-fig-dir report/figures

report:
	python3 scripts/generate_report_summary.py --summary results_full/summary.csv --out report/results_summary.md

clean:
	rm -rf build build_gcc
