# vector_store_part1 — a simple vector store, and a benchmark for it

## Layout
```
vector_store_part1
├─ impls/            one vector_store.h per folder — the implementations under test
│  ├─ initial_design_no_opt/       std::vector<double> rows; score every row, then sort
│  ├─ optimized_no_manual_simd/    flat float32 array + bounded size-k heap
│  └─ optimized_with_manual_simd/  the same, plus NEON kernels on ARM
└─ benchmark/        
   ├─ bench.py          library for: build / run / plot (build(), run(), benchmark(), speedup_chart())
   ├─ run_bench.ipynb   driver notebook: CONFIG cell + two plots
   ├─ bench_cpp.cpp     times our store; #include "vector_store.h" (impl chosen by -I); metric via argv
   ├─ bench_sqlite.cpp  times sqlite-vec's vec0 on the same data (comparison engine)
   ├─ gen_data.cpp      deterministic test-data generator (fixed seed)
   ├─ vendor/           vendored sqlite-vec sources (sqlite-vec.c/.h) — the vec0 extension
   ├─ data/             generated vectors_<N>_<dim>.txt + queries_<dim>.txt   (regenerated on demand)
   └─ build/            compiled per-engine binaries + gen_data + sqlite-vec.o  (regenerated on demand)
```
All three headers expose the same API — `add`, `remove`, `search`, `save`, `load`, `size`,
`get_dimension`.

## Benchmark
An **engine** is `implementation folder × compile flags × label`: the implementation is
selected at compile time via `-I<impls_dir>/<keyword>`, so one `bench_cpp.cpp` benchmarks any
`impls/<keyword>/vector_store.h` against any flags — no Makefile.

The notebook sets `bench.IMPLS_DIR = Path("../impls")`. That path is resolved relative to
`benchmark/` (where `bench.py` lives), not relative to this file.

## Dependencies
- `g++` / `cc` with C++17.
- For MacOS: Homebrew **sqlite** at `/opt/homebrew/opt/sqlite` (headers + `libsqlite3`), only
  for the `sqlite_vec` engine. If absent, that engine is skipped with a note; the
  C++ engines still run.
- For other platforms: Update the installed sqlite path in `SQLITE_INC` / `SQLITE_LIB` constants
  in `benchmark/bench.py`.
- Python with `matplotlib` (only needed for plotting, imported lazily).

## Usage
Open `benchmark/run_bench.ipynb`, edit the CONFIG cell (`ENGINES`, `SELECTED`, `SPEEDUP`,
`NS`, `DIMS`, `K`, `METRIC`), and run all cells. `bench.py` builds each selected
engine, generates any missing data, runs the sweep, and plots.
