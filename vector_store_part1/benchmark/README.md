# benchmark — vector store benchmark harness

Self-contained benchmark for the vector store. An **engine** is
`implementation folder × compile flags × label`: the implementation is selected
at compile time via `-I<impls_dir>/<keyword>`, so one `bench_cpp.cpp` benchmarks
any `impls/<keyword>/vector_store.h` against any flags — no Makefile.

## Layout
```
benchmark
├─ bench.py          harness library: build / run / plot (build(), run(), benchmark(), speedup_chart())
├─ run_bench.ipynb   driver notebook: CONFIG cell + two plots
├─ bench_cpp.cpp     times our store; #include "vector_store.h" (impl chosen by -I); metric via argv
├─ bench_sqlite.cpp  times sqlite-vec's vec0 on the same data (comparison engine)
├─ gen_data.cpp      deterministic test-data generator (fixed seed)
├─ vendor/           vendored sqlite-vec sources (sqlite-vec.c/.h) — the vec0 extension
├─ data/             generated vectors_<N>_<dim>.txt + queries_<dim>.txt   (regenerated on demand)
└─ build/            compiled per-engine binaries + gen_data + sqlite-vec.o  (regenerated on demand)
```
Implementations live *outside* this folder, one `vector_store.h` per folder:
`../impls/<keyword>/vector_store.h` (path set by `bench.IMPLS_DIR` in the notebook).

## Dependencies
- `g++` / `cc` with C++17.
- For MacOS: Homebrew **sqlite** at `/opt/homebrew/opt/sqlite` (headers + `libsqlite3`), only
  for the `sqlite_vec` engine. If absent, that engine is skipped with a note; the
  C++ engines still run.
- For other platforms: Update the installed sqlite path in `SQLITE_INC` / `SQLITE_LIB` constants 
  in `bench.py`.
- Python with `matplotlib` (only needed for plotting, imported lazily).

## Usage
Open `run_bench.ipynb`, edit the CONFIG cell (`ENGINES`, `SELECTED`, `SPEEDUP`,
`NS`, `DIMS`, `K`, `METRIC`), and run all cells. `bench.py` builds each selected
engine, generates any missing data, runs the sweep, and plots. Re-run the notebook 
(or call `bench.benchmark(...)`); binaries, `gen_data`,`sqlite-vec.o`, and the 
datasets are all regenerated from the sources here.
