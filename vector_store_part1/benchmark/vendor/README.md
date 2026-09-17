# vendor/ — third-party code

This folder contains **sqlite-vec**, an unmodified vendored copy of a third-party
SQLite extension used only by the `sqlite_vec` comparison engine in the benchmark.

- **Project:** sqlite-vec by Alex Garcia — https://github.com/asg017/sqlite-vec
- **Version:** v0.1.9 (single-file "amalgamation" build)
- **Files:** `sqlite-vec.c`, `sqlite-vec.h` (the vec0 vector-search virtual table)
- **Source artifact:** `sqlite-vec-0.1.9-amalgamation.tar.gz`
  (SHA-256 recorded in `DOWNLOAD_SHA256.txt`)
- **Upstream commit:** `e9f598abfa0c06b328d8fe5da9c3760cce74be10` (see `SQLITE_VEC_SOURCE` in the header)

## License

sqlite-vec is dual-licensed **MIT OR Apache-2.0**. It is redistributed here under
the **MIT** license — see `LICENSE-MIT` (© 2024 Alex Garcia). The Apache-2.0
alternative is available upstream as `LICENSE-APACHE`.

These files are **not** part of this project's own source; they are included so
the benchmark builds reproducibly offline without re-fetching the release.
