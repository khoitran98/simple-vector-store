"""Quick vector-store benchmark.

except for sqlite-vec, engine = implementation folder x compile flags x label. The implementation is
chosen at compile time via -I<IMPLS_DIR>/<impl>, so one bench_cpp.cpp benchmarks
any impls/<keyword>/vector_store.h against any flags. Binaries run with cwd=data/
so their derived queries_<dim>.txt is found there.
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA, BUILD = HERE / "data", HERE / "build"
VENDOR = HERE / "vendor"                        # local vendored sqlite-vec sources
SQLITE_INC, SQLITE_LIB = "/opt/homebrew/opt/sqlite/include", "/opt/homebrew/opt/sqlite/lib"

IMPLS_DIR = Path("../impls")   # where <impl>/vector_store.h live


@dataclass
class Engine:
    impl: str = ""              # folder under IMPLS_DIR (ignored for kind="sqlite")
    flags: str = ""             # raw g++ flags, e.g. "-O2 -ffast-math"
    color: str = "tab:blue"
    kind: str = "cpp"           # "cpp" or "sqlite"


def _sh(cmd, cwd=None):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(" ".join(map(str, cmd)) + "\n" + r.stderr + r.stdout)
    return r


def ensure_data(n, dim):
    """Generate data/vectors_<n>_<dim>.txt + queries_<dim>.txt if missing (deterministic)."""
    DATA.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    if (DATA / f"vectors_{n}_{dim}.txt").exists() and (DATA / f"queries_{dim}.txt").exists():
        return
    gen = BUILD / "gen_data"
    if not gen.exists():
        _sh(["g++", "-std=c++17", "-O2", HERE / "gen_data.cpp", "-o", gen])
    _sh([gen, n, dim], cwd=DATA)


def build(label, e):
    """Compile one engine to build/<label>. Returns the path, or None if skipped."""
    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / label
    if e.kind == "sqlite":
        if not (Path(SQLITE_INC) / "sqlite3.h").exists():
            print(f"[skip] {label}: Homebrew sqlite not found at {SQLITE_INC}")
            return None
        obj = BUILD / "sqlite-vec.o"
        if not obj.exists():
            _sh(["cc", "-O3", "-DSQLITE_CORE", "-DSQLITE_VEC_STATIC", "-DSQLITE_VEC_OMIT_FS",
                 f"-I{SQLITE_INC}", "-c", VENDOR / "sqlite-vec.c", "-o", obj])
        # -I{HERE} lets bench_sqlite.cpp's #include "vendor/sqlite-vec.h" resolve locally.
        _sh(["g++", "-std=c++17", *e.flags.split(), f"-I{HERE}", f"-I{SQLITE_INC}",
             HERE / "bench_sqlite.cpp", obj, f"-L{SQLITE_LIB}", "-lsqlite3", "-o", out])
        return out
    inc = HERE / IMPLS_DIR / e.impl   # absolute IMPLS_DIR overrides HERE; relative resolves to it
    if not (inc / "vector_store.h").exists():
        print(f"[skip] {label}: no vector_store.h in {inc}")
        return None
    _sh(["g++", "-std=c++17", *e.flags.split(), f'-DENGINE_LABEL="{label}"',
         f"-I{inc}", HERE / "bench_cpp.cpp", "-o", out])
    return out


_RE = re.compile(r"(\w+)=([-\d.eE+]+|\w+)")


METRICS = {"euclidean", "cosine"}  


def _norm_metric(metric):
    m = metric.lower()
    if m == "l2":
        m = "euclidean"
    if m not in METRICS:
        raise ValueError(f"unsupported metric {metric!r}; use one of {sorted(METRICS)}")
    return m


def run(out, n, dim, k, metric="euclidean"):
    """Run a built engine on (n, dim, k, metric) and return per-query latency in ms."""
    ensure_data(n, dim)
    r = _sh([out, f"vectors_{n}_{dim}.txt", dim, k, _norm_metric(metric)], cwd=DATA)
    line = next(l for l in r.stdout.splitlines() if l.startswith("engine="))
    fields = dict(_RE.findall(line))
    print(f"[sum]  {Path(out).name} n={n} dim={dim}: checksum={fields.get('checksum', '?')}")
    return float(fields["per_query_ms"])


def benchmark(engines, selected, NS, DIMS, K, metric="euclidean"):
    """Build the selected engines and plot per-query latency vs N (one figure per dim)."""
    import matplotlib.pyplot as plt
    metric = _norm_metric(metric)
    bins = {}
    for label in selected:
        out = build(label, engines[label])
        if out:
            bins[label] = out
            print(f"[ok]   {label}")
    for dim in DIMS:
        plt.figure(figsize=(7, 5))
        for label, out in bins.items():
            ys = [run(out, n, dim, K, metric) for n in NS]
            plt.plot(NS, ys, marker="o", color=engines[label].color, label=label)
            print(f"[run]  {label} dim={dim}: " + ", ".join(f"{n}:{y:.4g}ms" for n, y in zip(NS, ys)))
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("N (stored vectors)"); plt.ylabel("per-query latency (ms)")
        plt.title(f"Per-query latency vs N  (dim={dim}, k={K}, metric={metric})")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend(); plt.tight_layout()
    plt.show()


def speedup_chart(engines, three, NS, DIMS, K, metric="euclidean"):
    """Normalized speedup bar chart for 3 engines: [baseline, e2, e3].
    """
    import matplotlib.pyplot as plt
    metric = _norm_metric(metric)
    if len(three) != 3:
        raise ValueError("speedup_chart expects 3 engine labels: [baseline, e2, e3]")
    base, others = three[0], three[1:]

    bins = {}
    for label in three:
        out = build(label, engines[label])
        if out:
            bins[label] = out
            print(f"[ok]   {label}")
    if base not in bins:
        print(f"[error] baseline {base!r} failed to build; cannot compute speedup")
        return

    x = list(range(len(NS)))
    comps = [l for l in others if l in bins]
    width = 0.8 / max(len(comps), 1)
    for dim in DIMS:
        lat = {label: [run(bins[label], n, dim, K, metric) for n in NS]
               for label in bins}
        plt.figure(figsize=(7, 5))
        for j, label in enumerate(comps):
            speedups = [lat[base][i] / lat[label][i] for i in range(len(NS))]
            offs = (j - (len(comps) - 1) / 2) * width
            bars = plt.bar([xi + offs for xi in x], speedups, width,
                           color=engines[label].color, label=f"{label} vs {base}")
            try:
                plt.bar_label(bars, fmt="%.2fx", padding=2, fontsize=8)
            except AttributeError:
                pass  # matplotlib < 3.4 has no bar_label
            print(f"[spd]  dim={dim} {label} vs {base}: "
                  + ", ".join(f"{n}:{s:.2f}x" for n, s in zip(NS, speedups)))
        plt.axhline(1.0, color="gray", ls="--", lw=1, label=f"{base} (baseline)")
        plt.xticks(x, [str(n) for n in NS])
        plt.xlabel("N (stored vectors)")
        plt.ylabel(f"speedup vs {base}  (x)")
        plt.title(f"Normalized speedup  (dim={dim}, k={K}, metric={metric})")
        plt.grid(True, axis="y", ls="--", alpha=0.4)
        plt.legend(); plt.tight_layout()
    plt.show()
