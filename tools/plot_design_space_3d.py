#!/usr/bin/env python3
"""ComBit thesis design-space figure (CUBIT Fig. 1 style, 3-D framing).

Renders ONE combined figure with 4 panels rendering the 3-D conceptual
space (compression × op-perf × density-robustness):

  (A) 3-D trajectory plot in (op_ms, comp_factor, density) — each
      backend is a curve through density space.  Compact, level curve
      => robust.
  (B) CUBIT-style 2-D footprint in (op latency, compression factor).
      Each backend rendered as a median diamond + horizontal range bar
      (op-time spread across density) + vertical range bar (comp spread).
      Short bars in op-time => density-robust performance.
  (C) Compression factor vs density.  Flat curve = robust.
  (D) Pure-op OR latency vs density.  Flat curve = robust.

Data sources (real benchmarks, no synthetic numbers):
  /home/lichenhang/lee/thesis/check/lee/combit/results_full.csv
      ComBit micro-benchmark: 5 backends × 15 cardinalities (densities
      from 50% down to 0.01%) × 3 iterations + dedicated PureOps OR_op
      measurement.  We use OR_op (pure op timings) for fairness across
      backends (build cost excluded).
  /home/lichenhang/lee/thesis/check/lee/duckdb-dev/bm_results.csv
      Real TPC-H SF10 operating points (Q3 PhaseB_okey / Q6 OR_quantity
      / Q14 OR) overlaid as star markers on Panel B.

Output: bm_design_space_3d.{png,pdf} written to duckdb-dev repo root.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-d proj)

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
REPO = Path("/home/lichenhang/lee/thesis/check/lee/duckdb-dev")
COMBIT = Path("/home/lichenhang/lee/thesis/check/lee/combit")
MICRO_CSV = COMBIT / "results_full.csv"
TPCH_CSV  = REPO / "bm_results.csv"
OUT_PNG = REPO / "bm_design_space_3d.png"
OUT_PDF = REPO / "bm_design_space_3d.pdf"

# ----------------------------------------------------------------------
# Backend canonicalisation
# ----------------------------------------------------------------------
MICRO_NAME = {
    "ComBIT (New)":  "ComBit",
    "WAH (FastBit)": "WAH",
    "CRoaring":      "CRoaring",
    "EWAH":          "EWAH",
    "Concise":       "Concise",
}
TPCH_TAG = {
    "CB":     "ComBit",
    "CB+BPE": "ComBit",
    "WAH":    "WAH",
    "CR":     "CRoaring",
    "CR+BPE": "CRoaring",
    "CRR":    "CRoaring",
    "EW":     "EWAH",
    "CON":    "Concise",
}
BACKENDS = ["ComBit", "CRoaring", "WAH", "EWAH", "Concise"]
COLOR = {
    "ComBit":   "#d62728",   # red  — our system
    "CRoaring": "#1f77b4",   # blue
    "WAH":      "#2ca02c",   # green
    "EWAH":     "#9467bd",   # purple
    "Concise":  "#ff7f0e",   # orange
}
MARKER = {
    "ComBit": "o", "CRoaring": "s", "WAH": "^", "EWAH": "D", "Concise": "v",
}

# ----------------------------------------------------------------------
# Load micro-benchmark from results_full.csv.
#
# We want per (backend, cardinality):
#   - op_ms     = median of OR_op rows (pure op, build excluded)
#   - ratio     = compression ratio from any OR row (constant per group)
# ----------------------------------------------------------------------
full = pd.read_csv(MICRO_CSV)
full["Backend"] = full["backend"].map(MICRO_NAME)
full = full.dropna(subset=["Backend"]).copy()

op = (full[full["operation"] == "OR_op"]
      .groupby(["Backend", "cardinality"], as_index=False)["time_ms"]
      .median()
      .rename(columns={"time_ms": "op_ms"}))

# Compression ratio is constant for a given (backend, cardinality);
# grab from any non-NaN compression_ratio row.
rat = (full[full["compression_ratio"].notna() & (full["compression_ratio"] > 0)]
       .groupby(["Backend", "cardinality"], as_index=False)["compression_ratio"]
       .median()
       .rename(columns={"compression_ratio": "ratio"}))

m = op.merge(rat, on=["Backend", "cardinality"], how="inner")
m["density"]      = 1.0 / m["cardinality"]
m["comp_factor"]  = 1.0 / m["ratio"].clip(lower=1e-9)

# ----------------------------------------------------------------------
# Load duckdb-dev TPC-H operating points (real workload anchors)
# ----------------------------------------------------------------------
t = pd.read_csv(TPCH_CSV)
t["Backend"] = t["Backend"].map(TPCH_TAG)
t = t.dropna(subset=["Backend"]).copy()
ANCHOR_PHASES = [
    ("Q3",  "PhaseB_okey"),   # 220K-key sparse OR over l_orderkey
    ("Q6",  "OR_quantity"),   # 25-key dense OR over l_quantity
    ("Q14", "OR"),            # range OR over shipdate
]
anchors = []
for q, phase in ANCHOR_PHASES:
    sub = t[(t["Q"] == q) & (t["Phase"] == phase)]
    if sub.empty:
        continue
    # Pick the best variant per backend family.
    for bk in BACKENDS:
        rows = sub[sub["Backend"] == bk]
        if rows.empty:
            continue
        rows = rows[rows["Median_ms"] > 0]
        if rows.empty:
            continue
        best = rows.loc[rows["Median_ms"].idxmin()]
        anchors.append({
            "tag":    f"{q}",
            "Backend": bk,
            "op_ms":   float(best["Median_ms"]),
        })

# ----------------------------------------------------------------------
# Robustness stats per backend (op-latency variability across density).
# ----------------------------------------------------------------------
stats = (m.groupby("Backend")
         .agg(op_min=("op_ms", "min"), op_max=("op_ms", "max"),
              op_med=("op_ms", "median"),
              cf_min=("comp_factor", "min"), cf_max=("comp_factor", "max"),
              cf_med=("comp_factor", "median"))
         .reindex(BACKENDS))
stats["op_span"] = stats["op_max"] / stats["op_min"]
print("=== Robustness summary (smaller op_span = more robust) ===")
print(stats[["op_min", "op_max", "op_span", "cf_min", "cf_max"]].round(3))

# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})
fig = plt.figure(figsize=(15, 10), constrained_layout=False)
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1, 1],
                      hspace=0.32, wspace=0.22,
                      left=0.05, right=0.97, top=0.91, bottom=0.07)
ax3d = fig.add_subplot(gs[0, 0], projection="3d")
axB  = fig.add_subplot(gs[0, 1])
axC  = fig.add_subplot(gs[1, 0])
axD  = fig.add_subplot(gs[1, 1])

# ======================================================================
# (A) 3-D trajectory: (log10 op_ms, log10 comp_factor, log10 density)
# ======================================================================
for bk in BACKENDS:
    sub = m[m["Backend"] == bk].sort_values("density")
    if sub.empty:
        continue
    xs = np.log10(sub["op_ms"].values)
    ys = np.log10(sub["comp_factor"].values)
    zs = np.log10(sub["density"].values)
    ax3d.plot(xs, ys, zs, color=COLOR[bk], marker=MARKER[bk],
              markersize=5, linewidth=2.0, alpha=0.92, label=bk)
# Highlight "better corner" (low op, high comp)
ax3d.set_xlabel("log₁₀ OR latency (ms)\n← faster", labelpad=2)
ax3d.set_ylabel("log₁₀ compression factor (×)\nbetter →", labelpad=2)
ax3d.set_zlabel("log₁₀ density", labelpad=2)
ax3d.set_title("(A) 3-D design space across density sweep\n"
               "(compact curve = density-robust)")
ax3d.tick_params(labelsize=7)
ax3d.view_init(elev=20, azim=-60)
ax3d.legend(loc="upper left", framealpha=0.92,
            bbox_to_anchor=(-0.02, 0.98))

# ======================================================================
# (B) CUBIT-style footprint: median diamond + range bars
# ======================================================================
for bk in BACKENDS:
    s = stats.loc[bk]
    cx_lo, cx_hi = np.log10(s["op_min"]), np.log10(s["op_max"])
    cy_lo, cy_hi = np.log10(s["cf_min"]), np.log10(s["cf_max"])
    cx_med, cy_med = np.log10(s["op_med"]), np.log10(s["cf_med"])
    c = COLOR[bk]
    # Horizontal bar (op-latency spread = inverse-robustness in X dimension).
    axB.plot([cx_lo, cx_hi], [cy_med, cy_med],
             color=c, linewidth=2.2, alpha=0.55)
    axB.plot([cx_lo, cx_lo], [cy_med - 0.03, cy_med + 0.03],
             color=c, linewidth=2.2, alpha=0.55)
    axB.plot([cx_hi, cx_hi], [cy_med - 0.03, cy_med + 0.03],
             color=c, linewidth=2.2, alpha=0.55)
    # Vertical bar (compression spread across density).
    axB.plot([cx_med, cx_med], [cy_lo, cy_hi],
             color=c, linewidth=2.2, alpha=0.35, linestyle=":")
    # Median marker.
    axB.scatter(cx_med, cy_med, marker="D", s=180, color=c,
                edgecolor="black", linewidth=1.0, zorder=4)
    # Backend label — placed offset from median to avoid overlap.
    axB.annotate(f"{bk}\n(op {s['op_span']:.1f}× spread)",
                 xy=(cx_med, cy_med),
                 xytext=(0, -28 if bk in ("ComBit", "EWAH") else 22),
                 textcoords="offset points",
                 fontsize=9, color=c, fontweight="bold",
                 ha="center", va="center",
                 bbox=dict(facecolor="white", edgecolor=c, alpha=0.93,
                           boxstyle="round,pad=0.25"))

# Overlay TPC-H operating points as stars.
for a in anchors:
    bk = a["Backend"]
    if bk not in COLOR:
        continue
    # Place at backend's median compression factor (TPC-H CSV has no
    # compression column; the X-coordinate is what matters for "where
    # does this real workload sit on the op-latency axis").
    cy = np.log10(stats.loc[bk, "cf_med"])
    axB.scatter(np.log10(a["op_ms"]), cy,
                marker="*", s=140, color=COLOR[bk],
                edgecolor="black", linewidth=0.7, zorder=5)
axB.set_xlabel("log₁₀ OR latency (ms)  ←  faster")
axB.set_ylabel("log₁₀ compression factor  →  better")
axB.set_title("(B) Footprint in (op-perf × compression) plane\n"
              "diamond=median, ─ horizontal=op-spread, ┊ vertical=comp-spread, ★=TPC-H SF10")
axB.grid(True, linestyle=":", alpha=0.5)
# Better direction arrow.
axB.annotate("", xy=(0.06, 0.96), xytext=(0.30, 0.65),
             xycoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color="black", lw=1.6))
axB.text(0.07, 0.90, "better\n(fast + small)", transform=axB.transAxes,
         fontsize=9, color="black",
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

# ======================================================================
# (C) Compression factor vs density
# ======================================================================
for bk in BACKENDS:
    sub = m[m["Backend"] == bk].sort_values("density")
    axC.plot(sub["density"], sub["comp_factor"],
             color=COLOR[bk], marker=MARKER[bk], markersize=5,
             linewidth=1.9, label=bk)
axC.set_xscale("log"); axC.set_yscale("log")
axC.set_xlabel("Bit density (set-fraction per bitmap)")
axC.set_ylabel("Compression factor ×  (higher = better)")
axC.set_title("(C) Compression vs density  (no method dominates everywhere)")
axC.grid(True, which="both", linestyle=":", alpha=0.5)
axC.legend(ncol=2, loc="lower left")

# ======================================================================
# (D) Pure-op OR latency vs density
# ======================================================================
for bk in BACKENDS:
    sub = m[m["Backend"] == bk].sort_values("density")
    axD.plot(sub["density"], sub["op_ms"],
             color=COLOR[bk], marker=MARKER[bk], markersize=5,
             linewidth=1.9, label=bk)
axD.set_xscale("log"); axD.set_yscale("log")
axD.set_xlabel("Bit density (set-fraction per bitmap)")
axD.set_ylabel("Pure OR latency (ms)  (lower = better)")
axD.set_title("(D) OR latency vs density  (ComBit is flattest)")
axD.grid(True, which="both", linestyle=":", alpha=0.5)
axD.legend(ncol=2, loc="lower left")
# Annotation calling out ComBit's flatness.
axD.annotate(
    f"ComBit op-latency stays in {stats.loc['ComBit', 'op_min']:.2f}–"
    f"{stats.loc['ComBit', 'op_max']:.2f} ms across\n5000× density range "
    f"({stats.loc['ComBit', 'op_span']:.1f}× spread); "
    f"WAH spans {stats.loc['WAH', 'op_span']:.0f}×, "
    f"Concise {stats.loc['Concise', 'op_span']:.0f}×.",
    xy=(0.40, 0.96), xycoords="axes fraction",
    xytext=(0.40, 0.96), textcoords="axes fraction",
    fontsize=8, color="black", ha="left", va="top",
    bbox=dict(facecolor="#fff7cc", edgecolor="#d4a017", alpha=0.92,
              boxstyle="round,pad=0.3"),
)

# ----------------------------------------------------------------------
# Super-title
# ----------------------------------------------------------------------
fig.suptitle(
    "ComBit design space — compression × op-perf × density-robustness\n"
    "real data: combit micro-benchmark (100M rows, 5 backends, 15 densities) + "
    "TPC-H SF10 operating points",
    fontsize=12, fontweight="bold", y=0.985,
)

fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"[OK] wrote {OUT_PNG}")
print(f"[OK] wrote {OUT_PDF}")
