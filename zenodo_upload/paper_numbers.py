"""Extract exact numbers for the manuscript and run the beta-sensitivity analysis.

Reads data/tradeoff_results.csv (produced by workload_simulation.py) and
recomputes the schedule metrics with a variable staleness penalty beta to
answer: "under what convergence assumptions does the makespan-vs-carbon
trade-off actually exist?"

Outputs:
  data/paper_numbers.txt   exact numbers used in the manuscript (abstract,
                           tables, discussion)
  figures/fig_t5_beta_sensitivity.png
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import workload_simulation as W   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "tradeoff_results.csv")
OUT_TXT = os.path.join(ROOT, "data", "paper_numbers.txt")
OUT_FIG = os.path.join(ROOT, "figures", "fig_t5_beta_sensitivity.png")


def load_rows():
    with open(CSV_PATH, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def get(site, f, k, ci):
    for r in load_rows():
        if (r["site"] == site and abs(float(r["f"]) - f) < 1e-9
                and int(r["k"]) == k and r["ci_profile"] == ci):
            return r
    raise KeyError((site, f, k, ci))


rows = load_rows()
lines = []


def note(*args):
    lines.append(" ".join(str(a) for a in args))


note("=== Headline numbers (Xining, measured TMYx weather, 2,266 m) ===")
note()
r1 = get("Xining", 0.80, 1, "qinghai")      # frequent sync
r64 = get("Xining", 0.80, 64, "qinghai")    # sparse sync
m1, m64 = float(r1["makespan_h"]), float(r64["makespan_h"])
c1, c64 = float(r1["carbon_kg"]), float(r64["carbon_kg"])
note(f"f=0.80, Qinghai-type CI:")
note(f"  k=1 : makespan {m1:.4f} h, carbon {c1:.2f} kg, E_fac {float(r1['facility_energy_kWh']):.2f} kWh, PUE {r1['run_PUE']}")
note(f"  k=64: makespan {m64:.4f} h, carbon {c64:.2f} kg, E_fac {float(r64['facility_energy_kWh']):.2f} kWh, PUE {r64['run_PUE']}")
note(f"  makespan k=64->k=1: +{(m1-m64)/m64*100:.2f}%   (relative to k=64)")
note(f"  carbon   k=64->k=1: {(c1-c64)/c64*100:.2f}%")
note(f"  makespan k=1->k=64: {(m64-m1)/m1*100:.2f}%   (relative to k=1)")
note(f"  carbon   k=1->k=64: {(c64-c1)/c1*100:.2f}%")
note()

note("=== Liquid-fraction lever (Xining, k=1, Qinghai-type CI) ===")
for f in [0.60, 0.80, 0.90, 0.96, 1.00]:
    r = get("Xining", f, 1, "qinghai")
    note(f"  f={f:.2f}: PUE {r['run_PUE']}, carbon {float(r['carbon_kg']):.2f} kg, E_fac {float(r['facility_energy_kWh']):.2f} kWh")
r060 = get("Xining", 0.60, 1, "qinghai")
r100 = get("Xining", 1.00, 1, "qinghai")
note(f"  f=0.60->1.00 carbon: {(float(r100['carbon_kg'])/float(r060['carbon_kg'])-1)*100:.2f}%")
note()

note("=== Grid lever (Xining, f=0.80, k=8) ===")
rq = get("Xining", 0.80, 8, "qinghai")
rc = get("Xining", 0.80, 8, "coal")
note(f"  Qinghai: carbon {float(rq['carbon_kg']):.2f} kg;  Coal: {float(rc['carbon_kg']):.2f} kg")
note(f"  ratio coal/qinghai = {float(rc['carbon_kg'])/float(rq['carbon_kg']):.2f}x")
note(f"  effective CI: qinghai 123 g/kWh, coal 727 g/kWh (model inputs)")
note()

note("=== Time-shift lever (Xining, f=0.80, k=8, Qinghai-type CI) ===")
xn = W.sites["Xining"]
pue_d = W.pue_daily_profile(xn["t"], xn["rho_r"], 0.80)
mh, e_it, _, _, pavg = W.workload_metrics(8)
c_vals = []
for h0 in range(24):
    _, carb, _ = W.run_carbon(pue_d, e_it, pavg, mh, h0, "qinghai")
    c_vals.append(carb)
best, worst = min(c_vals), max(c_vals)
note(f"  carbon by start hour: min {best:.2f} kg at h={c_vals.index(best)}, "
     f"max {worst:.2f} kg at h={c_vals.index(worst)}")
note(f"  max reduction vs worst: {(worst-best)/worst*100:.1f}%")
note()

note("=== Beta sensitivity (Xining, f=0.80, Qinghai-type CI) ===")
betas = [0.0, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]
sweep = []
for b in betas:
    W.BETA = b
    out = {}
    for k in (1, 64):
        mh, e_it, _, _, pavg = W.workload_metrics(k)
        _, carb, _ = W.run_carbon(pue_d, e_it, pavg, mh, 0, "qinghai")
        out[k] = (mh, carb)
    sweep.append((b, out[1], out[64]))
    note(f"  beta={b:.2f}: k=1 M={out[1][0]:.4f}h C={out[1][1]:.2f}kg | "
         f"k=64 M={out[64][0]:.4f}h C={out[64][1]:.2f}kg | "
         f"M-ratio {out[1][0]/out[64][0]:.4f} C-ratio {out[1][1]/out[64][1]:.4f}")

# crossover beta where C(k=1) == C(k=64)
bs = np.linspace(0.0, 0.60, 601)
cross = None
for b in bs:
    W.BETA = float(b)
    m1b, e1, _, _, p1 = W.workload_metrics(1)
    m64b, e64, _, _, p64 = W.workload_metrics(64)
    _, c1b, _ = W.run_carbon(pue_d, e1, p1, m1b, 0, "qinghai")
    _, c64b, _ = W.run_carbon(pue_d, e64, p64, m64b, 0, "qinghai")
    if c1b <= c64b:
        cross = float(b)
        break
note(f"  crossover beta* (C(k=1) <= C(k=64) first time): {cross:.3f}")
note()

with open(OUT_TXT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------------------------------------------------------------------------
# Figure T5: when does the trade-off exist?
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
                     "legend.fontsize": 8, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linestyle": ":", "figure.dpi": 300, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
bs = np.linspace(0.0, 0.60, 601)
m_rat, c_rat, c1s, c64s = [], [], [], []
for b in bs:
    W.BETA = float(b)
    m1b, e1, _, _, p1 = W.workload_metrics(1)
    m64b, e64, _, _, p64 = W.workload_metrics(64)
    _, c1b, _ = W.run_carbon(pue_d, e1, p1, m1b, 0, "qinghai")
    _, c64b, _ = W.run_carbon(pue_d, e64, p64, m64b, 0, "qinghai")
    m_rat.append(m1b / m64b)
    c_rat.append(c1b / c64b)
    c1s.append(c1b)
    c64s.append(c64b)

ax = axes[0]
ax.axhline(1.0, color="#999", lw=1, ls="--")
ax.plot(bs, m_rat, lw=1.8, color="#2980b9", label="makespan ratio M(k=1)/M(k=64)")
ax.plot(bs, c_rat, lw=1.8, color="#c0392b", label="carbon ratio C(k=1)/C(k=64)")
if cross is not None:
    ax.axvline(cross, color="#16a085", lw=1.2, ls=":")
    ax.annotate(f"$\\beta^*$ = {cross:.2f}", xy=(cross, 1.0), xytext=(cross + 0.02, 1.04),
                fontsize=8, color="#16a085")
ax.set_xlabel("Staleness penalty $\\beta$")
ax.set_ylabel("Ratio k=1 / k=64")
ax.set_title("(a) When the trade-off exists", fontsize=9)
ax.legend(loc="upper left", fontsize=7)
ax.text(0.03, 0.10, "left of $\\beta^*$: sparse sync (k=64)\ndominates in both time and carbon",
        transform=ax.transAxes, fontsize=6.5, color="#555")
ax.text(0.55, 0.55, "trade-off region:\nfaster  <=>  cleaner",
        transform=ax.transAxes, fontsize=6.5, color="#555")

ax = axes[1]
ax.plot(bs, np.array(c1s) / 1000.0, lw=1.8, color="#2980b9", label="k = 1 (frequent sync)")
ax.plot(bs, np.array(c64s) / 1000.0, lw=1.8, color="#c0392b", label="k = 64 (sparse sync)")
if cross is not None:
    ax.axvline(cross, color="#16a085", lw=1.2, ls=":")
ax.set_xlabel("Staleness penalty $\\beta$")
ax.set_ylabel("Carbon (tCO$_2$ per job)")
ax.set_title("(b) Absolute carbon, f = 0.80", fontsize=9)
ax.legend(loc="upper left", fontsize=7)

fig.suptitle("The makespan-carbon trade-off exists only if sparse synchronisation "
             "costs convergence (Xining, Qinghai-type CI)", fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT_FIG)
plt.close(fig)
print("\nwrote", OUT_FIG)
