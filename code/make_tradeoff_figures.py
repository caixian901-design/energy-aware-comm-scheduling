"""Trade-off figures for the pilot study on energy-aware communication scheduling.

Figures (all for the Xining plateau site unless stated, 300 dpi):
  fig_t1_makespan_carbon_tradeoff.png  makespan vs total carbon, one panel per
                                       carbon profile, series = liquid fraction f,
                                       points = sync interval k  (the headline)
  fig_t2_makespan_pue.png              makespan vs run PUE, series = f
  fig_t3_start_hour_carbon.png         carbon vs training start hour (time-shift
                                       scheduling value), Qinghai-type profile
  fig_t4_carbon_profiles.png           the two illustrative carbon profiles

Reuses the figure style of the author's make_figures.py (Paper A).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workload_simulation import (   # noqa: E402
    SITES, FS, KS, CI_KINDS, START_HOUR, carbon_profile,
    pue_daily_profile, run_carbon, workload_metrics, sites, WX,
)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

F_COLORS = {0.60: "#8e44ad", 0.80: "#2980b9", 0.90: "#2e8b57",
            0.96: "#c0392b", 1.00: "#e67e22"}
CI_COLORS = {"qinghai": "#16a085", "coal": "#7f8c8d"}

xn = sites["Xining"]


def xining_run(f, k, ci, start_hour=START_HOUR):
    pue_d = pue_daily_profile(xn["t"], xn["rho_r"], f)
    makespan_h, e_it, _, _, p_avg = workload_metrics(k)
    fac_energy, carbon, run_pue = run_carbon(pue_d, e_it, p_avg, makespan_h,
                                             start_hour, ci)
    return makespan_h, run_pue, fac_energy / 1000.0, carbon / 1000.0


# ---------------------------------------------------------------- Fig. T1 (headline)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
for ax, ci in zip(axes, CI_KINDS):
    for f in FS:
        ms = []
        cs = []
        for k in KS:
            m, _, _, c = xining_run(f, k, ci)
            ms.append(m)
            cs.append(c)
        ax.plot(ms, cs, "-o", ms=3.5, lw=1.4, color=F_COLORS[f],
                label=f"f = {f:.2f}")
    ax.set_title("carbon profile: " + ("Qinghai-type (illustrative)"
                 if ci == "qinghai" else "coal-type (illustrative)"), fontsize=8.5)
    ax.set_xlabel("Training makespan (h)")
    ax.set_ylabel("Total carbon (tCO$_2$)")
    ax.legend(loc="upper left", fontsize=6.8, framealpha=0.9)
    ax.text(0.97, 0.03, "points: sync interval k = 1..64", transform=ax.transAxes,
            ha="right", fontsize=6.5, color="#555")
fig.suptitle("Communication interval x liquid cooling x carbon: "
             "makespan vs emissions trade-off (Xining, 2,266 m)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "fig_t1_makespan_carbon_tradeoff.png"))
plt.close(fig)

# ---------------------------------------------------------------- Fig. T2
fig, ax = plt.subplots(figsize=(3.6, 2.7))
for f in FS:
    ms = []
    ps = []
    for k in KS:
        m, p, _, _ = xining_run(f, k, "qinghai")
        ms.append(m)
        ps.append(p)
    ax.plot(ms, ps, "-o", ms=3.5, lw=1.4, color=F_COLORS[f], label=f"f = {f:.2f}")
ax.set_xlabel("Training makespan (h)")
ax.set_ylabel("Run PUE (energy-weighted)")
ax.legend(loc="upper right", fontsize=6.8, framealpha=0.9)
ax.set_title("Makespan vs PUE (Xining, Qinghai-type CI)", fontsize=9)
fig.savefig(os.path.join(OUT, "fig_t2_makespan_pue.png"))
plt.close(fig)

# ---------------------------------------------------------------- Fig. T3 (time shift)
fig, ax = plt.subplots(figsize=(3.6, 2.7))
hours = list(range(24))
for k, c in [(1, "#2980b9"), (8, "#2e8b57"), (64, "#c0392b")]:
    carb = []
    for h0 in hours:
        _, _, _, c0 = xining_run(0.80, k, "qinghai", start_hour=h0)
        carb.append(c0)
    ax.plot(hours, carb, "-o", ms=3, lw=1.4, color=c, label=f"k = {k}")
ax.set_xlabel("Training start hour of day")
ax.set_ylabel("Total carbon (tCO$_2$)")
ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
ax.set_title("Time-shift value (f = 0.80, Qinghai-type CI)", fontsize=9)
mn = min([xining_run(0.80, k, "qinghai", h)[3] for k in KS for h in hours])
mx = max([xining_run(0.80, k, "qinghai", h)[3] for k in KS for h in hours])
ax.text(0.97, 0.04, f"range {mx-mn:.2f} tCO$_2$ "
        f"({(mx-mn)/mx*100:.0f}% of max)", transform=ax.transAxes,
        ha="right", fontsize=6.5, color="#555")
fig.savefig(os.path.join(OUT, "fig_t3_start_hour_carbon.png"))
plt.close(fig)

# ---------------------------------------------------------------- Fig. T4 (profiles)
fig, ax = plt.subplots(figsize=(3.6, 2.7))
h24 = np.arange(24)
for ci in CI_KINDS:
    ax.plot(h24, [carbon_profile(ci, h) for h in h24], "-o", ms=3, lw=1.5,
            color=CI_COLORS[ci], label={"qinghai": "Qinghai-type",
                                        "coal": "coal-type"}[ci])
ax.set_xlabel("Hour of day")
ax.set_ylabel("Carbon intensity (kgCO$_2$/kWh)")
ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
ax.set_title("Illustrative carbon-intensity profiles (model inputs)", fontsize=9)
ax.text(0.97, 0.04, "stylised inputs, not measured grid data",
        transform=ax.transAxes, ha="right", fontsize=6.3, color="#888")
fig.savefig(os.path.join(OUT, "fig_t4_carbon_profiles.png"))
plt.close(fig)

print("figures written to", OUT)
for f in sorted(os.listdir(OUT)):
    if f.startswith("fig_t"):
        print(" ", f, os.path.getsize(os.path.join(OUT, f)), "bytes")
