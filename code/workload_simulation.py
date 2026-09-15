"""Energy-aware communication scheduling for distributed AI training clusters.

Extension of the author's prior work:
  X. Cai, "Cooling Architecture Optimization and PUE Modelling for Hyperscale AI
  Data Centres in High-Altitude Low-Pressure Environments", preprint 2026,
  doi: 10.5281/zenodo.22743562

What is added here (the pilot study):
  1. A GPU distributed-training workload model on top of the cooling/PUE model:
     each iteration has a compute phase and a (per-k) all-reduce communication
     phase; an all-reduce is issued once every k iterations (sync interval).
  2. A staleness/convergence model: fewer syncs require more iterations to
     converge, so the *effective* iteration count grows with k.
  3. Hourly facility energy from the Paper-A PUE model driven by measured
     TMYx/EPW weather (Xining / Beijing / Shanghai).
  4. Two illustrative hourly grid carbon-intensity profiles:
       - 'qinghai' : low-carbon, high-renewable profile (stylised)
       - 'coal'    : coal-heavy reference profile (stylised)
     Both are ILLUSTRATIVE model inputs, not measured grid data.
  5. Parameter sweep: sync interval k x liquid-cooling fraction f x carbon
     profile, reporting makespan, IT energy, facility energy, run PUE and
     total carbon emissions.

The single headline question of the pilot:
  "Communication scheduling for distributed training is usually optimised for
   makespan.  What happens to facility energy, PUE and carbon when the sync
   interval, the liquid-cooling fraction and the start time are varied?"

Run:
  python workload_simulation.py
Outputs:
  data/tradeoff_results.csv   (full sweep results, three sites)
Console: summary table for Xining (the plateau site).
"""
import csv
import glob
import os

import numpy as np

# ----------------------------------------------------------------------------
# Paths (edit if the weather files live elsewhere)
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# TMYx 2011-2025 EPW files; override with the WEATHER_DIR environment variable
# (the default path is the author's local copy of the EPW archive).
WX = os.environ.get("WEATHER_DIR", r"C:\Users\29102\dsh_fig\weather")
DATA_OUT = os.path.join(ROOT, "data")
os.makedirs(DATA_OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Paper-A cooling / PUE model (identical functions and parameters)
# ----------------------------------------------------------------------------
P0, T0K, RS = 101.325, 288.15, 287.05
COP_AIR_FC, COP_LIQ_FC = 10.0, 20.0
ETA_C = 0.35
T_AIR_FC, T_LIQ_FC = 20.0, 32.0
T_EVAP_AIR, T_EVAP_LIQ = 287.15, 288.15
LAM_D, LAM_O = 0.055, 0.02
GAMMA = 2.0


def rho_r_at(elev_m):
    """ISA density ratio at elevation elev_m (m)."""
    p = P0 * (1.0 - 2.25577e-5 * elev_m) ** 5.25588
    t = T0K - 0.0065 * elev_m
    rho = (p * 1000.0) / (RS * t)
    rho0 = (P0 * 1000.0) / (RS * T0K)
    return rho / rho0


def read_epw(path):
    """Read dry-bulb temperature (degC) and elevation from an EPW file."""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        rows = list(csv.reader(fh))
    elevation = float(rows[0][9])
    city = rows[0][1]
    temps = []
    for r in rows[8:]:
        if len(r) < 10 or not r[6]:
            continue
        try:
            temps.append(float(r[6]))
        except ValueError:
            continue
    return dict(city=city, elevation=elevation, t=np.array(temps[:8760]))


def cop_air(T, rr):
    """Hourly air-side COP (free cooling + chiller)."""
    if T <= T_AIR_FC:
        return COP_AIR_FC * rr ** GAMMA
    t_cond = T + 5.0 + 273.15
    return max(1.5, ETA_C * T_EVAP_AIR / max(t_cond - T_EVAP_AIR, 3.0))


def cop_liq(T, rr):
    """Hourly liquid-side COP (dry cooler + chiller assist)."""
    if T <= T_LIQ_FC:
        return COP_LIQ_FC * rr ** 0.3
    t_cond = T + 5.0 + 273.15
    return max(2.0, ETA_C * T_EVAP_LIQ / max(t_cond - T_EVAP_LIQ, 3.0))


def pue_hourly(T, rr, f):
    """Hourly PUE for liquid-cooling fraction f (Paper A, Eq. 12)."""
    ca = np.array([cop_air(t, rr) for t in T])
    cl = np.array([cop_liq(t, rr) for t in T])
    return 1.0 + (1.0 - f) / ca + f / cl + LAM_D + LAM_O


# ----------------------------------------------------------------------------
# GPU distributed-training workload model
# ----------------------------------------------------------------------------
# Cluster: one data-parallel training pod of 512 accelerators.
N_GPU = 512
P_C = 700.0                 # W per GPU during compute phase
P_W = 200.0                 # W per GPU during communication / wait phase
T_C = 1.0                   # s per iteration, compute phase
T_S = 0.4                   # s per all-reduce sync (bandwidth-bound, fixed)
N_ITER = 10000              # baseline iteration count
BETA = 0.30                 # staleness penalty: N_eff(k) = N*(1+BETA*(1-1/k))
                            # (fewer syncs -> more iterations to converge)


def workload_metrics(k):
    """Return (makespan_h, IT_energy_kWh, P_IT_compute_kW, P_IT_comm_kW).

    k: all-reduce every k iterations.
    Effective iteration count grows with k (staleness model, stylised).
    """
    n_eff = N_ITER * (1.0 + BETA * (1.0 - 1.0 / k))
    t_comp = n_eff * T_C                 # s
    t_comm = (n_eff / k) * T_S           # s
    makespan_s = t_comp + t_comm
    e_it = (t_comp * P_C + t_comm * P_W) * N_GPU / 3.6e6   # kWh (J -> kWh)
    p_comp = N_GPU * P_C / 1000.0        # kW
    p_wait = N_GPU * P_W / 1000.0        # kW
    p_avg = e_it * 3600.0 / makespan_s   # kW, energy-weighted mean IT power
    return makespan_s / 3600.0, e_it, p_comp, p_wait, p_avg


# ----------------------------------------------------------------------------
# Carbon-intensity profiles (ILLUSTRATIVE, not measured grid data)
# ----------------------------------------------------------------------------
def carbon_profile(kind, hour_of_day, day_of_year=180):
    """kgCO2/kWh at a given hour of day (0-23) and day of year (1-365).

    'qinghai': stylised high-renewable (hydro+wind+solar) profile, low carbon
               intensity, midday solar dip, night wind peak.
    'coal'   : stylised coal-heavy reference, high and nearly flat.
    Both profiles are illustrative inputs for the pilot study.
    """
    if kind == "qinghai":
        base = 0.100 - 0.035 * np.cos(2 * np.pi * (hour_of_day - 14) / 24.0)
        seasonal = 1.0 + 0.08 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0)
        return float(base * seasonal)
    # coal
    return float(0.72 + 0.02 * np.sin(2 * np.pi * hour_of_day / 24.0))


def pue_daily_profile(T, rr, f):
    """24-hour mean PUE curve: hourly PUE averaged over all 365 days."""
    pue_h = pue_hourly(T, rr, f)
    return pue_h.reshape(365, 24).mean(axis=0)


# ----------------------------------------------------------------------------
# Facility + carbon accounting for one training run
# ----------------------------------------------------------------------------
def run_carbon(pue_daily, e_it, p_avg, makespan_h, start_hour, ci_kind):
    """Total carbon (kgCO2) and run PUE for a run starting at start_hour.

    The run occupies makespan_h hours of a repeating daily pattern; the last
    hour is counted pro-rata.  Hourly facility power = p_avg * PUE(hour-of-day).
    """
    full = int(np.floor(makespan_h))
    frac = makespan_h - full
    fac_energy = 0.0
    carbon = 0.0
    for i in range(full):
        h = (start_hour + i) % 24
        fac_energy += p_avg * pue_daily[h]
        carbon += p_avg * pue_daily[h] * carbon_profile(ci_kind, h)
    if frac > 1e-9:
        h = (start_hour + full) % 24
        fac_energy += p_avg * frac * pue_daily[h]
        carbon += p_avg * frac * pue_daily[h] * carbon_profile(ci_kind, h)
    run_pue = fac_energy / e_it if e_it > 0 else float("nan")
    return fac_energy, carbon, run_pue


# ----------------------------------------------------------------------------
# Load weather and precompute hourly PUE for the three sites
# ----------------------------------------------------------------------------
SITES = {
    "Xining": "xining",
    "Beijing": "beijing",
    "Shanghai": "shanghai",
}
sites = {}
for label, key in SITES.items():
    f_epw = glob.glob(os.path.join(WX, key, "*.epw"))[0]
    d = read_epw(f_epw)
    d["rho_r"] = rho_r_at(d["elevation"])
    sites[label] = d
    print(f"{label:9s} elev={d['elevation']:7.1f} m  rho_r={d['rho_r']:.4f}  "
          f"Tmean={d['t'].mean():5.2f} C")

# ----------------------------------------------------------------------------
# Sweep: k x f x carbon profile (Xining full; others in CSV too)
# ----------------------------------------------------------------------------
KS = [1, 2, 4, 8, 16, 32, 64]
FS = [0.60, 0.80, 0.90, 0.96, 1.00]
CI_KINDS = ["qinghai", "coal"]
START_HOUR = 0                     # baseline start; time-shift analysed in figures

rows = []
for label, d in sites.items():
    for f in FS:
        pue_d = pue_daily_profile(d["t"], d["rho_r"], f)
        for k in KS:
            makespan_h, e_it, p_comp, p_wait, p_avg = workload_metrics(k)
            for ci in CI_KINDS:
                fac_energy, carbon, run_pue = run_carbon(
                    pue_d, e_it, p_avg, makespan_h, START_HOUR, ci)
                rows.append({
                    "site": label, "elevation_m": d["elevation"], "rho_r": d["rho_r"],
                    "f": f, "k": k, "ci_profile": ci,
                    "makespan_h": round(makespan_h, 4),
                    "IT_energy_kWh": round(e_it, 2),
                    "facility_energy_kWh": round(fac_energy, 2),
                    "run_PUE": round(run_pue, 5),
                    "carbon_kg": round(carbon, 1),
                    "carbon_t": round(carbon / 1000.0, 4),
                })

csv_path = os.path.join(DATA_OUT, "tradeoff_results.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)
print("\nresults written to", csv_path, f"({len(rows)} rows)")

# ----------------------------------------------------------------------------
# Console summary: Xining, both carbon profiles, makespan & carbon vs k
# ----------------------------------------------------------------------------
print("\n=== Xining (plateau, 2,266 m): makespan / PUE / carbon vs sync interval ===")
print(f"{'k':>3s} {'f':>5s} {'CI':>8s} {'M(h)':>7s} {'PUE':>7s} {'E_fac(MWh)':>11s} "
      f"{'Carbon(t)':>10s} {'effCI(g)':>8s}")
xn = sites["Xining"]
for ci in CI_KINDS:
    for f in [0.6, 0.8, 0.96, 1.0]:
        pue_d = pue_daily_profile(xn["t"], xn["rho_r"], f)
        for k in [1, 8, 64]:
            makespan_h, e_it, _, _, p_avg = workload_metrics(k)
            fac_energy, carbon, run_pue = run_carbon(
                pue_d, e_it, p_avg, makespan_h, START_HOUR, ci)
            eff = carbon / fac_energy * 1000.0
            print(f"{k:3d} {f:5.2f} {ci:>8s} {makespan_h:7.3f} {run_pue:7.4f} "
                  f"{fac_energy/1000.0:11.2f} {carbon/1000.0:10.3f} {eff:8.0f}")

print("\ndone.")
