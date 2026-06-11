
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import gurobipy as gp
from gurobipy import GRB

# Çıktıyı buffer'lamadan akıt (PyCharm Run penceresinde anında görünür)
sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# 0. VERİ YÜKLEME
# =============================================================================

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

donors_df   = pd.read_csv(os.path.join(DATA_DIR, "donors.csv"))
dc_df       = pd.read_csv(os.path.join(DATA_DIR, "distribution_centers.csv"))
charity_df  = pd.read_csv(os.path.join(DATA_DIR, "charities.csv"))
scenario_df = pd.read_csv(os.path.join(DATA_DIR, "scenarios.csv"))
dist_df     = pd.read_csv(os.path.join(DATA_DIR, "distance_matrix.csv"))

donors    = donors_df["donor_id"].tolist()
dcs       = dc_df["dc_id"].tolist()
charities = charity_df["charity_id"].tolist()

I = len(donors)
J = len(dcs)
K = len(charities)
S = len(scenario_df)

print(f"Problem boyutu: {I} donör | {J} DC | {K} hayır kurumu | {S} senaryo",
      flush=True)

# =============================================================================
# 1. PARAMETRE HESAPLAMA
# =============================================================================

p_s = [1.0 / S] * S

supply_base = donors_df.set_index("donor_id")["avg_daily_surplus_kg"]
demand_base = charity_df.set_index("charity_id")["avg_daily_demand_kg"]
supply_mult = scenario_df["donor_supply_multiplier"].values
demand_mult = scenario_df["demand_multiplier"].values

S_is = {}
for i, did in enumerate(donors):
    for s in range(S):
        S_is[i, s] = float(supply_base[did]) * float(supply_mult[s])

D_ks = {}
for k, cid in enumerate(charities):
    for s in range(S):
        D_ks[k, s] = float(demand_base[cid]) * float(demand_mult[s])

C_j  = {j: float(dc_df.iloc[j]["capacity_kg"]) for j in range(J)}
fc_j = {j: float(dc_df.iloc[j]["fixed_cost"])  for j in range(J)}

refrig_cost_map = {"high": 1400.0, "medium": 1000.0, "low": 600.0}
eps_j = {
    j: refrig_cost_map.get(dc_df.iloc[j]["refrigeration_level"], 1000.0)
    for j in range(J)
}

kappa0 = 0.05
theta  = 4.0
eps_ac = 1500.0
R_gas  = 8.314
kap    = kappa0 * np.exp(-eps_ac / (R_gas * (theta + 273.15)))
print(f"Decay rate κ(θ={theta}°C) = {kap:.6f} per km", flush=True)

max_shelf = donors_df["shelf_life_hours"].max()
L_i = {
    did: 0.5 + 0.5 * (
        float(donors_df.loc[donors_df["donor_id"] == did,
                            "shelf_life_hours"].values[0]) / max_shelf
    )
    for did in donors
}

dist_lookup = dist_df.set_index(
    ["origin_id", "destination_id"])["distance_km"].to_dict()

def get_dist(o, d):
    v = dist_lookup.get((o, d))
    if v is not None:
        return float(v)
    v = dist_lookup.get((d, o))
    return float(v) if v is not None else 9999.0

d_ij = {(i, j): get_dist(donors[i], dcs[j])    for i in range(I) for j in range(J)}
d_jk = {(j, k): get_dist(dcs[j], charities[k]) for j in range(J) for k in range(K)}

alpha_ij = {}
for i in range(I):
    for j in range(J):
        val = 1.0 - L_i[donors[i]] * np.exp(-kap * d_ij[i, j])
        alpha_ij[i, j] = min(max(val, 0.0), 0.99)

beta_jk = {}
for j in range(J):
    for k in range(K):
        val = 1.0 - np.exp(-kap * d_jk[j, k])
        beta_jk[j, k] = min(max(val, 0.0), 0.99)

gamma = 0.05
phi   = 0.03
pi_p  = 2.0
eta   = 0.08

M_big = max(S_is.values()) * I * 1.5

MIN_SERVICE_RATE = 0.30
total_demand_s = {s: sum(D_ks[k, s] for k in range(K)) for s in range(S)}
print(f"Hizmet seviyesi kısıtı: her senaryoda en az "
      f"%{MIN_SERVICE_RATE*100:.0f} talep karşılanmalı", flush=True)

# =============================================================================
# 2. TEMEL MODEL BUILDER
# =============================================================================

def build_model(m, e1=None, e3=None, obj="Z2",
                method="epsilon",
                r1=1.0, r3=1.0, aug_eps=1e-4):
    x = m.addVars(I, J, S, lb=0, name="x")
    r = m.addVars(I, J, S, lb=0, name="r")
    w = m.addVars(J, K, S, lb=0, name="w")
    y = m.addVars(J, K, S, lb=0, name="y")
    Omega     = m.addVars(K, S, lb=0, ub=1, name="Omega")
    Omega_bar = m.addVars(S,    lb=0, ub=1, name="Obar")
    u         = m.addVars(K, S, lb=0,       name="u")
    z         = m.addVars(J, vtype=GRB.BINARY, name="z")

    s1 = m.addVar(lb=0, name="s1") if (method in ("augmecon", "augmecon2") and e1 is not None) else None
    s3 = m.addVar(lb=0, name="s3") if (method in ("augmecon", "augmecon2") and e3 is not None) else None

    m.update()

    transport_cost = gp.quicksum(
        p_s[s] * (gamma + phi) * (
            gp.quicksum(d_ij[i,j] * x[i,j,s] for i in range(I) for j in range(J)) +
            gp.quicksum(d_jk[j,k] * w[j,k,s] for j in range(J) for k in range(K))
        ) for s in range(S)
    )
    penalty_cost = gp.quicksum(
        p_s[s] * (pi_p + eta) * (
            gp.quicksum(alpha_ij[i,j] * x[i,j,s] for i in range(I) for j in range(J)) +
            gp.quicksum(beta_jk[j,k]  * w[j,k,s] for j in range(J) for k in range(K))
        ) for s in range(S)
    )
    fixed_cost = gp.quicksum((fc_j[j] + eps_j[j]) * z[j] for j in range(J))

    Z1_expr = transport_cost + penalty_cost + fixed_cost
    Z2_expr = gp.quicksum(
        p_s[s] * y[j,k,s] for s in range(S) for j in range(J) for k in range(K)
    )
    Z3_expr = gp.quicksum(
        p_s[s] * u[k,s] for s in range(S) for k in range(K)
    )

    for i in range(I):
        for s in range(S):
            m.addConstr(gp.quicksum(x[i,j,s] for j in range(J)) <= S_is[i,s])
    for i in range(I):
        for j in range(J):
            for s in range(S):
                m.addConstr(r[i,j,s] == (1.0 - alpha_ij[i,j]) * x[i,j,s])
    for j in range(J):
        for s in range(S):
            m.addConstr(
                gp.quicksum(r[i,j,s] for i in range(I)) ==
                gp.quicksum(w[j,k,s] for k in range(K)))
    for j in range(J):
        for s in range(S):
            m.addConstr(
                gp.quicksum(r[i,j,s] for i in range(I)) <= C_j[j] * z[j])
    for j in range(J):
        for k in range(K):
            for s in range(S):
                m.addConstr(y[j,k,s] == (1.0 - beta_jk[j,k]) * w[j,k,s])
    for k in range(K):
        for s in range(S):
            m.addConstr(gp.quicksum(y[j,k,s] for j in range(J)) <= D_ks[k,s])
    for k in range(K):
        for s in range(S):
            m.addConstr(
                Omega[k,s] * D_ks[k,s] ==
                gp.quicksum(y[j,k,s] for j in range(J)))
    for s in range(S):
        m.addConstr(
            Omega_bar[s] * K == gp.quicksum(Omega[k,s] for k in range(K)))
    for k in range(K):
        for s in range(S):
            m.addConstr(u[k,s] >= Omega[k,s] - Omega_bar[s])
            m.addConstr(u[k,s] >= Omega_bar[s] - Omega[k,s])
    for i in range(I):
        for j in range(J):
            for s in range(S):
                m.addConstr(x[i,j,s] <= M_big * z[j])
    for j in range(J):
        for k in range(K):
            for s in range(S):
                m.addConstr(y[j,k,s] <= M_big * z[j])

    for s in range(S):
        m.addConstr(
            gp.quicksum(y[j,k,s] for j in range(J) for k in range(K))
            >= MIN_SERVICE_RATE * total_demand_s[s],
            name=f"min_service_s{s}"
        )

    if method == "epsilon":
        if e1 is not None:
            m.addConstr(Z1_expr <= e1)
        if e3 is not None:
            m.addConstr(Z3_expr <= e3)
    else:
        if e1 is not None and s1 is not None:
            m.addConstr(Z1_expr + s1 == e1)
        if e3 is not None and s3 is not None:
            m.addConstr(Z3_expr + s3 == e3)

    if obj == "Z1":
        m.setObjective(Z1_expr, GRB.MINIMIZE)
    elif obj == "Z3":
        m.setObjective(Z3_expr, GRB.MINIMIZE)
    elif method == "epsilon":
        m.setObjective(Z2_expr, GRB.MAXIMIZE)
    elif method == "augmecon":
        aug = 0.0
        if s1 is not None:
            aug += aug_eps * s1
        if s3 is not None:
            aug += aug_eps * s3
        m.setObjective(Z2_expr + aug, GRB.MAXIMIZE)
    elif method == "augmecon2":
        aug = 0.0
        if s1 is not None and r1 > 0:
            aug += aug_eps * (s1 / r1)
        if s3 is not None and r3 > 0:
            aug += aug_eps * 1e-1 * (s3 / r3)
        m.setObjective(Z2_expr + aug, GRB.MAXIMIZE)

    return Z1_expr, Z2_expr, Z3_expr, z, Omega, Omega_bar, s1, s3


def solve_one(e1=None, e3=None, obj="Z2",
              method="epsilon",
              r1=1.0, r3=1.0, aug_eps=1e-4,
              fixed_objectives=None,
              verbose=False, time_limit=60):
    m = gp.Model()
    m.setParam("OutputFlag", 1 if verbose else 0)
    m.setParam("TimeLimit", time_limit)

    Z1_expr, Z2_expr, Z3_expr, z, Omega, Omega_bar, s1, s3 = \
        build_model(m, e1=e1, e3=e3, obj=obj, method=method,
                    r1=r1, r3=r3, aug_eps=aug_eps)

    if fixed_objectives:
        for name, val in fixed_objectives.items():
            tol_rel = 1e-4
            tol_abs = 1e-3
            if name == "Z1":
                m.addConstr(Z1_expr <= val + max(tol_abs, tol_rel*abs(val)))
                m.addConstr(Z1_expr >= val - max(tol_abs, tol_rel*abs(val)))
            elif name == "Z2":
                m.addConstr(Z2_expr <= val + max(tol_abs, tol_rel*abs(val)))
                m.addConstr(Z2_expr >= val - max(tol_abs, tol_rel*abs(val)))
            elif name == "Z3":
                m.addConstr(Z3_expr <= val + 1e-4)
                m.addConstr(Z3_expr >= val - 1e-4)

    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or m.SolCount == 0:
        m.dispose()
        return None

    result = {
        "Z1": Z1_expr.getValue(),
        "Z2": Z2_expr.getValue(),
        "Z3": Z3_expr.getValue(),
        "z":  {j: round(z[j].X) for j in range(J)},
        "Omega":     {(k,s): Omega[k,s].X for k in range(K) for s in range(S)},
        "Omega_bar": {s: Omega_bar[s].X   for s in range(S)},
        "s1": s1.X if s1 is not None else None,
        "s3": s3.X if s3 is not None else None,
        "models_solved": 1,
    }
    m.dispose()
    return result

# =============================================================================
# 3. PAYOFF TABLOSU — LEXICOGRAPHIC
# =============================================================================

def compute_payoff():
    print("\n" + "="*60, flush=True)
    print("Lexicographic Payoff Tablosu (Mavrotas 2009, Tablo 2)", flush=True)
    print("="*60, flush=True)

    payoff_table = {}

    print("  [Satır 1] min Z1 → max Z2 (Z1 sabit) → min Z3 (Z1, Z2 sabit)",
          flush=True)
    s = solve_one(obj="Z1", method="epsilon")
    if s is None:
        print("  HATA: min Z1 INFEASIBLE.", flush=True)
        return None
    Z1_star = s["Z1"]
    s = solve_one(obj="Z2", method="epsilon",
                  fixed_objectives={"Z1": Z1_star})
    Z2_at = s["Z2"]
    s = solve_one(obj="Z3", method="epsilon",
                  fixed_objectives={"Z1": Z1_star, "Z2": Z2_at})
    Z3_at = s["Z3"]
    payoff_table["min_Z1"] = {"Z1": Z1_star, "Z2": Z2_at, "Z3": Z3_at}

    print("  [Satır 2] max Z2 → min Z1 (Z2 sabit) → min Z3 (Z1, Z2 sabit)",
          flush=True)
    s = solve_one(obj="Z2", method="epsilon")
    Z2_star = s["Z2"]
    s = solve_one(obj="Z1", method="epsilon",
                  fixed_objectives={"Z2": Z2_star})
    Z1_at = s["Z1"]
    s = solve_one(obj="Z3", method="epsilon",
                  fixed_objectives={"Z2": Z2_star, "Z1": Z1_at})
    Z3_at = s["Z3"]
    payoff_table["max_Z2"] = {"Z1": Z1_at, "Z2": Z2_star, "Z3": Z3_at}

    print("  [Satır 3] min Z3 → min Z1 (Z3 sabit) → max Z2 (Z3, Z1 sabit)",
          flush=True)
    s = solve_one(obj="Z3", method="epsilon")
    Z3_star = s["Z3"]
    s = solve_one(obj="Z1", method="epsilon",
                  fixed_objectives={"Z3": Z3_star})
    Z1_at = s["Z1"]
    s = solve_one(obj="Z2", method="epsilon",
                  fixed_objectives={"Z3": Z3_star, "Z1": Z1_at})
    Z2_at = s["Z2"]
    payoff_table["min_Z3"] = {"Z1": Z1_at, "Z2": Z2_at, "Z3": Z3_star}

    Z1_vals = [payoff_table[r]["Z1"] for r in payoff_table]
    Z2_vals = [payoff_table[r]["Z2"] for r in payoff_table]
    Z3_vals = [payoff_table[r]["Z3"] for r in payoff_table]

    Z1_ideal = min(Z1_vals);   Z1_nadir = max(Z1_vals)
    Z2_ideal = max(Z2_vals);   Z2_nadir = min(Z2_vals)
    Z3_ideal = min(Z3_vals);   Z3_nadir = max(Z3_vals)

    if Z3_nadir - Z3_ideal < 1e-4:
        s_only_Z2 = solve_one(obj="Z2", method="epsilon")
        s_only_Z1 = solve_one(obj="Z1", method="epsilon")
        candidates = [s_only_Z2["Z3"], s_only_Z1["Z3"]]
        Z3_nadir = max(Z3_nadir, max(candidates))
        if Z3_nadir - Z3_ideal < 1e-4:
            Z3_nadir = Z3_ideal + 0.5

    r1 = max(Z1_nadir - Z1_ideal, 1e-6)
    r3 = max(Z3_nadir - Z3_ideal, 1e-6)

    print("\n  Lex Payoff Tablosu:", flush=True)
    for row, vals in payoff_table.items():
        print(f"    {row:>7}: Z1={vals['Z1']:>12,.0f}  "
              f"Z2={vals['Z2']:>10,.1f}  Z3={vals['Z3']:>7.4f}", flush=True)
    print(f"\n  Z1: ideal={Z1_ideal:>11,.0f}  nadir={Z1_nadir:>11,.0f}  r1={r1:,.0f}",
          flush=True)
    print(f"  Z2: ideal={Z2_ideal:>11,.0f}  nadir={Z2_nadir:>11,.0f}", flush=True)
    print(f"  Z3: ideal={Z3_ideal:>11.4f}  nadir={Z3_nadir:>11.4f}  r3={r3:.4f}",
          flush=True)

    return {
        "Z1_ideal": Z1_ideal, "Z1_nadir": Z1_nadir,
        "Z2_ideal": Z2_ideal, "Z2_nadir": Z2_nadir,
        "Z3_ideal": Z3_ideal, "Z3_nadir": Z3_nadir,
        "r1": r1, "r3": r3,
    }

# =============================================================================
# YARDIMCI: progres bar
# =============================================================================
def fmt_eta(seconds):
    if seconds < 60:
        return f"{seconds:.0f}sn"
    elif seconds < 3600:
        return f"{seconds/60:.1f}dk"
    else:
        return f"{seconds/3600:.1f}sa"

# =============================================================================
# 4. EPSILON-CONSTRAINT — İlerleme göstergeli
# =============================================================================

def run_epsilon(payoff, n1=20, n3=15):
    print("\n" + "="*70, flush=True)
    print(f"Epsilon-Constraint  (grid: e3={n3} × e1={n1} = {n1*n3} nokta)",
          flush=True)
    print("Sembol: '.' = feasible | 'X' = infeasible", flush=True)
    print("-"*70, flush=True)

    Z1_ideal = payoff["Z1_ideal"];  Z1_nadir = payoff["Z1_nadir"]
    Z3_ideal = payoff["Z3_ideal"];  Z3_nadir = payoff["Z3_nadir"]

    t0 = time.time()
    solutions = []
    models_solved = 0
    skipped_infeas = 0
    total = n1 * n3

    e3_vals = np.linspace(Z3_nadir, Z3_ideal, n3)
    e1_vals = np.linspace(Z1_nadir, Z1_ideal, n1)

    for i_e3, e3 in enumerate(e3_vals):
        row_solutions = 0
        print(f"  [{i_e3+1:>2}/{n3}] e3={e3:6.3f} ", end="", flush=True)
        t_row_start = time.time()
        for e1 in e1_vals:
            sol = solve_one(e1=e1, e3=e3, obj="Z2", method="epsilon")
            models_solved += 1
            if sol:
                sol["e1"] = e1; sol["e3"] = e3
                solutions.append(sol)
                row_solutions += 1
                print(".", end="", flush=True)
            else:
                skipped_infeas += 1
                print("X", end="", flush=True)

        elapsed_row = time.time() - t_row_start
        elapsed_total = time.time() - t0
        progress = models_solved / total
        eta = elapsed_total / progress - elapsed_total if progress > 0 else 0
        print(f" | {row_solutions:>2}/{n1} feasible | "
              f"satır={elapsed_row:.0f}sn | toplam={fmt_eta(elapsed_total)} | "
              f"ETA={fmt_eta(eta)}", flush=True)

    elapsed = time.time() - t0
    pareto = filter_pareto(solutions)
    print("-"*70, flush=True)
    print(f"  ε-Constraint BİTTİ | Model: {models_solved} | "
          f"Infeasible: {skipped_infeas} | Pareto: {len(pareto)} | "
          f"Süre: {fmt_eta(elapsed)}", flush=True)
    return pareto, models_solved, elapsed

# =============================================================================
# 5. AUGMECON
# =============================================================================

def run_augmecon(payoff, n1=20, n3=15, aug_eps=1e-6):
    print("\n" + "="*70, flush=True)
    print(f"AUGMECON (aug_eps={aug_eps:.0e}, grid: {n3}×{n1}={n1*n3} nokta)",
          flush=True)
    print("Sembol: '.' = feasible | 'X' = infeasible", flush=True)
    print("-"*70, flush=True)

    Z1_ideal = payoff["Z1_ideal"];  Z1_nadir = payoff["Z1_nadir"]
    Z3_ideal = payoff["Z3_ideal"];  Z3_nadir = payoff["Z3_nadir"]
    r1 = payoff["r1"];  r3 = payoff["r3"]

    t0 = time.time()
    solutions = []
    models_solved = 0
    skipped_infeas = 0
    total = n1 * n3

    e3_vals = np.linspace(Z3_nadir, Z3_ideal, n3)
    e1_vals = np.linspace(Z1_nadir, Z1_ideal, n1)

    for i_e3, e3 in enumerate(e3_vals):
        row_solutions = 0
        print(f"  [{i_e3+1:>2}/{n3}] e3={e3:6.3f} ", end="", flush=True)
        t_row_start = time.time()
        for e1 in e1_vals:
            sol = solve_one(e1=e1, e3=e3, obj="Z2",
                            method="augmecon", r1=r1, r3=r3,
                            aug_eps=aug_eps)
            models_solved += 1
            if sol:
                sol["e1"] = e1; sol["e3"] = e3
                solutions.append(sol)
                row_solutions += 1
                print(".", end="", flush=True)
            else:
                skipped_infeas += 1
                print("X", end="", flush=True)

        elapsed_row = time.time() - t_row_start
        elapsed_total = time.time() - t0
        progress = models_solved / total
        eta = elapsed_total / progress - elapsed_total if progress > 0 else 0
        print(f" | {row_solutions:>2}/{n1} feasible | "
              f"satır={elapsed_row:.0f}sn | toplam={fmt_eta(elapsed_total)} | "
              f"ETA={fmt_eta(eta)}", flush=True)

    elapsed = time.time() - t0
    pareto = filter_pareto(solutions)
    print("-"*70, flush=True)
    print(f"  AUGMECON BİTTİ | Model: {models_solved} | "
          f"Infeasible: {skipped_infeas} | Pareto: {len(pareto)} | "
          f"Süre: {fmt_eta(elapsed)}", flush=True)
    return pareto, models_solved, elapsed

# =============================================================================
# 6. AUGMECON2 — JUMP göstergeli
# =============================================================================

def run_augmecon2(payoff, n1=20, n3=15, aug_eps=1e-3):
    print("\n" + "="*70, flush=True)
    print(f"AUGMECON2 (aug_eps={aug_eps:.0e}, grid: {n3}×{n1}={n1*n3} nokta)",
          flush=True)
    print("Sembol: '.' = feasible | 'X' = infeasible-break | '>N' = N adım JUMP",
          flush=True)
    print("-"*70, flush=True)

    Z1_ideal = payoff["Z1_ideal"];  Z1_nadir = payoff["Z1_nadir"]
    Z3_ideal = payoff["Z3_ideal"];  Z3_nadir = payoff["Z3_nadir"]
    r1 = payoff["r1"];  r3 = payoff["r3"]

    t0 = time.time()
    solutions = []
    models_solved = 0
    skipped_infeas = 0
    total_jumps = 0

    e3_vals = np.linspace(Z3_nadir, Z3_ideal, n3)
    e1_vals = np.linspace(Z1_nadir, Z1_ideal, n1)
    step1 = abs(e1_vals[1] - e1_vals[0]) if len(e1_vals) > 1 else 1.0

    for i_e3, e3 in enumerate(e3_vals):
        row_solutions = 0
        row_jumps = 0
        print(f"  [{i_e3+1:>2}/{n3}] e3={e3:6.3f} ", end="", flush=True)
        t_row_start = time.time()
        j1 = 0
        while j1 < len(e1_vals):
            e1 = e1_vals[j1]
            sol = solve_one(e1=e1, e3=e3, obj="Z2",
                            method="augmecon2", r1=r1, r3=r3,
                            aug_eps=aug_eps)
            models_solved += 1

            if sol:
                sol["e1"] = e1; sol["e3"] = e3
                solutions.append(sol)
                row_solutions += 1
                jump = 0
                if sol["s1"] is not None and step1 > 1e-9:
                    jump = max(0, int(sol["s1"] / step1))
                total_jumps += jump
                row_jumps += jump
                if jump > 0:
                    print(f".>{jump}", end="", flush=True)
                else:
                    print(".", end="", flush=True)
                j1 += jump + 1
            else:
                skipped_infeas += 1
                print("X", end="", flush=True)
                break

        elapsed_row = time.time() - t_row_start
        elapsed_total = time.time() - t0
        # Toplam tahmini için kabaca grid-based progress kullan
        rough_progress = (i_e3 + 1) / n3
        eta = elapsed_total / rough_progress - elapsed_total if rough_progress > 0 else 0
        print(f" | {row_solutions:>2} feas + {row_jumps} jump | "
              f"satır={elapsed_row:.0f}sn | toplam={fmt_eta(elapsed_total)} | "
              f"ETA={fmt_eta(eta)}", flush=True)

    elapsed = time.time() - t0
    pareto = filter_pareto(solutions)
    print("-"*70, flush=True)
    print(f"  AUGMECON2 BİTTİ | Model: {models_solved} | "
          f"Toplam JUMP: {total_jumps} grid noktası tasarrufu | "
          f"Pareto: {len(pareto)} | Süre: {fmt_eta(elapsed)}", flush=True)
    return pareto, models_solved, elapsed

# =============================================================================
# 7. PARETO FİLTRE
# =============================================================================

# =============================================================================
# 7. PARETO FİLTRE — sağlam duplicate eliminasyonu
# =============================================================================

def filter_pareto(solutions):
    """
    İki aşamalı filtre:
    Aşama 1: Agresif bucket yuvarlaması ile sayısal duplikatları birleştir.
             - Z1: 50$ hassasiyet (Gurobi alternatif optima gürültüsünü yakala)
             - Z2: tam sayı (ondalık gürültüyü elimine)
             - Z3: 3 ondalık basamak
    Aşama 2: Standart güçlü-Pareto filtresi.
    """
    # Trivial çözümleri çıkar
    solutions = [s for s in solutions if s["Z2"] > 1e-3]

    if not solutions:
        return []

    # ===== Aşama 1: Sayısal duplikatları birleştir =====
    bucket = {}
    for s in solutions:
        # Z1: 50$ hassasiyet | Z2: tam sayı | Z3: 3 ondalık basamak
        key = (round(s["Z2"]), round(s["Z3"], 2))
        if key not in bucket or s["Z3"] < bucket[key]["Z3"] - 1e-6:
            bucket[key] = s
    unique = list(bucket.values())

    # ===== Aşama 2: Güçlü Pareto filtresi =====
    pareto = []
    for i, a in enumerate(unique):
        t1 = max(10.0, 1e-3 * abs(a["Z1"]))   # 10$ veya %0.1
        t2 = max(1.0,  1e-3 * abs(a["Z2"]))   # 1 kg veya %0.1
        t3 = max(1e-3, 1e-3 * abs(a["Z3"]))   # 0.001 veya %0.1
        dominated = False
        for j, b in enumerate(unique):
            if i == j:
                continue
            # b, a'yı domine ediyor mu?
            if (b["Z1"] <= a["Z1"] + t1 and
                b["Z2"] >= a["Z2"] - t2 and
                b["Z3"] <= a["Z3"] + t3 and
                (b["Z1"] < a["Z1"] - t1 or
                 b["Z2"] > a["Z2"] + t2 or
                 b["Z3"] < a["Z3"] - t3)):
                dominated = True
                break
        if not dominated:
            pareto.append(a)
    return pareto

# =============================================================================
# 8. KARŞILAŞTIRMA TABLOSU
# =============================================================================

def print_comparison(payoff,
                     eps_pareto, eps_models, eps_time,
                     aug_pareto, aug_models, aug_time,
                     aug2_pareto, aug2_models, aug2_time):

    print("\n" + "="*78, flush=True)
    print("YÖNTEM KARŞILAŞTIRMASI — Mavrotas & Florios (2013) Tablo 6 stili",
          flush=True)
    print(f"{'Kriter':<32} {'ε-Constraint':>14} {'AUGMECON':>14} {'AUGMECON2':>14}",
          flush=True)
    print("-"*78, flush=True)

    def stats(pareto):
        if not pareto:
            return "—", "—", "—", "—"
        df = pd.DataFrame(pareto)
        return (f"{len(pareto)}",
                f"{df['Z1'].min():,.0f}–{df['Z1'].max():,.0f}",
                f"{df['Z2'].min():,.0f}–{df['Z2'].max():,.0f}",
                f"{df['Z3'].min():.2f}–{df['Z3'].max():.2f}")

    e_n, e_z1, e_z2, e_z3 = stats(eps_pareto)
    a_n, a_z1, a_z2, a_z3 = stats(aug_pareto)
    a2_n, a2_z1, a2_z2, a2_z3 = stats(aug2_pareto)

    print(f"{'Models solved':<32} {eps_models:>14} {aug_models:>14} {aug2_models:>14}",
          flush=True)
    print(f"{'Güçlü Pareto |PF*|':<32} {e_n:>14} {a_n:>14} {a2_n:>14}",
          flush=True)
    print(f"{'Z1 aralığı ($)':<32} {e_z1:>14} {a_z1:>14} {a2_z1:>14}",
          flush=True)
    print(f"{'Z2 aralığı (kg)':<32} {e_z2:>14} {a_z2:>14} {a2_z2:>14}",
          flush=True)
    print(f"{'Z3 aralığı':<32} {e_z3:>14} {a_z3:>14} {a2_z3:>14}",
          flush=True)
    print(f"{'CPU süresi (sn)':<32} {eps_time:>14.1f} {aug_time:>14.1f} {aug2_time:>14.1f}",
          flush=True)

    if aug2_models > 0:
        sp_eps = eps_models / aug2_models
        sp_aug = aug_models / aug2_models
        print(f"{'AUGMECON2 hızlanma (model)':<32} {sp_eps:>13.2f}× {sp_aug:>13.2f}× {'1.00×':>14}",
              flush=True)
    if aug2_time > 0:
        sp_t_eps = eps_time / aug2_time
        sp_t_aug = aug_time / aug2_time
        print(f"{'AUGMECON2 hızlanma (CPU)':<32} {sp_t_eps:>13.2f}× {sp_t_aug:>13.2f}× {'1.00×':>14}",
              flush=True)
    print("="*78, flush=True)

    print("\nPayoff Tablosu (lex):", flush=True)
    print(f"  Z1: ideal={payoff['Z1_ideal']:,.0f}  nadir={payoff['Z1_nadir']:,.0f}",
          flush=True)
    print(f"  Z2: ideal={payoff['Z2_ideal']:,.0f}  nadir={payoff['Z2_nadir']:,.0f}",
          flush=True)
    print(f"  Z3: ideal={payoff['Z3_ideal']:.4f}   nadir={payoff['Z3_nadir']:.4f}",
          flush=True)

# =============================================================================
# 9. GÖRSELLEŞTİRME
# =============================================================================

def plot_comparison(eps_pareto, aug_pareto, aug2_pareto):
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.38)

    colors = {"ε-Constraint": "#E65100",
              "AUGMECON":     "#1565C0",
              "AUGMECON2":    "#2E8B57"}

    datasets = [
        ("ε-Constraint", eps_pareto),
        ("AUGMECON",     aug_pareto),
        ("AUGMECON2",    aug2_pareto),
    ]

    ax1 = fig.add_subplot(gs[0, 0])
    for label, pareto in datasets:
        if pareto:
            df = pd.DataFrame(pareto).sort_values("Z2")
            ax1.plot(df["Z2"], df["Z1"]/1e3, "o-",
                     color=colors[label], lw=2, ms=5, label=label, alpha=0.8)
    ax1.set_xlabel("Z₂ — Teslimat (kg)", fontsize=9)
    ax1.set_ylabel("Z₁ — Maliyet (k$)", fontsize=9)
    ax1.set_title("Z₁ – Z₂ Pareto Sınırı", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    for label, pareto in datasets:
        if pareto:
            df = pd.DataFrame(pareto).sort_values("Z3")
            ax2.plot(df["Z3"], df["Z1"]/1e3, "s-",
                     color=colors[label], lw=2, ms=5, label=label, alpha=0.8)
    ax2.set_xlabel("Z₃ — Adalet sapması", fontsize=9)
    ax2.set_ylabel("Z₁ — Maliyet (k$)", fontsize=9)
    ax2.set_title("Z₁ – Z₃ Pareto Sınırı", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    for label, pareto in datasets:
        if pareto:
            df = pd.DataFrame(pareto)
            ax3.scatter(df["Z2"], df["Z3"],
                        color=colors[label], s=40,
                        edgecolors="white", lw=0.5, label=label, alpha=0.8)
    ax3.set_xlabel("Z₂ — Teslimat (kg)", fontsize=9)
    ax3.set_ylabel("Z₃ — Adalet sapması", fontsize=9)
    ax3.set_title("Z₂ – Z₃ Dağılımı", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, :2], projection="3d")
    for label, pareto in datasets:
        if pareto:
            df = pd.DataFrame(pareto)
            ax4.scatter(df["Z2"], df["Z3"], df["Z1"]/1e3,
                        color=colors[label], s=40,
                        edgecolors="white", lw=0.3, label=label, alpha=0.8)
    ax4.set_xlabel("Z₂ (kg)", fontsize=8, labelpad=6)
    ax4.set_ylabel("Z₃", fontsize=8, labelpad=6)
    ax4.set_zlabel("Z₁ (k$)", fontsize=8, labelpad=6)
    ax4.set_title("3D Pareto Yüzeyi", fontsize=10, fontweight="bold")
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(gs[1, 2])
    labels = ["ε-Constraint", "AUGMECON", "AUGMECON2"]
    counts = [len(eps_pareto), len(aug_pareto), len(aug2_pareto)]
    bars = ax5.bar(labels, counts,
                   color=[colors[l] for l in labels],
                   alpha=0.85, edgecolor="white")
    for bar, cnt in zip(bars, counts):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 str(cnt), ha="center", fontsize=10, fontweight="bold")
    ax5.set_ylabel("|PF*| — Güçlü Pareto Nokta Sayısı", fontsize=9)
    ax5.set_title("Yöntem Karşılaştırması\n|PF*|", fontsize=10, fontweight="bold")
    ax5.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Epsilon-Constraint vs AUGMECON vs AUGMECON2\n"
        f"Stokastik Gıda Bankası — {I} donör · {J} DC · {K} hayır kurumu · {S} senaryo "
        f"· Hizmet ≥%{MIN_SERVICE_RATE*100:.0f}",
        fontsize=13, fontweight="bold", y=1.01
    )

    out_path = os.path.join(DATA_DIR, "comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nGrafik kaydedildi: {out_path}", flush=True)
    plt.show()

# =============================================================================
# 10. CSV KAYDET
# =============================================================================

def save_results(eps_pareto, aug_pareto, aug2_pareto):
    rows = []
    for method, pareto in [("epsilon", eps_pareto),
                            ("augmecon", aug_pareto),
                            ("augmecon2", aug2_pareto)]:
        for r in pareto:
            rows.append({
                "method":   method,
                "Z1":       r["Z1"],
                "Z2":       r["Z2"],
                "Z3":       r["Z3"],
                "open_DCs": str([dcs[j] for j in range(J) if r["z"][j] > 0.5]),
            })
    csv_path = os.path.join(DATA_DIR, "comparison_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Sonuçlar kaydedildi: {csv_path}", flush=True)

# =============================================================================
# 11. MAIN
# =============================================================================

if __name__ == "__main__":
    N1 = 15
    N3 = 20

    print("="*60, flush=True)
    print(" Yöntem Karşılaştırması: ε-Constraint | AUGMECON | AUGMECON2",
          flush=True)
    print(f" Ana amaç: max Z2 | Kısıtlar: Z1 <= e1, Z3 <= e3", flush=True)
    print(f" Grid: outer=e3({N3}) × inner=e1({N1}) = {N1*N3} nokta", flush=True)
    print(f" Hizmet seviyesi kısıtı: ≥%{MIN_SERVICE_RATE*100:.0f}", flush=True)
    print(f" Tahmini toplam süre: ~10 dakika", flush=True)
    print("="*60, flush=True)

    t_start = time.time()

    payoff = compute_payoff()
    if payoff is None:
        print("Payoff hesaplanamadı. MIN_SERVICE_RATE'i düşürün.", flush=True)
        exit(1)

    eps_pareto,  eps_models,  eps_time  = run_epsilon(payoff,  n1=N1, n3=N3)
    aug_pareto,  aug_models,  aug_time  = run_augmecon(payoff, n1=N1, n3=N3)
    aug2_pareto, aug2_models, aug2_time = run_augmecon2(payoff, n1=N1, n3=N3)

    print_comparison(payoff,
                     eps_pareto,  eps_models,  eps_time,
                     aug_pareto,  aug_models,  aug_time,
                     aug2_pareto, aug2_models, aug2_time)

    plot_comparison(eps_pareto, aug_pareto, aug2_pareto)
    save_results(eps_pareto, aug_pareto, aug2_pareto)

    total_time = time.time() - t_start
    print(f"\nTamamlandı. Toplam süre: {fmt_eta(total_time)}", flush=True)
    print("  → comparison.png", flush=True)
    print("  → comparison_results.csv", flush=True)