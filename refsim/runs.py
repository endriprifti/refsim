"""Run modes of run.py: packages, grid, maps, types, drift, signatures, pipeline, convergence, reproduce."""
from __future__ import annotations

import datetime as dt
import math
import time

import numpy as np

from . import campaign, opinion, pipeline, regime_b, report, signatures, window
from .drift import ig_band, lock_table, municipal_drift, pr_ever_falls
from .electorate import Electorate, national, rho_path, adult_decline_from_instat
from .evaluate import cell, evaluate_package, load_yaml, local_units, package_uncertainty
from .game import Game, pi_star, rho_lock, tau_lock
from .priors import Prior, uniform
from .rules import V1, V2, V3, V4, V6, V7, make_rule, menu
from .simulate import convergence

PACKAGES = ["P0", "P1", "P1b", "P2", "P3", "P3a", "P4", "P5", "P5c"]


def _log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------------ packages
def run_packages(P, scen, n_draws=60):
    rows = []
    for pk in PACKAGES:
        rows += evaluate_package(P, pk)
    report.write_csv(rows, "packages_central", P, scen,
                     note="Central cell: tau = 0.30 of E_full, t_min = 0.20, pi = 0, uniform prior, sigma_tau = 0.04")
    unc, draws = [], []
    for pk in PACKAGES:
        _log(f"uncertainty draws for {pk}")
        su, dr = package_uncertainty(P, pk, n_draws=n_draws, return_draws=True)
        unc += su
        draws += dr
    report.write_csv(unc, "packages_uncertainty", P, scen,
                     note="Mean, 2.5-97.5% interval, min and max over R ~ U[1.89,1.95]M, tau ~ U[0.20,0.50], t_min ~ {0,0.2,0.35}, pi ~ U[0,0.3]; common draws for every package")
    report.write_csv(draws, "packages_draws", P, scen, note="Every parameter draw of the uncertainty run (common across packages)")
    paired = []
    base = {(d["type"], d["draw"]): d for d in draws if d["package"] == "P0"}
    for pk in PACKAGES[1:]:
        for T in ("T1", "T3", "T4", "T5"):
            for k in ("P_change", "C1", "C1p", "C1R", "C4_friendly", "C4_opposed"):
                diffs = [d[k] - base[(T, d["draw"])][k] for d in draws
                         if d["package"] == pk and d["type"] == T and (T, d["draw"]) in base
                         and d[k] is not None and base[(T, d["draw"])][k] is not None]
                if diffs:
                    paired.append(dict(package=pk, type=T, criterion=k, n=len(diffs), mean_diff_vs_P0=float(np.mean(diffs)),
                                       lo95=float(np.percentile(diffs, 2.5)), hi95=float(np.percentile(diffs, 97.5)),
                                       min=float(np.min(diffs)), max=float(np.max(diffs)),
                                       share_positive=float(np.mean(np.array(diffs) > 1e-9)),
                                       share_negative=float(np.mean(np.array(diffs) < -1e-9))))
    report.write_csv(paired, "packages_paired", P, scen, note="Package minus P0 on common draws (paired differences)")
    run_local_units(P, scen)
    fig_packages(rows, scen)
    return rows



def run_local_units(P, scen):
    """T7 and T8 under each package's local rule on the active municipal map (v1.1.5: split out of run_packages)."""
    pk_cfg = load_yaml("packages.yaml")
    loc = []
    for pk in PACKAGES:
        for T, level in (("T7", "municipality"), ("T8", "qark")):
            cfg = pk_cfg[pk]["rules"][T]
            us = local_units(P, cfg, level=level)
            for u in us:
                u["package"] = pk
                u["type"] = T
            loc += us
    report.write_csv(loc, "local_units", P, scen, note="T7 municipalities and T8 qarks, unit denominators (D4), R_hat rebased to R = 1.92M (D11); fixed resident turnout t0 = 0.580 (national tau = 0.30), t_min = 0.20")
    summ = []
    for pk in PACKAGES:
        for T in ("T7", "T8"):
            us = [u for u in loc if u["package"] == pk and u["type"] == T and u.get("feasible")]
            if not us:
                continue
            pc = np.array([u["P_change"] for u in us])
            summ.append(dict(package=pk, type=T, units=len(us), units_change_possible=int(sum(1 for u in us if not u["regime"].startswith("lock"))),
                             units_locked=int(sum(1 for u in us if u["regime"].startswith("lock"))),
                             median_P_change=float(np.median(pc)), median_C1=float(np.median([u["C1"] for u in us])),
                             remark1_exceptions=",".join(u["unit"] for u in us if u["rho_u"] < 1.5)))
    report.write_csv(summ, "local_summary", P, scen)
    return loc, summ

def fig_packages(rows, scen):
    import matplotlib.pyplot as plt
    pk = PACKAGES
    T1 = {r["package"]: r for r in rows if r["type"] == "T1"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(pk))[::-1]
    for j, (key, lab) in enumerate((("P_change", "Pr[change]"), ("C1", "C1 (majority of voters overridden)"),
                                    ("C4_friendly", "C4 (valid result, friendly majority)"))):
        vals = [T1[p][key] if T1[p].get("feasible") else np.nan for p in pk]
        ax.scatter(vals, y + (j - 1) * 0.22, s=36, color=report.SLOT[j], label=lab, zorder=3,
                   edgecolor=report.SURFACE, linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(pk)
    ax.set_xlim(-0.02, 1.02)
    report.style(ax, "Figure 6.3 — Packages for T1 (abrogative initiative), central cell",
                 "probability", None)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    report.save(fig, "fig6_3_packages_T1", scen)


# ------------------------------------------------------------------ menu grid
def run_grid(P, scen, taus=None, tmins=None, pis=(0.0, 0.3), dens=("E_full", "E_dom")):
    taus = taus or P.behaviour.tau_grid
    tmins = tmins or P.behaviour.t_min_grid
    rows = []
    t0 = time.time()
    for den in dens:
        el = national(P, den)
        for r in menu():
            for tau in taus:
                for tm in tmins:
                    for pi in pis:
                        c = cell(r, el, tau, tm, pi, 0.5, "uniform", 1.0, P.behaviour.sigma_tau)
                        if c is None:
                            continue
                        rows.append({k: c[k] for k in ("rule", "code", "denominator", "tau", "t_min", "pi", "regime", "sx", "sd",
                                                       "mu_hat_eq", "P_change", "P_advisory", "P_dec", "P_void", "C1", "C1p",
                                                       "C1R", "C2", "turnout_E", "rho", "exploitability")})
    _log(f"menu grid: {len(rows)} cells in {time.time() - t0:.0f} s")
    report.write_csv(rows, "menu_grid", P, scen, note="Every menu rule over D3 x D9 x pi x denominator; criteria with A11 noise")
    fig_frontier(rows, scen)
    return rows


def short_label(desc: str) -> str:
    import re
    code = desc.split(" ")[0]
    if code == "V11":
        return "V11"
    nums = re.findall(r"(qS|q|qa|qb|qT|k_U)=([0-9.]+)", desc)
    keep = [f"{v[:4].rstrip('.')}" for k, v in nums if k in ("q", "qS", "qT")][:1]
    return code + (" " + keep[0] if keep else "")


def fig_frontier(rows, scen):
    import matplotlib.pyplot as plt
    sel = [r for r in rows if r["denominator"] == "E_full" and abs(r["tau"] - 0.40) < 1e-9 and r["t_min"] == 0.20 and r["pi"] == 0.0]
    groups = {}
    for r in sel:
        key = (round(r["C1"], 3), round(r["C1p"], 3))
        g = groups.setdefault(key, dict(labels=[], lock=True))
        g["labels"].append(short_label(r["rule"]))
        g["lock"] = g["lock"] and r["regime"].startswith("lock")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    flip = 0
    for (x, y), g in sorted(groups.items()):
        ax.scatter(x, y, s=42, color=report.GRAY if g["lock"] else report.SLOT[0], zorder=3,
                   edgecolor=report.SURFACE, linewidth=1.5)
        if y < 0.02:                       # crowded baseline: alternate above / below, angled
            off = (4, 6) if flip % 2 == 0 else (4, -14)
            flip += 1
            ax.annotate("\n".join(sorted(set(g["labels"]))), (x, y), fontsize=7, color=report.INK2,
                        xytext=off, textcoords="offset points", va="bottom" if off[1] > 0 else "top")
        else:
            ax.annotate(" · ".join(sorted(set(g["labels"]))), (x, y), fontsize=7, color=report.INK2,
                        xytext=(5, 3), textcoords="offset points")
    ax.set_ylim(-0.035, None)
    report.style(ax, "Figure 6.6 — The menu on (C1, C1′): E_full, τ = 0.40, t_min = 0.20, π = 0",
                 "C1  (voters' majority overridden)", "C1′  (change against a resident majority)")
    ax.set_xlim(-0.03, 0.75)
    ax.text(0.99, 0.98, "gray = lock (nobody campaigns)", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=report.INK2)
    report.save(fig, "fig6_6_menu_frontier", scen)


# ------------------------------------------------------------------ regime maps
def run_maps(P, scen):
    rows = []
    rhos = np.round(np.arange(1.0, 2.2001, 0.05), 3)
    taus = np.round(np.arange(0.20, 0.5001, 0.01), 3)
    for rho in rhos:
        el = Electorate(E=rho, R=1.0, label=f"rho={rho}")
        for tau in taus:
            for tm in (0.0, 0.20, 0.35):
                g = Game(V3(), el, tau=tau, t_min=tm)
                if not g.feasible:
                    continue
                eq = g.solve()
                o = g.outcome(eq.sx, eq.sd) if eq.pure else None
                rows.append(dict(rule="V3 q=1/3", rho=rho, tau=tau, t_min=tm, regime=g.regime(eq),
                                 sx=eq.sx, sd=eq.sd, P_change=float(o["P_change"]) if o else np.nan,
                                 mu_hat=float(o["mu_hat"]) if o else np.nan, Q_ge_half=rho / 3 >= 0.5))
    report.write_csv(rows, "regime_map_V3", P, scen, note="Regime A over (rho, tau) at pi = 0, uniform prior, noiseless; t_min in D9")
    fig_regime_map(rows, scen)
    # Pareto map over a small menu
    par = []
    cand = [V1(), V3(q=1 / 3), V6(q=1 / 3, variant="d"), V7(qS=0.6), V2(q=0.40)]
    for rho in np.round(np.arange(1.0, 2.2001, 0.1), 2):
        el = Electorate(E=rho, R=1.0)
        for tau in np.round(np.arange(0.20, 0.5001, 0.05), 2):
            crit = {}
            for r in cand:
                g = Game(r, el, tau=tau, t_min=0.20)
                if not g.feasible:
                    continue
                c = g.criteria(g.solve())
                crit[r.describe()] = (c["C1"], c["C1p"], c["C2"], c["P_change"])
            names = list(crit)
            pareto = []
            for a in names:
                dom = any(all(crit[b][k] <= crit[a][k] + 1e-9 for k in range(3)) and any(crit[b][k] < crit[a][k] - 1e-9 for k in range(3))
                          for b in names if b != a)
                if not dom:
                    pareto.append(a.split(" ")[0])
            par.append(dict(rho=rho, tau=tau, t_min=0.20, pareto_set=" ".join(pareto),
                            **{f"C1[{n.split(' ')[0]}]": crit[n][0] for n in names},
                            **{f"C1p[{n.split(' ')[0]}]": crit[n][1] for n in names}))
    report.write_csv(par, "regime_map_pareto", P, scen, note="Pareto set on (C1, C1', C2) among V1, V3, V6d, V7(0.6), V2(0.40); t_min = 0.20, pi = 0")
    run_prior_maps(P, scen)
    run_drift_maps(P, scen)
    run_campaign_maps(P, scen)
    run_opinion_window_tables(P, scen)


def fig_regime_map(rows, scen):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    sel = [r for r in rows if r["t_min"] == 0.20]
    rhos = sorted(set(r["rho"] for r in sel))
    taus = sorted(set(r["tau"] for r in sel))
    code = {"lock (mutual abstention)": 0, "default demobilises (capped)": 1, "default demobilises": 2,
            "slack (V1 outcome)": 3, "contested": 3}
    Z = np.full((len(taus), len(rhos)), np.nan)
    for r in sel:
        k = code.get(r["regime"], 4)
        Z[taus.index(r["tau"]), rhos.index(r["rho"])] = k
    cmap = ListedColormap([report.GRAY, report.SLOT[0], "#86b6ef", report.SLOT[2], report.SLOT[1]])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.pcolormesh(rhos, taus, Z, cmap=cmap, vmin=-0.5, vmax=4.5, shading="nearest", edgecolors=report.SURFACE, linewidth=0.3)
    for rho_mark, lab in ((3713761 / 1.92e6, "E_full 2025"), (3467826 / 1.92e6, "E_dom 2025")):
        ax.axvline(rho_mark, color=report.INK, linewidth=0.8, linestyle="--")
        ax.text(rho_mark - 0.008, 0.215, lab, rotation=90, fontsize=7, color=report.INK, ha="right", va="bottom")
    report.style(ax, "Figure 6.1 — Regime A (V3, one third of E) by register ratio and turnout, t_min = 0.20",
                 "ρ = E/R", "stand-alone turnout τ (share of E)")
    ax.grid(False)
    import matplotlib.patches as mpatches
    present = set(int(v) for v in Z[~np.isnan(Z)].ravel())
    cats = [(0, report.GRAY, "lock: nobody campaigns, default certain"),
            (1, report.SLOT[0], "default demobilises, challenger at full reach"),
            (2, "#86b6ef", "default demobilises, challenger interior"),
            (3, report.SLOT[2], "slack or contested (V1-like)"),
            (4, report.SLOT[1], "no pure equilibrium")]
    hs = [mpatches.Patch(color=c, label=l) for k, c, l in cats if k in present]
    hs.append(mpatches.Patch(facecolor=report.SURFACE, edgecolor=report.GRAY, label="outside the model (t0 ≤ t_min or t0 > 1)"))
    ax.legend(handles=hs, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    report.save(fig, "fig6_1_regime_map_V3", scen)


def run_prior_maps(P, scen):
    import matplotlib.pyplot as plt
    rows = []
    aB = P.behaviour.a_B_grid
    for den, rho in (("E_full", 3713761 / 1.92e6), ("E_dom", 3467826 / 1.92e6)):
        el = Electorate(E=rho, R=1.0, label=den)
        for tm in (0.0, 0.20, 0.35):
            for a in aB:
                pr = Prior(a, a)
                tl = tau_lock(V3(), el, tm, prior=pr, recalibrate=True)
                tf = tau_lock(V3(), el, tm, prior=pr, recalibrate=False)
                t5 = regime_b.tau_threshold_solver(pr, rho, tm)
                rows.append(dict(denominator=den, rho=rho, t_min=tm, a_B=a, tau_lock_recal=tl if tl is not None and tl <= 0.5 else "locked in D3",
                                 tau_lock_fixed=tf if tf is not None and tf <= 0.5 else "locked in D3",
                                 T5_opposition_campaigns_above=t5 if t5 is not None and t5 <= 0.5 else "never in D3"))
    report.write_csv(rows, "prior_maps", P, scen, note="Lock boundary of Regime A and the T5 opposition threshold (Proposition 30) across Beta(a_B, a_B) priors")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for j, tm in enumerate((0.0, 0.20, 0.35)):
        xs, ys = [], []
        for r in rows:
            if r["denominator"] == "E_full" and r["t_min"] == tm and not isinstance(r["tau_lock_recal"], str):
                xs.append(r["a_B"]); ys.append(r["tau_lock_recal"])
        ax.plot(xs, ys, color=report.SLOT[j], linewidth=2, marker="o", markersize=5)
        if xs:
            ax.annotate(f"t_min = {tm:g}", (xs[-1], ys[-1]), fontsize=8, color=report.INK2, xytext=(5, 0), textcoords="offset points")
    ax.axhline(0.5, color=report.GRAY, linewidth=1)
    ax.text(0.52, 0.505, "top of D3: above this line every τ is locked", fontsize=7, color=report.INK2)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2.5, 4, 8, 16]); ax.set_xticklabels(["0.5", "1", "2.5", "4", "8", "16"])
    report.style(ax, "Figure 6.2 — Lock boundary of Regime A by prior shape (E_full, κ recalibrated)",
                 "a_B (Beta(a_B, a_B); 1 = uniform A2; < 1 U-shaped)", "τ_lock (share of E)")
    report.save(fig, "fig6_2_lock_boundary_by_prior", scen)
    # Regime B (Proposition 30): finer grid of U-shaped priors
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    grid_a = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    fine = []
    for j, (den, rho) in enumerate((("E_dom", 3467826 / 1.92e6), ("E_full", 3713761 / 1.92e6))):
        xs, ys = [], []
        for a in grid_a:
            t5 = regime_b.tau_threshold_solver(Prior(a, a), rho, 0.20)
            fine.append(dict(denominator=den, a_B=a, t_min=0.20, T5_opposition_campaigns_above=t5 if t5 is not None and t5 <= 0.5 else "never in D3"))
            if t5 is not None and t5 <= 0.5:
                xs.append(a); ys.append(t5)
        ax.plot(xs, ys, color=report.SLOT[j], linewidth=2, marker="o", markersize=5)
        if xs:
            ax.annotate(f"{den}: campaigns above this turnout", (xs[-1], ys[-1]), fontsize=8, color=report.INK2,
                        xytext=(6, 0), textcoords="offset points", va="center")
    report.write_csv(fine, "regime_B_U_shaped", P, scen, note="Proposition 30(ii): T5 opposition threshold under U-shaped Beta priors, t_min = 0.20")
    ax.axhline(0.5, color=report.GRAY, linewidth=1)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.3, 0.52)
    report.style(ax, "Figure 6.5 — T5: turnout above which the opposition campaigns (t_min = 0.20)",
                 "a_B (Beta(a_B, a_B); below 1 = U-shaped; at 1 and above: never within D3, Proposition 30)", "τ (share of E)")
    report.save(fig, "fig6_5_regime_B_priors", scen)


def run_drift_maps(P, scen):
    import matplotlib.pyplot as plt
    d = P.drift
    E0, R0 = P.electorate.E_full, P.electorate.R
    rows = []
    for dR in (0.004, 0.0085, 0.0116):
        for r in lock_table(E0, R0, d.gamma_E, dR):
            rows.append(dict(d_R=dR, **r))
    report.write_csv(rows, "drift_lock_years", P, scen, note="A29 deterministic path from April 2025 (E_full); d_R 0.0085 central, 0.0116 fast (D11), 0.004 slow")
    md = []
    for g, lab in ((E0 and (d.gamma_E / E0 + d.d_R_central), "central"), (d.gamma_E / E0 + d.d_R_fast, "fast")):
        for rebase in (True, False):
            m = municipal_drift(g, rebase=rebase)
            md.append(dict(case=lab, gamma=g, basis=m["basis"], **{f"count_y{y}": c for y, c in m["counts"].items()},
                           joiners="; ".join(f"{n} {t:.2f}" for t, n in m["joiners"]),
                           remark1_exits="; ".join(f"{n} {t:.2f}" for t, n in m["remark1_exits"])))
    report.write_csv(md, "drift_municipal", P, scen, note="Municipalities with q_A rho_u > 1 by years from 2025; rebased (D11) and as printed")
    ig = []
    rho_dom = (E0 - P.electorate.E_ocv) / R0
    for sig in (0.01, 0.02):
        for gam, lab in ((0.017, "central"), (0.020, "fast")):
            ig.append(dict(sigma_lnrho=sig, gamma=gam, case=lab,
                           Pr_rho_dom_falls_to_1_7803=pr_ever_falls(rho_dom, 1.7803, gam, sig),
                           IG_5_95_years_to_2_0158=str(ig_band(3713761 / 1.92e6, 2.0158, gam, sig)),
                           IG_5_95_years_to_2_4182=str(ig_band(3713761 / 1.92e6, 2.4182, gam, sig))))
    report.write_csv(ig, "drift_ig", P, scen)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    t = np.linspace(0, 20, 201)
    for j, (dR, lab) in enumerate(((0.0085, "adult rate (central, D11)"), (0.0116, "all-resident rate (fast)"))):
        y = rho_path(t, E0, R0, d.gamma_E, dR)
        ax.plot(2025.25 + t, y, color=report.SLOT[j], linewidth=2)
        ax.annotate(lab, (2045.25, y[-1]), fontsize=8, color=report.INK2, xytext=(4, 0), textcoords="offset points", va="center")
    for t0, rl, xl in ((0.60, 2.0158, 2036.0), (0.75, 2.4182, 2025.5), (0.90, 2.619, 2025.5)):
        ax.axhline(rl, color=report.GRAY, linewidth=1, linestyle="--")
        ax.text(xl, rl + 0.012, f"lock at resident turnout {t0:.2f}", fontsize=7, color=report.INK2)
    ax.set_xticks(range(2025, 2046, 5))
    report.style(ax, "Figure 6.4 — Register drift against the Regime A lock (E_full, t_min = 0.20)", "year", "ρ = E/R")
    report.save(fig, "fig6_4_drift", scen)


def run_campaign_maps(P, scen):
    rows = []
    for den, Q in (("E_full", (3713761 / 1.92e6) / 3), ("E_dom", (3467826 / 1.92e6) / 3)):
        for L in (30, 45, 60, 90):
            for beta in (0.5, 1.0, 2.0):
                for pi in (0.0, 0.3):
                    for tm in (0.0, 0.20, 0.35):
                        rows.append(dict(denominator=den, L_days=L, beta_I=beta, pi=pi, t_min=tm,
                                         k_min_infeasible=campaign.k_min(Q, pi, tm, beta, L=L),
                                         A_I_k1=campaign.A_I(L, 1, beta), A_I_k10=campaign.A_I(L, 10, beta),
                                         L_star_k10_80pct=campaign.L_star(10, beta, 0.8)))
    report.write_csv(rows, "campaign_L_k", P, scen, note="Corollary 7 sensitivity: question complexity k at which change becomes infeasible (illustrative calibration)")


def run_opinion_window_tables(P, scen):
    rows = []
    for T_rel in (1, 2, 3, 4, 5):
        for q in (0.10, 0.25, 1 / math.e, 0.5):
            rows.append(dict(T_rel_years=T_rel, q_chi=q, fidelity_horizon_years=opinion.fidelity_horizon(T_rel, q),
                             bar_exceeds=opinion.bar_exceeds_fidelity(3, T_rel, q)))
    report.write_csv(rows, "opinion_bar", P, scen, note="Proposition 25: the three-year bar against the fidelity horizon")
    w = []
    for T_pre in (0.5, 1.0, 1.5, 2.0):
        for T_post in (1.0, 2.12, 3.0, 4.0):
            r = dict(T_pre=T_pre, T_post=T_post)
            for cyc in (2021, 2025):
                c = window.window_complies(T_pre, T_post, window.D_PRE[cyc], window.D_POST[cyc])
                r[f"pre_ok_{cyc}"] = c["pre"]
                r[f"post_ok_{cyc}"] = c["post"]
            r["anchored_window_blocked_share"] = window.anchored_window_blocked_share(T_pre, T_post)
            w.append(r)
    report.write_csv(w, "window_compliance", P, scen, note="Proposition 28 at q_w = 0.10 for the 2021 and 2025 cycles")
    tv = [dict(T_post=T, d_el=d, pi_star=p) for T in (1.0, 2.2, 4.0) for d, p in window.pi_star_at_distance(3713761 / 1.92e6, T_post=T)]
    report.write_csv(tv, "window_lock_by_distance", P, scen, note="Table V.24 construction (A5): kappa at the stand-alone calibration")


# ------------------------------------------------------------------ other modes
def run_types(P, scen):
    rows = []
    tys = load_yaml("types.yaml")
    for T in ("T1", "T2", "T3", "T4", "T5", "T6"):
        el = national(P, "E_full")
        rcfg = dict(tys[T]["rule_in_force"])
        for tau in P.behaviour.tau_grid:
            for tm in P.behaviour.t_min_grid:
                c = cell(make_rule(rcfg), el, tau, tm, 0.0, tys[T].get("agenda_mu_b", 0.5), "uniform", 1.0, P.behaviour.sigma_tau)
                if c:
                    rows.append(dict(type=T, tau=tau, t_min=tm, rule=c["rule"], regime=c["regime"], P_change=c["P_change"],
                                     P_dec=c["P_dec"], C1=c["C1"], C1p=c["C1p"], C2=c["C2"], mu_hat=c["mu_hat_eq"]))
    report.write_csv(rows, "types_in_force", P, scen, note="Rules in force by type over D3 x D9 (pi = 0); T6 under the assumed rule (D5)")
    return rows


def run_signatures(P, scen):
    rows = signatures.error_table()
    for v in (0.6, 0.7, 0.85, 0.95):
        rows.append(dict(unit="bound", S=50000, v_L=v, margin="n_min (4.20)", R3_lines=signatures.n_min(v)))
    report.write_csv(rows, "signatures_C4prime", P, scen, note="False admission (C4') and false rejection: R0 vs the F44 band rule; exact hypergeometric")


def run_pipeline(P, scen):
    rows = []
    for opt in ("O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9"):
        for pres in ("earliest", "latest"):
            for rule in ("R0", "R3"):
                FR = pipeline.kqz_false_rejection(rule)
                rows.append(dict(verification=rule, FR=FR, **pipeline.pr_vote(opt, pres, FR=FR)))
    report.write_csv(rows, "pipeline_Pr_vote", P, scen, note="Exact enumeration (B.43), regular terms, valid national initiative, uniform filing week")


def run_convergence(P, scen, n=20000):
    rows = []
    el = national(P, "E_full")
    for r in (V1(), V2(q=0.40), V3(), V6(q=1 / 3, variant="d"), V7(qS=0.6)):
        for tau in (0.30, 0.40, 0.50):
            for tm in (0.0, 0.20):
                g = Game(r, el, tau=tau, t_min=tm)
                if not g.feasible:
                    continue
                eq = g.solve()
                cv = convergence(g, eq, n=n, sigma_tau=P.behaviour.sigma_tau)
                rows.append(dict(rule=r.describe(), tau=tau, t_min=tm, n=n, max_z=cv["max_z"], max_half_diff=cv["max_half_diff"],
                                 P_change_exact=cv["exact"]["P_change"], P_change_mc=cv["mc"]["P_change"],
                                 P_change_lo=cv["mc"]["P_change_lo"], P_change_hi=cv["mc"]["P_change_hi"]))
    report.write_csv(rows, "convergence", P, scen, note="Monte Carlo (ex-post rule evaluation on vote counts) vs exact quadrature; max |diff| in MC standard errors")
    return rows


def run_reproduce(P, scen):
    """Key values of Sections III-V recomputed by the simulator (the regression suite asserts them)."""
    rf, rd = 3713761 / 1.92e6, 3467826 / 1.92e6
    el = Electorate(E=1.93, R=1.0)
    rows = []
    add = lambda what, book, val: rows.append(dict(item=what, book=book, simulator=val))
    for tm, b in ((0, 0.294), (0.2, 0.351), (0.35, 0.389)):
        add(f"pi* tau=0.40 rho=1.93 t_min={tm} (Table III.3)", b, round(pi_star(V3(), el, 0.40, tm), 3))
    for tm, b in ((0, 0.306), (0.2, 0.297), (0.35, 0.297)):
        add(f"tau_lock rho=1.93 t_min={tm} (Prop. 18, Cor. 1)", b, round(tau_lock(V3(), el, tm), 3))
    add("rho_lock t0=0.60 (Table V.25)", 2.016, round(rho_lock(0.60, 0.2), 3))
    add("rho_lock t0=0.75 (Table V.25)", 2.418, round(rho_lock(0.75, 0.2), 3))
    g = Game(V3(), Electorate(E=1.93, R=1.0), tau=0.40, t_min=0.2)
    c = g.criteria(sigma_tau=0.04)
    add("F38 Pr[pass | mu>1/2] E_full sigma_tau=0.04", 0.669, round((c["P_change"] - c["C1p"]) / 0.5, 3))
    for a, b in ((0.5, 0.348),):
        add(f"T5 opposition threshold E_dom a_B={a} t_min=0 (CC2 R9)", b, round(regime_b.tau_threshold_solver(Prior(a, a), rd, 0.0), 3))
    pv = pipeline.pr_vote("O0", "earliest")
    add("Pr_vote(12) TS0 prompt (Table V.5)", 0.061, round(pv["Pr_vote_12"], 3))
    add("Pr_vote(24) TS0 prompt", 0.995, round(pv["Pr_vote_24"], 3))
    add("mean months TS0 prompt", 16.43, round(pv["mean_months"], 2))
    add("R3 worst-case error (F44)", 0.0091, round(signatures.worst_case("R3", 50000, 0.6, W=60)[0], 4))
    add("two-point bound t_L=0.38 (5.34)", 2.12, round(window.two_point_bound(0.38), 2))
    add("Table V.24 d=0 (all T)", 0.53, round(window.pi_star_at_distance(rf, T_post=2.2)[0][1], 2))
    dec = adult_decline_from_instat()
    add("INSTAT 15+ decline 2025-26 (D11)", 0.0083, round(dec["d_15plus"], 4))
    report.write_csv(rows, "reproduce", P, scen, note="Simulator against printed values of Sections III-V and Consistency check 2")
    return rows


# ------------------------------------------------------------------ sensitivity (v1.0.1; T8 literal reading v1.0.2)
def run_literal_reading(P, scen):
    """D4 boundary case: Art. 118(3) read literally (one third of the national register) for T7 and T8."""
    from .electorate import load_municipalities
    m = load_municipalities()
    lit = []
    for i, name in enumerate(m.names):
        Ru = m.R_hat(1.92e6)[i]
        Q = (1 / 3) * P.electorate.E_full / Ru
        lit.append(dict(unit=name, R_u=Ru, Q_literal=Q, feasible_at_full_turnout=Q <= 1.0))
    report.write_csv(lit, "T7_literal_national_register", P, scen, note="D4 boundary case: Art. 118(3) read literally (one third of the national register) applied to a municipal vote")
    litq = []
    for name, q in m.qarks(R_now=1.92e6).items():
        Q = (1 / 3) * P.electorate.E_full / q["R"]
        litq.append(dict(unit=name, R_u=q["R"], Q_literal=Q, feasible_at_full_turnout=Q <= 1.0))
    report.write_csv(litq, "T8_literal_national_register", P, scen, note="D4 boundary case: Art. 118(3) read literally (one third of the national register) applied to a qark vote")


def run_sensitivity(P, scen):
    """sigma_tau sweep; noisy Pareto map; w_4(d_rev); the literal national-register reading for T7 (D4)."""
    from .evaluate import t4_feasible_share
    from .electorate import load_municipalities
    rows = []
    el = national(P, "E_full")
    for sig in (0.0, 0.02, 0.04, 0.06, 0.08):
        for r in (V1(), V3(), V6(q=1 / 3, variant="d"), V7(qS=0.6), V2(q=0.40)):
            for tau in (0.30, 0.40):
                c = cell(r, el, tau, 0.20, 0.0, 0.5, "uniform", 1.0, sig)
                rows.append(dict(sigma_tau=sig, rule=r.describe(), tau=tau, regime=c["regime"], P_change=c["P_change"],
                                 P_dec=c["P_dec"], C1=c["C1"], C1p=c["C1p"], C1R=c["C1R"], C2=c["C2"]))
    report.write_csv(rows, "sensitivity_sigma_tau", P, scen, note="Turnout-noise sweep at t_min = 0.20, pi = 0, E_full (strategies from the noiseless game, A31)")
    par = []
    cand = [V1(), V3(q=1 / 3), V6(q=1 / 3, variant="d"), V7(qS=0.6), V2(q=0.40)]
    for rho in np.round(np.arange(1.0, 2.2001, 0.1), 2):
        e = Electorate(E=rho, R=1.0)
        for tau in np.round(np.arange(0.20, 0.5001, 0.05), 2):
            crit = {}
            for r in cand:
                g = Game(r, e, tau=tau, t_min=0.20)
                if not g.feasible:
                    continue
                c = g.criteria(g.solve(), sigma_tau=P.behaviour.sigma_tau)
                crit[r.describe().split(" ")[0]] = (c["C1"], c["C1p"], c["C2"])
            pareto = [a for a in crit if not any(all(crit[b][k] <= crit[a][k] + 1e-9 for k in range(3)) and
                                                  any(crit[b][k] < crit[a][k] - 1e-9 for k in range(3)) for b in crit if b != a)]
            par.append(dict(rho=rho, tau=tau, t_min=0.20, sigma_tau=P.behaviour.sigma_tau, pareto_set=" ".join(pareto),
                            **{f"C1[{n}]": v[0] for n, v in crit.items()}, **{f"C1p[{n}]": v[1] for n, v in crit.items()}))
    report.write_csv(par, "regime_map_pareto_noisy", P, scen, note="Pareto set on (C1, C1', C2) with A11 noise (sigma_tau = 0.04); the noiseless map is regime_map_pareto.csv")
    w4 = [dict(d_rev_days=d, w4_lead_30_days=t4_feasible_share(d, 30), w4_lead_42_days=t4_feasible_share(d, 42)) for d in range(10, 63)]
    report.write_csv(w4, "t4_w4", P, scen, note="Proposition 23: share of adoption weekdays with a feasible polling Sunday within 60 days (weekday arithmetic only; windows and election dates not removed, see F50)")
    run_literal_reading(P, scen)
    dl = []
    for lab, r in (("P0 current law, V3 at 1/3 of E", V3()), ("P3 T1, V2 at 40% of E", V2(q=0.40)), ("P3a T1, V3 at 40% of E", V3(q=0.40)),
                   ("P3 T4, V2 at 50% of E", V2(q=0.50)), ("P3a T4, V3 at 50% of E", V3(q=0.50))):
        for tm in (0.0, 0.20, 0.35):
            t = tau_lock(r, el, tm)
            dl.append(dict(rule=lab, t_min=tm, lock_boundary_tau=t if t is not None else "locked at every tau up to the feasible maximum"))
    report.write_csv(dl, "draft_lock_boundaries", P, scen, note="Lowest stand-alone turnout (share of E_full) at which the challenger campaigns; pi = 0, A2")
