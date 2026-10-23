"""Tool runs TR11-TR19 on the calibrated scenario (Section VII Part 2, rerun request; refsim v1.1).

Every run reads config/calibrated_2026.yaml and writes outputs/calibrated_2026/vii2_TR*.csv (+ .meta.json).
Model changes of v1.1 used here (all off in baseline_2025):
  A36  floor by strategy: a default side under a participation-type rule may call to abstain (t_min_boycott);
  t_bar as a scenario parameter (0.90);
  pi and mu_b by type; Psi by type with mu_G (C3 column); the government's side by type;
  D13  calibrated prior: Beta with the variance of Beta(a_B, a_B) and type mean 1/2 + pi (mu_b - 1/2)
       ('beta_shift'); sensitivities 'beta' (no shift) and 'zealot_vm' (zealot family, a_p(pi) by (7.6)).
"""
from __future__ import annotations

import datetime as dt
import math
import time

import numpy as np

from . import campaign, opinion, report, signatures
from .drift import pr_ever_falls, ig_band
from .electorate import Electorate, national, t_cross, load_municipalities
from .evaluate import evaluate_package, load_yaml, local_units, scenario_settings, type_c4, electorate_for, _rule_from_cfg, cell, qark_weights
from .game import Game, rho_lock, tau_lock
from .priors import Prior, a_p_matched, kappa_calibration, make as make_prior
from .rules import V3, V4, V10, make_rule

PACKAGES = ["P0", "P1", "P1b", "P2", "P3", "P3a", "P3b", "P3c", "P4", "P5", "P5c"]
TYPES = ("T1", "T2", "T3", "T4", "T5", "T6")
KEYS = ("rule", "regime", "sx", "sd", "floor_d", "mu_hat_eq", "P_change", "P_advisory", "P_dec", "C1", "C1p", "C1R", "C2",
        "C3", "C4_opposed", "C4_friendly", "turnout_E", "t0", "pi", "mu_b", "prior", "Psi", "mu_Gx")


def _log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _rows(P, pk, **kw):
    out = []
    for r in evaluate_package(P, pk, **kw):
        out.append({k: r.get(k) for k in ("package", "type", "feasible") + KEYS})
    return out


# ------------------------------------------------------------------ TR11 central cell and one-at-a-time variants
VARIANTS = [
    ("central", {}, {}),
    ("kappa fixed", {}, dict(recalibrate=False)),
    ("prior: beta (no type shift)", {}, dict(family="beta")),
    ("prior: zealot_vm (7.6)", {}, dict(family="zealot_vm")),
    ("A36 off (one floor)", {}, dict(t_min_b=None)),
    ("t_bar = 1.00", {}, dict(t_bar=1.0)),
    ("t_bar = 0.85", {}, dict(t_bar=0.85)),
    ("a_B = 1.5", {}, dict(a_B=1.5)),
    ("a_B = 5.0", {}, dict(a_B=5.0)),
    ("tau = 0.20", dict(tau=0.20), {}),
    ("tau = 0.42", dict(tau=0.42), {}),
    ("t_min = 0.20", dict(t_min=0.20), {}),
    ("t_min = 0.38", dict(t_min=0.38), {}),
    ("sigma_tau = 0.03", dict(sigma_tau=0.03), {}),
    ("sigma_tau = 0.06", dict(sigma_tau=0.06), {}),
]


def run_TR11(P, scen):
    rows = []
    for pk in PACKAGES:
        rows += _rows(P, pk)
    report.write_csv(rows, "vii2_TR11_packages_central", P, scen,
                     note="TR11: packages at the calibrated central cell (tau 0.33 of E_full; t_min 0.30, boycott floor 0.08 (A36); t_bar 0.90; "
                          "Beta-shift a_B = 3 with pi and mu_b by type; sigma_tau 0.04; Psi by type, mu_G 0.47)")
    var = []
    for lab, kw, ov in VARIANTS:
        for pk in PACKAGES:
            for r in _rows(P, pk, overrides=ov, types=("T1", "T3", "T4", "T5"), **kw):
                r["variant"] = lab
                var.append(r)
    report.write_csv(var, "vii2_TR11_variants", P, scen,
                     note="TR11 one-at-a-time variants around the calibrated central cell (types T1, T3, T4, T5)")
    return rows


# ------------------------------------------------------------------ TR12 common draws from the calibrated ranges
UNC = ("P_change", "P_dec", "C1", "C1p", "C1R", "C2", "C3", "C4_opposed", "C4_friendly")


def draw_params(P, n_draws=60, seed=7):
    rng = np.random.default_rng(seed)
    lo_R, hi_R = P.electorate.meta("R")["range"]
    out = []
    for i in range(n_draws):
        d = dict(draw=i, R=rng.uniform(lo_R, hi_R), tau=rng.uniform(0.20, 0.42), t_min=rng.uniform(0.20, 0.38),
                 t_min_b=rng.uniform(0.05, 0.15), t_bar=rng.uniform(0.85, 1.00), a_B=rng.uniform(1.5, 5.0),
                 pi_12=rng.uniform(0.05, 0.50), pi_36=rng.uniform(0.25, 0.85), sigma_tau=rng.uniform(0.03, 0.06),
                 Psi=rng.uniform(0.05, 0.45), mu_G=rng.uniform(0.36, 0.58))
        out.append(d)
    return out


def run_TR12(P, scen, n_draws=60, seed=7, types=("T1", "T3", "T4", "T5")):
    draws = draw_params(P, n_draws, seed)
    allrows = []
    for pk in PACKAGES:
        _log(f"TR12 draws for {pk}")
        for d in draws:
            p = P.override(**{"electorate.R": d["R"]})
            ov = dict(t_min_b=d["t_min_b"], t_bar=d["t_bar"], a_B=d["a_B"], mu_G=d["mu_G"],
                      pi_by_type={"T1": d["pi_12"], "T2": d["pi_12"], "T3": d["pi_36"], "T4": d["pi_36"], "T5": d["pi_36"], "T6": d["pi_36"]},
                      Psi_by_type={T: d["Psi"] for T in TYPES})
            if p.electorate.E_full / d["R"] * d["tau"] > d["t_bar"]:
                continue
            for r in evaluate_package(p, pk, tau=d["tau"], t_min=d["t_min"], sigma_tau=d["sigma_tau"], types=types, overrides=ov):
                if not r.get("feasible"):
                    continue
                row = dict(package=pk, type=r["type"], regime=r["regime"], **{k: v for k, v in d.items()})
                for k in UNC:
                    v = r.get(k)
                    row[k] = v if (v is not None and not (isinstance(v, float) and math.isnan(v)) and v < 2) else None
                allrows.append(row)
    report.write_csv(allrows, "vii2_TR12_draws", P, scen,
                     note=f"TR12: {n_draws} common draws (seed {seed}), uniform in each calibrated range: R, tau [0.20,0.42], t_min [0.20,0.38], "
                          "t_min_boycott [0.05,0.15], t_bar [0.85,1], a_B [1.5,5], pi T1-T2 [0.05,0.50], pi T3-T6 [0.25,0.85], sigma_tau [0.03,0.06], "
                          "Psi [0.05,0.45], mu_G [0.36,0.58]")
    summ = []
    for pk in PACKAGES:
        for T in types:
            rr = [r for r in allrows if r["package"] == pk and r["type"] == T]
            row = dict(package=pk, type=T, n=len(rr), n_locked=sum(1 for r in rr if str(r["regime"]).startswith("lock")))
            for k in UNC:
                v = [r[k] for r in rr if r[k] is not None]
                if v:
                    row.update({f"{k}_mean": float(np.mean(v)), f"{k}_min": float(np.min(v)), f"{k}_max": float(np.max(v)),
                                f"{k}_lo95": float(np.percentile(v, 2.5)), f"{k}_hi95": float(np.percentile(v, 97.5))})
            summ.append(row)
    report.write_csv(summ, "vii2_TR12_summary", P, scen, note="TR12 summary over the common draws")
    base = {(r["type"], r["draw"]): r for r in allrows if r["package"] == "P0"}
    paired = []
    for pk in PACKAGES[1:]:
        for T in types:
            for k in ("P_change", "C1", "C1p", "C1R", "C2", "C4_opposed"):
                dd = [r[k] - base[(T, r["draw"])][k] for r in allrows if r["package"] == pk and r["type"] == T
                      and (T, r["draw"]) in base and r[k] is not None and base[(T, r["draw"])][k] is not None]
                if dd:
                    paired.append(dict(package=pk, type=T, criterion=k, n=len(dd), mean=float(np.mean(dd)), min=float(np.min(dd)),
                                       max=float(np.max(dd)), share_pos=float(np.mean(np.array(dd) > 1e-9)),
                                       share_neg=float(np.mean(np.array(dd) < -1e-9))))
    report.write_csv(paired, "vii2_TR12_paired", P, scen, note="TR12 package minus P0 on the common draws")
    return summ


# ------------------------------------------------------------------ TR13 Regime A lock boundary
def _lock(rule, el, t_min, prior, recal, pi, t_bar, t_min_b=None):
    t = tau_lock(rule, el, t_min, prior=prior, recalibrate=recal, pi=pi, t_bar=t_bar, t_min_b=t_min_b)
    tmax = t_bar / el.rho
    if t is None:
        return "locked at every tau up to t_bar/rho", tmax
    lo = (t_min + el.n_ab) / el.rho + 1e-6
    if t <= max(0.05, lo) + 1e-6:
        return "campaigns at every feasible tau", tmax
    return float(t), tmax


def run_TR13(P, scen):
    rows = []
    aBs = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    piT1 = P.behaviour.pi_by_type["T1"]
    for den in ("E_full", "E_dom"):
        el = national(P, den)
        for fam in ("beta", "zealot_vm"):
            for recal in (True, False):
                for tb in (0.85, 0.90, 1.00):
                    for tm in (0.20, 0.30, 0.38):
                        if den == "E_dom" and not (tb == 0.90 and recal):
                            continue
                        for a in aBs:
                            pr = make_prior(fam, pi=piT1, a_B=a, mu_b=0.5)
                            t, tmax = _lock(V3(), el, tm, pr, recal, piT1, tb)
                            rows.append(dict(denominator=den, rho=el.rho, family=fam, kappa="recalibrated" if recal else "fixed",
                                             t_bar=tb, t_min=tm, a_B=a, pi_T1=piT1, prior=pr.label, tau_lock=t, tau_max=tmax,
                                             central_0_33_locked=(isinstance(t, str) and t.startswith("locked")) or (not isinstance(t, str) and t > 0.33)))
    report.write_csv(rows, "vii2_TR13_lock_boundary", P, scen,
                     note="TR13: Regime A (V3 at 1/3 of E) lock boundary in tau for T1 (pi = 0.25, mu_b = 1/2); 'beta' = Beta(a_B, a_B) "
                          "(= Beta-shift at mu_b = 1/2), 'zealot_vm' = zealot family with a_p(pi) by (7.6)")
    s1 = []
    el = national(P, "E_full")
    for fam in ("zealot_vm", "beta"):
        for recal in (True, False):
            for a in (1.5, 3.0, 5.0):
                for pi in (0.05, 0.25, 0.50):
                    try:
                        pr = make_prior(fam, pi=pi, a_B=a, mu_b=0.5)
                    except ValueError as e:
                        s1.append(dict(family=fam, kappa="recalibrated" if recal else "fixed", a_B=a, pi=pi, tau_lock=str(e)))
                        continue
                    t, tmax = _lock(V3(), el, 0.30, pr, recal, pi, 0.90)
                    s1.append(dict(family=fam, kappa="recalibrated" if recal else "fixed", a_B=a, pi=pi, prior=pr.label,
                                   tau_lock=t, tau_max=tmax))
    report.write_csv(s1, "vii2_TR13_aB_x_pi", P, scen,
                     note="Sensitivity 1: a_B x pi(T1), E_full, t_min 0.30, t_bar 0.90 (zealot_vm and beta; both kappa conventions)")
    return rows


# ------------------------------------------------------------------ TR14 T5 opposition thresholds (Regime B)
def run_TR14(P, scen):
    rows = []
    for den in ("E_dom", "E_full"):
        el = national(P, den)
        for tb in (0.90, 1.00):
            for tm in (0.20, 0.30, 0.38):
                for a in (1.5, 2.0, 3.0, 4.0, 5.0):
                    pr = Prior(a, a)
                    for recal in (True, False):
                        t, tmax = _lock(V4(q=0.5), el, tm, pr, recal, 0.0, tb)
                        rows.append(dict(denominator=den, family="beta", a_B=a, pi=0.0, a_p=None, t_bar=tb, t_min=tm,
                                         kappa="recalibrated" if recal else "fixed", T5_campaigns_above=t, tau_max=tmax))
                for pi in (0.30, 0.35, 0.40, 0.50):
                    ap = a_p_matched(3.0, pi)
                    pr = make_prior("zealot_vm", pi=pi, a_B=3.0, mu_b=0.5)
                    for recal in (True, False):
                        t, tmax = _lock(V4(q=0.5), el, tm, pr, recal, pi, tb)
                        rows.append(dict(denominator=den, family="zealot_vm", a_B=3.0, pi=pi, a_p=ap, t_bar=tb, t_min=tm,
                                         kappa="recalibrated" if recal else "fixed", T5_campaigns_above=t, tau_max=tmax))
    report.write_csv(rows, "vii2_TR14_T5_thresholds", P, scen,
                     note="TR14: registered turnout above which the T5 opposition campaigns under V4 (q_R = 1/2); symmetric Beta and zealot_vm at a_obs = 3")
    return rows


# ------------------------------------------------------------------ TR15 complexity bound with cue-taking
def run_TR15(P, scen):
    rows = []
    Q_full = P.electorate.E_full / P.electorate.R / 3
    for r_bar in (0.03, 0.07, 0.30):
        for beta in (0.2, 0.5, 1.0):
            for L in (30, 45, 60):
                for gc in (0.0, 0.4, 0.6, 0.8):
                    for tb in (0.90, 1.00):
                        for mu_hi, lab in ((1.0, "Beta prior (support to 1)"), (1 - 0.25 / 2, "A2 at pi = 0.25")):
                            tm = 0.30
                            kbar = None
                            for k in range(1, 100001):
                                A = campaign.A_I(L, k, beta, r_bar=r_bar)
                                A_eff = A + gc * (1 - A)
                                reach = tm + (tb - tm) * A_eff
                                if reach * mu_hi < Q_full:
                                    kbar = k
                                    break
                            A1 = campaign.A_I(L, 1, beta, r_bar=r_bar)
                            A10 = campaign.A_I(L, 10, beta, r_bar=r_bar)
                            Astar = (Q_full / mu_hi - tm) / (tb - tm)
                            need10 = max(0.0, (Astar - A10) / (1 - A10)) if A10 < 1 else 0.0
                            rows.append(dict(r_bar=r_bar, beta_I=beta, L_days=L, gamma_c=gc, t_bar=tb, support=lab, t_min=tm, Q=Q_full,
                                             k_bar_infeasible=kbar if kbar is not None else "never (feasible for every k)",
                                             A_I_k1=A1, A_I_k10=A10, A_needed=Astar, gamma_c_needed_k10=need10,
                                             L_80pct_k1=campaign.L_star(1, beta, 0.8, r_bar=r_bar),
                                             L_80pct_k124=campaign.L_star(124, beta, 0.8, r_bar=r_bar)))
    report.write_csv(rows, "vii2_TR15_complexity", P, scen,
                     note="TR15: smallest complexity k at which change becomes infeasible (reach t_min + (t_bar - t_min) A_eff, "
                          "A_eff = A_I + gamma_c (1 - A_I)); E_full, t_min 0.30; Corollary 7 with the cue-taking branch")
    return rows


# ------------------------------------------------------------------ TR16 contamination
def run_TR16(P, scen):
    rows = []
    for pk in PACKAGES:
        for Psi in (0.05, 0.25, 0.45):
            for mu_G in (0.36, 0.47, 0.58):
                for side in ("government backs change", "government opposes change"):
                    gb = {T: side.startswith("government backs") for T in TYPES}
                    ov = dict(Psi_by_type={T: Psi for T in TYPES}, mu_G=mu_G, gov_backs_change=gb)
                    for r in evaluate_package(P, pk, types=("T1", "T3", "T4"), overrides=ov):
                        if r.get("feasible"):
                            rows.append(dict(package=pk, type=r["type"], Psi=Psi, mu_G=mu_G, side=side, regime=r["regime"],
                                             mu_hat=r["mu_hat_eq"], P_change=r["P_change"], C3=r["C3"]))
    report.write_csv(rows, "vii2_TR16_contamination", P, scen,
                     note="TR16: C3 = Pr[mu between mu_hat and mu_hat_C] (Proposition 16) at the calibrated central cell")
    return rows


# ------------------------------------------------------------------ TR17 drift and first passage
def run_TR17(P, scen):
    d = P.drift
    E0, R0 = P.electorate.E_full, P.electorate.R
    pr = make_prior("beta_shift", pi=0.25, a_B=3.0, mu_b=0.5)
    rows = []
    for t0 in (0.55, 0.60, 0.638, 0.70, 0.75, 0.81):
        for tm, tb in ((0.30, 0.90), (0.30, 1.00), (0.20, 0.90)):
            rl = rho_lock(t0, tm, prior=pr, recalibrate=True, t_bar=tb, hi=4.0)
            rho_now = E0 / R0
            for dR in (0.004, 0.0085, 0.0116):
                if rl <= rho_now:
                    yrs, yr = 0.0, "locked now (2025)"
                elif math.isinf(rl):
                    yrs, yr = float("inf"), "never (campaigns at every rho up to 4)"
                else:
                    yrs = t_cross(rl, E0, R0, d.gamma_E, dR)
                    yr = int(2025.25 + yrs)
                rows.append(dict(t0=t0, tau_equiv_2025=t0 / rho_now, t_min=tm, t_bar=tb, prior=pr.label, rho_lock=rl,
                                 rho_2025=rho_now, d_R=dR, years=yrs, lock_year=yr))
    report.write_csv(rows, "vii2_TR17_drift_lock", P, scen,
                     note="TR17: register ratio at which Regime A locks for fixed resident turnout t0 (Beta(3,3), kappa recalibrated), "
                          "and the year the A29 path (gamma_E 31,700) reaches it")
    ig = []
    rho_dom = (E0 - P.electorate.E_ocv) / R0
    for sig in (0.003, 0.005, 0.010):
        for gam, lab in ((0.017, "central"), (0.020, "fast")):
            ig.append(dict(sigma_lnrho=sig, gamma=gam, case=lab, Pr_rho_dom_falls_to_1_7803=pr_ever_falls(rho_dom, 1.7803, gam, sig)))
    report.write_csv(ig, "vii2_TR17_ig", P, scen, note="F75 at the calibrated sigma_lnrho")
    fp = []
    Q = E0 / R0 / 3
    for a in (1.5, 3.0, 5.0):
        for tb in (0.90, 1.00):
            b = Q / tb
            m = opinion.mean_first_passage(a, a, b, 0.5) if b < 1 else float("inf")
            for T_rel in (5.0, 15.0, 30.0):
                fp.append(dict(a_B=a, t_bar=tb, barrier_Q_over_tbar=b, mean_first_passage_T_rel_units=m, T_rel_years=T_rel,
                               mean_years=m * T_rel))
    report.write_csv(fp, "vii2_TR17_first_passage", P, scen,
                     note="Proposition 27: mean first passage of support from 1/2 to Q_A/t_bar (E_full) under the Jacobi diffusion with stationary Beta(a_B, a_B)")
    return rows


# ------------------------------------------------------------------ TR18 local units
def run_TR18(P, scen):
    st = scenario_settings(P)
    t0 = national(P, "E_full").t0(P.behaviour.tau_central)
    pk_cfg = load_yaml("packages.yaml")
    loc, summ = [], []
    for pk in PACKAGES:
        for T, level in (("T7", "municipality"), ("T8", "qark")):
            cfg = pk_cfg[pk]["rules"][T]
            us = local_units(P, cfg, t0=t0, t_min=P.behaviour.t_min_central, pi=st["pi_by_type"]["T7"], level=level,
                             sigma_tau=P.behaviour.sigma_tau, family=st["family"], a_B=st["a_B"], t_bar=st["t_bar"],
                             t_min_b=st["t_min_b"], recalibrate=st["recalibrate"])
            for u in us:
                u.update(package=pk, type=T)
            loc += us
            ok = [u for u in us if u.get("feasible")]
            if ok:
                summ.append(dict(package=pk, type=T, units=len(us), feasible=len(ok),
                                 locked=sum(1 for u in ok if u["regime"].startswith("lock")),
                                 change_possible=sum(1 for u in ok if not u["regime"].startswith("lock")),
                                 median_P_change=float(np.median([u["P_change"] for u in ok])),
                                 unlocked_units=", ".join(u["unit"] for u in ok if not u["regime"].startswith("lock"))[:400]))
            else:
                summ.append(dict(package=pk, type=T, units=len(us), feasible=0))
    report.write_csv(loc, "vii2_TR18_local_units", P, scen,
                     note=f"TR18: T7 municipalities and T8 qarks at resident turnout t0 = {t0:.3f} (national tau 0.33), calibrated prior, t_min 0.30, t_bar 0.90, A36")
    report.write_csv(summ, "vii2_TR18_local_summary", P, scen)
    # P4 (V10) unit dispersion
    v10 = []
    for su in (0.045, 0.065, 0.10):
        for r in evaluate_package(P, "P4", types=("T1",), overrides=dict(sigma_u=su)):
            v10.append(dict(sigma_u_qark=su, regime=r.get("regime"), P_change=r.get("P_change"), C1=r.get("C1"), C1R=r.get("C1R")))
    report.write_csv(v10, "vii2_TR18_P4_sigma_u", P, scen, note="P4 (V10, 7 of 12 qarks) for T1 at the calibrated centre by sigma_u (qarks)")
    return summ


# ------------------------------------------------------------------ TR19 verification
def run_TR19(P, scen):
    rows = signatures.error_table(v_grid=(0.70, 0.85, 0.97))
    report.write_csv(rows, "vii2_TR19_verification", P, scen,
                     note="TR19: R0 (5% sample) and the statutory band rule (F44) at v_L 0.70, 0.85, 0.97; exact hypergeometric")
    return rows


def run_all_calibrated(P, scen, n_draws=60):
    for name, fn in (("TR11", run_TR11), ("TR13", run_TR13), ("TR14", run_TR14), ("TR15", run_TR15), ("TR16", run_TR16),
                     ("TR17", run_TR17), ("TR18", run_TR18), ("TR19", run_TR19)):
        t = time.time()
        fn(P, scen)
        _log(f"{name} done in {time.time() - t:.0f} s")
    t = time.time()
    run_TR12(P, scen, n_draws=n_draws)
    _log(f"TR12 done in {time.time() - t:.0f} s")
    t = time.time()
    run_TR20(P, scen)
    _log(f"TR20 done in {time.time() - t:.0f} s")


# ------------------------------------------------------------------ TR20 the menu at the calibrated values (for Section VIII.1)
def run_TR20(P, scen, taus=(0.20, 0.25, 0.30, 0.33, 0.36, 0.42), tmins=(0.20, 0.30, 0.38)):
    from .rules import menu, V10
    st = scenario_settings(P)
    rows = []
    for den in ("E_full", "E_dom"):
        el = national(P, den)
        for r in menu():
            if isinstance(r, V10):
                r = V10(k_U=r.params["k_U"], m_U=r.params["m_U"], sigma_u=st["sigma_u"])   # v1.1.1: label = sigma_u used
            uw = qark_weights()[1] if isinstance(r, V10) else None
            for tau in taus:
                for tm in tmins:
                    if el.t0(tau) > st["t_bar"]:
                        continue
                    c = cell(r, el, tau, tm, st["pi_by_type"]["T1"], 0.5, st["family"], st["a_B"], P.behaviour.sigma_tau,
                             P.behaviour.Theta, recalibrate=st["recalibrate"], unit_weights=uw, t_bar=st["t_bar"],
                             t_min_b=st["t_min_b"], sigma_u=st["sigma_u"])
                    if c is None:
                        continue
                    rows.append({k: c[k] for k in ("rule", "code", "denominator", "tau", "t_min", "regime", "sx", "sd", "floor_d",
                                                   "mu_hat_eq", "P_change", "P_advisory", "P_dec", "C1", "C1p", "C1R", "C2", "turnout_E")})
    report.write_csv(rows, "vii2_TR20_menu", P, scen,
                     note="TR20: every menu rule for T1 at the calibrated values (D13 prior, pi 0.25, t_bar 0.90, A36 boycott floor 0.08, sigma_tau 0.04)")
    par = []
    import itertools
    for (den, tau, tm), grp in itertools.groupby(sorted(rows, key=lambda x: (x["denominator"], x["tau"], x["t_min"])),
                                                 key=lambda x: (x["denominator"], x["tau"], x["t_min"])):
        g = list(grp)
        crit = {x["rule"]: (x["C1R"], x["C1p"], x["C2"]) for x in g}
        pareto = [a for a in crit if not any(all(crit[b][k] <= crit[a][k] + 1e-9 for k in range(3)) and
                                              any(crit[b][k] < crit[a][k] - 1e-9 for k in range(3)) for b in crit if b != a)]
        par.append(dict(denominator=den, tau=tau, t_min=tm, n_rules=len(g), pareto_on_C1R_C1p_C2=" | ".join(pareto)))
    report.write_csv(par, "vii2_TR20_pareto", P, scen, note="TR20: rules not dominated on (C1^R, C1', C2) in each cell (lower is better on each)")
    return rows
