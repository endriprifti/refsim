"""Evaluate rules, types and packages: the criteria C1, C1', C2, C3, C4, C4' with intervals.

Cells: one rule on one electorate at (tau, t_min, pi, mu_b, prior).  Each cell is solved by game.Game
(equilibrium, then exact quadrature under A11 noise) and, on request, checked by the Monte Carlo layer.
Packages (config/packages.yaml) assign a rule to each type (config/types.yaml); C4 combines the cell's
Pr_dec with the pipeline (national initiatives T1-T2), the 60-day window (T4), or nothing (T3, T5, T6).
Parameter uncertainty: draws of R ~ U[1.89, 1.95]M, tau ~ U[D3], t_min ~ {0, 0.20, 0.35}, pi ~ U[0, 0.3]
give 2.5-97.5% intervals for every criterion.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache

import numpy as np
import yaml

from . import contamination, pipeline, signatures
from .electorate import Electorate, load_municipalities, national
from .game import Game
from .params import CONFIG_DIR
from .priors import make as make_prior
from .rules import V10, make_rule
from .simulate import simulate


def load_yaml(name):
    return yaml.safe_load(open(os.path.join(CONFIG_DIR, name), encoding="utf-8"))


def t4_feasible_share(d_rev_days: float, g_min_days: float = 30.0) -> float:
    """Proposition 23: share of adoption weekdays with a feasible Sunday within 60 days."""
    tot = d_rev_days + g_min_days
    if tot <= 54:
        return 1.0
    if tot > 60:
        return 0.0
    return (61 - tot) / 7.0


def cell(rule, el: Electorate, tau: float, t_min: float, pi: float = 0.0, mu_b: float = 0.5,
         family: str = "uniform", a_B: float = 1.0, sigma_tau: float = 0.04, Theta: float = 0.5,
         recalibrate: bool = True, Psi: float = 0.0, mu_Gx: float = 0.5, mc: int = 0, seed: int = 20261012,
         unit_weights=None, t_bar: float = 1.0, t_min_b: float | None = None, sigma_u: float = 0.05,
         prior=None, Omega: float = 1.0) -> dict | None:
    """prior: an explicit Prior (v1.1.1); when given, family/a_B/mu_b are ignored for the prior (pi still sets kappa
    under the fixed convention). Omega (v1.1.3): the stake ratio B_x/B_d of Section III (Proposition 8); 1 = equal stakes."""
    rule = make_rule(rule)
    prior = make_prior(family, pi=pi, a_B=a_B, mu_b=mu_b) if prior is None else prior
    g = Game(rule, el, tau=tau, t_min=t_min, prior=prior, pi=pi, recalibrate=recalibrate, t_bar=t_bar, t_min_b=t_min_b,
             Omega=Omega)
    if not g.feasible:
        return None
    eq = g.solve()
    c = g.criteria(eq, sigma_tau=sigma_tau, Theta=Theta)
    mh = c["mu_hat_eq"]
    c["C3"] = contamination.C3(prior, mh, Psi, mu_Gx) if (Psi > 0 and mh < 2) else 0.0
    c["exploitability"] = eq.exploitability
    c.update(rule=rule.describe(), code=rule.code, denominator=el.label, tau=tau, t_min=t_min, pi=pi, mu_b=mu_b,
             prior=prior.label, sigma_tau=sigma_tau, Q_over_tbar_ge_half=None, t_bar=t_bar,
             floor_d=g.floor_d, Psi=Psi, mu_Gx=mu_Gx, Omega=Omega)
    if mc or isinstance(rule, V10):
        sim = simulate(g, eq, n=max(mc, 20000), seed=seed, sigma_tau=sigma_tau, Theta=Theta,
                       unit_weights=unit_weights, sigma_u=sigma_u)
        for k in ("P_change", "P_dec", "C1", "C1p", "C1R"):
            c[k + "_mc"] = sim[k]
            c[k + "_mc_lo"] = sim[k + "_lo"]
            c[k + "_mc_hi"] = sim[k + "_hi"]
        if isinstance(rule, V10) and unit_weights is not None:
            # unit heterogeneity is only in the Monte Carlo: report it as the cell's value
            for k in ("P_change", "P_dec", "C1", "C1p", "C1R"):
                c[k] = sim[k]
    return c


@lru_cache(maxsize=1)
def qark_weights():
    m = load_municipalities()
    q = m.qarks(R_now=1.92e6)
    names = sorted(q)
    return tuple(names), tuple(q[k]["E"] for k in names)


def type_c4(type_code, pkg, c, Pr_vote_24=None):
    """C4 for a type given the cell's Pr_dec and Pr[change] (Remark 4; Proposition 23)."""
    if c is None:
        return dict(C4=np.nan)
    P_dec, P_ch = c["P_dec"], c["P_change"]
    if type_code in ("T1", "T2"):
        mo = pkg.get("mooting", "none")
        return dict(C4_opposed=pipeline.c4(Pr_vote_24, P_dec, P_ch, mo, 1.0),
                    C4_friendly=pipeline.c4(Pr_vote_24, P_dec, P_ch, mo, 0.0), Pr_vote_24=Pr_vote_24)
    if type_code == "T4":
        sh = t4_feasible_share(pkg.get("T4_review_days", 60))
        return dict(C4_opposed=sh * P_dec, C4_friendly=sh * P_dec, T4_feasible_share=sh)
    return dict(C4_opposed=P_dec, C4_friendly=P_dec)


def electorate_for(params, rule_cfg, pkg):
    den = rule_cfg.get("denominator", pkg.get("denominator", "E_full")) if isinstance(rule_cfg, dict) else pkg.get("denominator", "E_full")
    return national(params, "R" if den == "R" else den)


def _rule_from_cfg(cfg):
    cfg = dict(cfg)
    cfg.pop("denominator", None)
    return make_rule(cfg)


def scenario_settings(params) -> dict:
    """Scenario-level behaviour settings with v1.0 defaults (baseline_2025 keeps every default)."""
    B = params.behaviour
    keys = B.keys()
    opt = lambda k, d: getattr(B, k) if k in keys else d
    C = params.contamination if "contamination" in params.tree else None
    ckeys = C.keys() if C is not None else []
    copt = lambda k, d: getattr(C, k) if k in ckeys else d
    return dict(t_bar=opt("t_bar", 1.0), t_min_b=opt("t_min_boycott", None) if opt("a36", False) else None,
                family=opt("prior_family", "uniform"), a_B=opt("a_B_central", 1.0),
                recalibrate=opt("recalibrate_kappa", True), pi_by_type=opt("pi_by_type", None),
                mu_b_by_type=opt("mu_b_by_type", None), sigma_u=opt("sigma_u_qark", 0.05),
                Psi=copt("Psi", 0.0), Psi_by_type=copt("Psi_by_type", None), mu_G=copt("mu_G", 0.5),
                gov_backs_change=copt("gov_backs_change", None),
                noise_scaled_on_R=opt("noise_scaled_on_R", False))


def evaluate_package(params, pkg_name, tau=None, t_min=None, pi=None, family=None, a_B=None,
                     sigma_tau=None, types=("T1", "T2", "T3", "T4", "T5", "T6"), overrides=None, president=None) -> list:
    """overrides: dict of scenario settings to replace (t_bar, t_min_b, a_B, family, recalibrate, Psi, mu_G, ...).
    pkg_name: a key of config/packages.yaml or (v1.1.1) a package dict with a 'label'; president: 'earliest' or 'latest'
    (default: the scenario's pipeline.president)."""
    st = scenario_settings(params)
    if overrides:
        st.update(overrides)
    family = st["family"] if family is None else family
    a_B = st["a_B"] if a_B is None else a_B
    pkgs = load_yaml("packages.yaml")
    tys = load_yaml("types.yaml")
    if isinstance(pkg_name, dict):
        pkg = pkg_name
        pkg_name = pkg.get("label", "custom")
    else:
        pkg = pkgs[pkg_name]
    if tau is None:
        t0r = params.behaviour.t0_resident if "t0_resident" in params.behaviour.keys() else None
        tau = (t0r / national(params, "E_full").rho) if t0r else params.behaviour.tau_central
    t_min = params.behaviour.t_min_central if t_min is None else t_min
    sigma_tau = params.behaviour.sigma_tau if sigma_tau is None else sigma_tau
    FR = pipeline.kqz_false_rejection(pkg.get("verification", "R0"))
    pv = pipeline.pr_vote(pkg.get("timing", "O0"), president or params.pipeline.president, FR=FR)
    rows = []
    for T in types:
        rcfg = pkg["rules"][T]
        el = electorate_for(params, rcfg, pkg)
        if isinstance(rcfg, dict) and rcfg.get("code") == "V10" and "sigma_u" not in rcfg:
            rcfg = dict(rcfg, sigma_u=st["sigma_u"])      # v1.1.1: label carries the sigma_u actually used
        rule = _rule_from_cfg(rcfg)
        mu_b = (st["mu_b_by_type"] or {}).get(T, tys[T].get("agenda_mu_b", 0.5))
        pi_T = pi if pi is not None else (st["pi_by_type"] or {}).get(T, 0.0)
        Psi_T = (st["Psi_by_type"] or {}).get(T, st["Psi"])
        gb = (st["gov_backs_change"] or {}).get(T, False)
        mu_Gx = st["mu_G"] if gb else 1.0 - st["mu_G"]
        uw = qark_weights()[1] if isinstance(rule, V10) else None
        # v1.1.1: sigma_tau is a share of E_full; on a resident denominator the same turnout noise is sigma_tau x rho
        sig_c = sigma_tau * national(params, "E_full").rho if (el.label == "R" and st.get("noise_scaled_on_R")) else sigma_tau
        c = cell(rule, el, tau if el.label != "R" else tau * national(params, "E_full").rho, t_min, pi_T, mu_b,
                 family, a_B, sig_c, params.behaviour.Theta, recalibrate=st["recalibrate"], Psi=Psi_T,
                 mu_Gx=mu_Gx, unit_weights=uw, t_bar=st["t_bar"], t_min_b=st["t_min_b"], sigma_u=st["sigma_u"])
        if c is not None and el.label == "R":
            c["tau"] = tau
            c["turnout_E"] = c["turnout_E"] / national(params, "E_full").rho   # back to a share of E_full
        row = dict(package=pkg_name, package_name=pkg.get("name", pkg_name), type=T, type_name=tys[T]["name"])
        if c is None:
            row.update(feasible=False)
        else:
            row.update(feasible=True, **{k: c[k] for k in ("rule", "regime", "sx", "sd", "mu_hat_eq", "mu_V_eq", "P_change",
                                                        "P_advisory", "P_dec", "P_void", "C1", "C1p", "C1R", "C2", "C3",
                                                        "turnout_E", "rho", "t0")})
            row.update(pi=pi_T, mu_b=mu_b, prior=c["prior"], t_bar=c["t_bar"], floor_d=c["floor_d"], Psi=Psi_T,
                       mu_Gx=mu_Gx, tau=c["tau"], t_min=t_min)
        row.update(type_c4(T, pkg, c, pv["Pr_vote_24"]))
        row.update(Pr_vote_12=pv["Pr_vote_12"], Pr_vote_48=pv["Pr_vote_48"], mean_months=pv["mean_months"],
                   timing=pv["label"], verification=pkg.get("verification"), mooting=pkg.get("mooting"))
        rows.append(row)
    return rows


KEYS_UNC = ("P_change", "P_dec", "C1", "C1p", "C1R", "C2", "turnout_E", "mu_hat_eq", "C4_opposed", "C4_friendly")


def package_uncertainty(params, pkg_name, n_draws=60, seed=7, types=("T1", "T3", "T4", "T5"), family="uniform",
                        return_draws=False):
    """Mean, 2.5-97.5% interval, minimum and maximum of the criteria over parameter draws (R, tau, t_min, pi).
    The draw sequence depends only on `seed`, so every package sees the same draws (paired comparisons)."""
    rng = np.random.default_rng(seed)
    base = params
    out = {T: {k: [] for k in KEYS_UNC} for T in types}
    draws = []
    lo_R, hi_R = params.electorate.meta("R")["range"]
    for i in range(n_draws):
        R = rng.uniform(lo_R, hi_R)
        tau = rng.uniform(0.20, 0.50)
        tm = float(rng.choice([0.0, 0.20, 0.35]))
        pi = rng.uniform(0.0, 0.3)
        p = base.override(**{"electorate.R": R})
        if p.electorate.E_full / R * tau <= tm + 1e-6:
            continue
        rows = evaluate_package(p, pkg_name, tau=tau, t_min=tm, pi=pi, family=family, types=types)
        for r in rows:
            if not r.get("feasible"):
                continue
            d = dict(package=pkg_name, type=r["type"], draw=i, R=R, tau=tau, t_min=tm, pi=pi, regime=r.get("regime"))
            for k in out[r["type"]]:
                v = r.get(k)
                ok = v is not None and not (isinstance(v, float) and math.isnan(v)) and v < 2
                if ok:
                    out[r["type"]][k].append(v)
                d[k] = v if ok else None
            draws.append(d)
    summ = []
    for T, dd in out.items():
        row = dict(package=pkg_name, type=T, n=len(dd["P_change"]),
                   n_locked=sum(1 for d in draws if d["type"] == T and str(d["regime"]).startswith("lock")))
        for k, v in dd.items():
            if v:
                row[k + "_mean"] = float(np.mean(v))
                row[k + "_lo95"] = float(np.percentile(v, 2.5))
                row[k + "_hi95"] = float(np.percentile(v, 97.5))
                row[k + "_min"] = float(np.min(v))
                row[k + "_max"] = float(np.max(v))
        summ.append(row)
    return (summ, draws) if return_draws else summ


def local_units(params, rule_cfg, t0=None, t_min=0.20, pi=0.0, level="municipality", sigma_tau=0.04, R_now=1.92e6,
                family="uniform", a_B=1.0, t_bar=1.0, t_min_b=None, recalibrate=True):
    """T7 (the municipalities of the active map: 46 from 2027, 61 on the 2014 map) or T8 (12 qarks) on unit denominators (D4), R_hat rebased (D11).
    Resident behaviour is held fixed across units: stand-alone resident turnout t0 (default: the national
    value at tau = 0.30 of E_full, 0.580), so registered turnout varies with rho_u."""
    if t0 is None:
        t0 = national(params, "E_full").t0(params.behaviour.tau_central)
    m = load_municipalities()
    rule = _rule_from_cfg(rule_cfg)
    rows = []
    if level == "municipality":
        units = [(m.names[i], m.E_hat[i], m.R_hat(R_now)[i]) for i in range(len(m.names))]
    else:
        q = m.qarks(R_now)
        units = [(k, v["E"], v["R"]) for k, v in sorted(q.items())]
    for name, Eu, Ru in units:
        el = Electorate(E=float(Eu), R=float(Ru), label=f"E_u {name}")
        c = cell(rule, el, el.tau(t0), t_min, pi, 0.5, family, a_B, sigma_tau, recalibrate=recalibrate,
                 t_bar=t_bar, t_min_b=t_min_b)
        row = dict(unit=name, level=level, E_u=Eu, R_u=Ru, rho_u=Eu / Ru, rule=rule.describe(), t0=t0, tau_u=el.tau(t0))
        if c is None:
            row.update(feasible=False)
        else:
            row.update(feasible=True, regime=c["regime"], P_change=c["P_change"], P_dec=c["P_dec"], C1=c["C1"],
                       C1p=c["C1p"], C2=c["C2"], mu_hat=c["mu_hat_eq"], sd=c["sd"], sx=c["sx"],
                       infeasible_bar=(rule.code in ("V3",) and rule.params.get("q", 1 / 3) * Eu / Ru > 1.0))
        rows.append(row)
    return rows


def c4_prime_table():
    """C4' (false admission) and false rejection: R0 vs the F44 rule, national / Korce / Pustec."""
    return signatures.error_table()
