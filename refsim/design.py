"""Tool runs TR21-TR28 for Section VIII, Part 1 (refsim v1.1.1) and TR29 for Appendix H (v1.1.2), scenario calibrated_2026.

TR21  combined packages: validity rules of P1, P1b, P4, V1 and V7 at 0.55, with and without P5's procedure
      (TS9, R3, substance mooting rule, 24-day T4 schedule); centre and the 60 common draws of TR12; C4 under a
      prompt and a latest-date President; paired differences against P0.
TR22  binding-effect screen of the TR20 menu: C1' and C1^R on binding results only (an advisory or consultative
      Yes counts as no change); Pareto sets with and without V9 and V11, at full and at printed precision;
      C1^Sigma and the indifference weight against V1 for every rule and cell.
TR23  the menu for T3, T4 and T6 under the D13 type priors (and T6 under illustrative accession priors).
TR24  the T5 menu with the adopted amendment as default (V1; V7 'Yes must reach 55%'; V4 on residents).
TR25  the local menu (T7, T8) on unit denominators, with proxy error in the resident estimate.
TR26  the election window at the calibrated decay constants ((5.31)-(5.33)).
TR27  level boundaries of V3 (E_full, E_dom) and V6d (R) by turnout; C2 by turnout for V2 at 0.25 and V5p.
TR28  the complexity bound under V1 and V7 at 0.55 (reach branch of Corollary 7 and a random-vote branch).
TR29  (v1.1.2, Appendix H) the indifference weight w* against V1 for T6 under illustrative accession priors
      (Beta with the variance of Beta(3, 3), mean 0.50-0.80, or the author's mean): V7 at 0.55, the imported one-third
      quorum and V6d at 1/4 and 1/3 of R; E_full, t_min 0.30, tau 0.20-0.42 and TAU_ANCHOR (0.394; 0.393 before v1.1.4).
Every run writes outputs/<scenario>/viii1_TR*.csv (+ .meta.json).
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np

from . import campaign, pipeline, report, window
from .calibrated import TYPES, VARIANTS, _log, draw_params
from .electorate import Electorate, load_municipalities, national
from .evaluate import cell, evaluate_package, load_yaml, qark_weights, scenario_settings
from .game import tau_lock
from .priors import beta_shift, make as make_prior
from .rules import V1, V2, V3, V5, V6, V7, V10, make_rule, menu

PROC = dict(timing="O9", window="polling_day", verification="R3", mooting="substance", T4_review_days=24)
CUR = dict(timing="O0", window="mandate", verification="R0", mooting="none", T4_review_days=60)
ALL_T = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
TAUS = (0.20, 0.25, 0.30, 0.33, 0.36, 0.42)
# v1.1.4: the 2025 in-country anchor T_dom/E_full, with T_dom = 1,463,314 (OSCE/ODIHR 2025 final report, Annex I, CEC results;
# v1.1.3 used 1,458,251 from preliminary press data, giving 0.393)
TAU_ANCHOR = round(1463314 / 3713761, 3)
TMINS = (0.20, 0.30, 0.38)
METRICS = ("P_change", "P_dec", "C1p", "C1R", "C2", "C4_opposed", "C4_friendly", "C4_opposed_latest", "C4_friendly_latest")


def _pkg(label, rules, proc):
    d = dict(label=label, name=label, rules=rules, denominator="E_full")
    d.update(proc)
    return d


def combined_packages() -> dict:
    """P0 and P5 for reference; P1, P1b, P4 validity with P5's procedure; V1 and V7 at 0.55 with and without it.
    T5 under V7: the amendment stays in force only if Yes reaches 55% of valid votes, i.e. the challenger (No)
    wins at 45% (V7 with q_S = 0.45 for the No side)."""
    pk = load_yaml("packages.yaml")
    out = {"P0": _pkg("P0", pk["P0"]["rules"], CUR), "P5": _pkg("P5", pk["P5"]["rules"], PROC)}
    for b in ("P1", "P1b", "P4"):
        out[b + "+P5"] = _pkg(b + "+P5", pk[b]["rules"], PROC)
    v1 = {T: {"code": "V1"} for T in ALL_T}
    v7 = {T: {"code": "V7", "qS": 0.55} for T in ALL_T}
    v7["T5"] = {"code": "V7", "qS": 0.45}
    out["V1"] = _pkg("V1", v1, CUR)
    out["V1+P5"] = _pkg("V1+P5", v1, PROC)
    out["V7"] = _pkg("V7 0.55", v7, CUR)
    out["V7+P5"] = _pkg("V7 0.55+P5", v7, PROC)
    return out


def _latest_c4(pkg, T, r):
    """C4 with a latest-date President (T1-T2 only; other types do not use the pipeline)."""
    if not r.get("feasible"):
        return dict(C4_opposed_latest=None, C4_friendly_latest=None)
    if T not in ("T1", "T2"):
        return dict(C4_opposed_latest=r.get("C4_opposed"), C4_friendly_latest=r.get("C4_friendly"))
    FR = pipeline.kqz_false_rejection(pkg.get("verification", "R0"))
    pv = pipeline.pr_vote(pkg.get("timing", "O0"), "latest", FR=FR)["Pr_vote_24"]
    mo = pkg.get("mooting", "none")
    return dict(C4_opposed_latest=pipeline.c4(pv, r["P_dec"], r["P_change"], mo, 1.0),
                C4_friendly_latest=pipeline.c4(pv, r["P_dec"], r["P_change"], mo, 0.0), Pr_vote_24_latest=pv)


def _eval(P, pkg, **kw):
    rows = evaluate_package(P, pkg, president="earliest", **kw)
    T_list = kw.get("types", TYPES)
    for r, T in zip(rows, T_list):
        r.update(_latest_c4(pkg, T, r))
    return rows


# ------------------------------------------------------------------ TR21
def run_TR21(P, scen, n_draws=60, seed=7):
    pk = combined_packages()
    cen = []
    for name, d in pk.items():
        for r in _eval(P, d, types=TYPES):
            row = dict(package=name, type=r["type"], feasible=r.get("feasible"), rule=r.get("rule"), regime=r.get("regime"),
                       timing=r.get("timing"), verification=r.get("verification"), mooting=r.get("mooting"),
                       Pr_vote_24_prompt=r.get("Pr_vote_24"), Pr_vote_24_latest=r.get("Pr_vote_24_latest"))
            row.update({k: r.get(k) for k in METRICS})
            cen.append(row)
    report.write_csv(cen, "viii1_TR21_central", P, scen,
                     note="TR21: combined packages at the calibrated centre; C4 under a prompt (earliest) and a latest-date President "
                          "(T1-T2 pipeline; T4 60-day schedule; T3, T5, T6 = Pr_dec)")
    draws = draw_params(P, n_draws, seed)
    names = ["P0", "P1+P5", "P1b+P5", "P4+P5", "V1", "V1+P5", "V7", "V7+P5"]
    types = ("T1", "T3", "T4", "T5")
    allrows = []
    for name in names:
        _log(f"TR21 draws for {name}")
        for d in draws:
            p = P.override(**{"electorate.R": d["R"]})
            if p.electorate.E_full / d["R"] * d["tau"] > d["t_bar"]:
                continue
            ov = dict(t_min_b=d["t_min_b"], t_bar=d["t_bar"], a_B=d["a_B"], mu_G=d["mu_G"],
                      pi_by_type={"T1": d["pi_12"], "T2": d["pi_12"], "T3": d["pi_36"], "T4": d["pi_36"], "T5": d["pi_36"], "T6": d["pi_36"]},
                      Psi_by_type={T: d["Psi"] for T in TYPES})
            for r in _eval(p, pk[name], tau=d["tau"], t_min=d["t_min"], sigma_tau=d["sigma_tau"], types=types, overrides=ov):
                if not r.get("feasible"):
                    continue
                row = dict(package=name, type=r["type"], regime=r["regime"], draw=d["draw"])
                for k in METRICS:
                    v = r.get(k)
                    row[k] = v if (v is not None and not (isinstance(v, float) and math.isnan(v)) and v < 2) else None
                allrows.append(row)
    report.write_csv(allrows, "viii1_TR21_draws", P, scen, note=f"TR21: {n_draws} common draws (seed {seed}, as TR12)")
    summ = []
    for name in names:
        for T in types:
            rr = [r for r in allrows if r["package"] == name and r["type"] == T]
            row = dict(package=name, type=T, n=len(rr), n_locked=sum(1 for r in rr if str(r["regime"]).startswith("lock")))
            for k in METRICS:
                v = [r[k] for r in rr if r[k] is not None]
                if v:
                    row.update({f"{k}_mean": float(np.mean(v)), f"{k}_min": float(np.min(v)), f"{k}_max": float(np.max(v))})
            summ.append(row)
    report.write_csv(summ, "viii1_TR21_summary", P, scen, note="TR21 summary over the common draws")
    base = {(r["type"], r["draw"]): r for r in allrows if r["package"] == "P0"}
    paired = []
    for name in names[1:]:
        for T in types:
            for k in ("P_change", "C1p", "C1R", "C2", "C4_opposed", "C4_friendly", "C4_opposed_latest"):
                dd = [r[k] - base[(T, r["draw"])][k] for r in allrows if r["package"] == name and r["type"] == T
                      and (T, r["draw"]) in base and r[k] is not None and base[(T, r["draw"])][k] is not None]
                if dd:
                    a = np.array(dd)
                    paired.append(dict(package=name, type=T, criterion=k, n=len(dd), mean=float(a.mean()), min=float(a.min()),
                                       max=float(a.max()), share_pos=float(np.mean(a > 1e-6)), share_neg=float(np.mean(a < -1e-6))))
    report.write_csv(paired, "viii1_TR21_paired", P, scen, note="TR21: package minus P0 on the common draws (shares count |difference| > 1e-6)")
    return summ


# ------------------------------------------------------------------ TR22
def _menu_rows(P, taus=TAUS, tmins=TMINS, dens=("E_full", "E_dom")):
    """The TR20 cells (T1 menu) at full precision, without writing files."""
    st = scenario_settings(P)
    rows = []
    for den in dens:
        el = national(P, den)
        for r in menu():
            if isinstance(r, V10):
                r = V10(k_U=r.params["k_U"], m_U=r.params["m_U"], sigma_u=st["sigma_u"])
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
                    rows.append({k: c[k] for k in ("rule", "code", "denominator", "tau", "t_min", "regime", "sx", "sd",
                                                   "P_change", "P_advisory", "P_dec", "C1", "C1p", "C1R", "C2")})
    return rows


def binding_only(row, P_half):
    """C1' and C1^R on binding results: an advisory (V9) or consultative (V11) Yes counts as no change."""
    code = row["code"]
    if code == "V11":
        return dict(P_bind=0.0, C1p_b=0.0, C1R_b=P_half)
    if code == "V9":
        return dict(P_bind=row["P_change"], C1p_b=row["C1p"], C1R_b=P_half - (row["P_change"] - row["C1p"]))
    return dict(P_bind=row["P_change"], C1p_b=row["C1p"], C1R_b=row["C1R"])


def pareto(crit: dict, tol=1e-9) -> list:
    """Names not dominated (lower is better on every coordinate)."""
    out = []
    for a in crit:
        dom = False
        for b in crit:
            if b == a:
                continue
            if all(crit[b][k] <= crit[a][k] + tol for k in range(len(crit[a]))) and \
                    any(crit[b][k] < crit[a][k] - tol for k in range(len(crit[a]))):
                dom = True
                break
        if not dom:
            out.append(a)
    return out


def weight_vs(ref, x):
    """Indifference weight w* between rule x and the reference (V1): loss w*C1' + C1^R."""
    dp, dr = x["C1p_b"] - ref["C1p_b"], x["C1R_b"] - ref["C1R_b"]
    if abs(dp) < 1e-12 and abs(dr) < 1e-12:
        return None, "equal to V1"
    if dp <= 0 and dr <= 0:
        return None, "better than V1 at every w"
    if dp >= 0 and dr >= 0:
        return None, "V1 better at every w"
    if dp < 0:
        return dr / (-dp), "preferred to V1 iff w > w*"
    return (-dr) / dp, "preferred to V1 iff w < w*"


def run_TR22(P, scen):
    st = scenario_settings(P)
    pr = make_prior(st["family"], pi=st["pi_by_type"]["T1"], a_B=st["a_B"], mu_b=0.5)
    P_half = float(pr.mass(0.5, np.inf))
    rows = _menu_rows(P)
    for r in rows:
        r.update(binding_only(r, P_half))
        r["C1sum_b"] = r["C1p_b"] + r["C1R_b"]
    key = lambda r: (r["denominator"], r["tau"], r["t_min"])
    cells, wrows = [], []
    for k, grp in itertools.groupby(sorted(rows, key=key), key=key):
        g = list(grp)
        S0 = pareto({x["rule"]: (x["C1R"], x["C1p"], x["C2"]) for x in g})
        S1 = pareto({x["rule"]: (x["C1R_b"], x["C1p_b"], x["C2"]) for x in g})
        gb = [x for x in g if x["code"] not in ("V9", "V11")]
        S2 = pareto({x["rule"]: (x["C1R_b"], x["C1p_b"], x["C2"]) for x in gb})
        S3 = pareto({x["rule"]: (round(x["C1R_b"], 3), round(x["C1p_b"], 3), round(x["C2"], 3)) for x in gb}, tol=0.0)
        cells.append(dict(denominator=k[0], tau=k[1], t_min=k[2], n_rules=len(g),
                          S0_as_TR20=" | ".join(S0), S1_binding=" | ".join(S1), S2_binding_no_V9_V11=" | ".join(S2),
                          S3_binding_no_V9_V11_printed=" | ".join(S3)))
        ref = [x for x in g if x["code"] == "V1"][0]
        for x in g:
            w, how = weight_vs(ref, x)
            wrows.append(dict(denominator=k[0], tau=k[1], t_min=k[2], rule=x["rule"], code=x["code"], regime=x["regime"],
                              P_bind=x["P_bind"], C1p_b=x["C1p_b"], C1R_b=x["C1R_b"], C1sum_b=x["C1sum_b"], C2=x["C2"],
                              w_star_vs_V1=w, reading=how, in_S2=x["rule"] in S2, in_S3=x["rule"] in S3))
    report.write_csv(cells, "viii1_TR22_pareto", P, scen,
                     note="TR22: Pareto sets on (C1^R, C1', C2): S0 as TR20; S1 on binding results (advisory/consultative Yes = no change); "
                          "S2 = S1 without V9 and V11; S3 = S2 at printed precision (criteria rounded to three decimals, ties kept)")
    report.write_csv(wrows, "viii1_TR22_weights", P, scen,
                     note=f"TR22: binding-only criteria, C1^Sigma = C1' + C1^R and the indifference weight w* against V1 (loss w C1' + C1^R); "
                          f"Pr[mu > 1/2] = {P_half:.6f}")
    return cells


# ------------------------------------------------------------------ TR23
def menu_types(sigma_u):
    return [V1(), V6(q=0.25, variant="d"), V6(q=1 / 3, variant="d"), V7(qS=0.55), V7(qS=0.6), V7(qS=2 / 3),
            V3(q=0.20), V3(q=0.25), V3(q=1 / 3), V10(k_U=7, m_U=12, sigma_u=sigma_u)]


def run_TR23(P, scen):
    st = scenario_settings(P)
    el = national(P, "E_full")
    tys = load_yaml("types.yaml")
    rows = []
    for T in ("T3", "T4", "T6"):
        pi_T = st["pi_by_type"][T]
        mu_b = (st["mu_b_by_type"] or {}).get(T, tys[T].get("agenda_mu_b", 0.5))
        pr = make_prior(st["family"], pi=pi_T, a_B=st["a_B"], mu_b=mu_b)
        P_half = float(pr.mass(0.5, np.inf))
        for r in menu_types(st["sigma_u"]):
            uw = qark_weights()[1] if isinstance(r, V10) else None
            for tau in TAUS:
                for tm in TMINS:
                    c = cell(r, el, tau, tm, pi_T, mu_b, st["family"], st["a_B"], P.behaviour.sigma_tau, P.behaviour.Theta,
                             recalibrate=st["recalibrate"], unit_weights=uw, t_bar=st["t_bar"], t_min_b=st["t_min_b"],
                             sigma_u=st["sigma_u"])
                    if c is None:
                        continue
                    rows.append(dict(type=T, prior=pr.label, P_mu_gt_half=P_half, rule=c["rule"], code=c["code"], tau=tau, t_min=tm,
                                     regime=c["regime"], sx=c["sx"], sd=c["sd"], P_change=c["P_change"], P_dec=c["P_dec"],
                                     C1p=c["C1p"], C1R=c["C1R"], C2=c["C2"]))
    report.write_csv(rows, "viii1_TR23_menu_T3_T4_T6", P, scen,
                     note="TR23: menu for T3 (pi 0.50, mu_b 0.53), T4 (pi 0.50, mu_b 2/3) and T6 (pi 0.50, mu_b 1/2) under the D13 type priors; "
                          "E_full; V10 with the calibrated sigma_u")
    par = []
    key = lambda r: (r["type"], r["tau"], r["t_min"])
    for k, grp in itertools.groupby(sorted(rows, key=key), key=key):
        g = list(grp)
        S = pareto({x["rule"]: (x["C1R"], x["C1p"], x["C2"]) for x in g})
        Sp = pareto({x["rule"]: (round(x["C1R"], 3), round(x["C1p"], 3), round(x["C2"], 3)) for x in g}, tol=0.0)
        par.append(dict(type=k[0], tau=k[1], t_min=k[2], pareto=" | ".join(S), pareto_printed=" | ".join(Sp)))
    report.write_csv(par, "viii1_TR23_pareto", P, scen, note="TR23: Pareto sets on (C1^R, C1', C2) by type and cell (full and printed precision)")
    # T6 under illustrative accession priors (the author supplies the mean)
    acc = []
    for m in (0.50, 0.60, 0.70, 0.80):
        pr = beta_shift(pi=1.0, a_B=st["a_B"], mu_b=m)
        P_half = float(pr.mass(0.5, np.inf))
        for r in (V3(q=1 / 3), V1(), V7(qS=0.55), V6(q=0.25, variant="d"), V6(q=1 / 3, variant="d")):
            for tau in TAUS + (TAU_ANCHOR,):
                c = cell(r, el, tau, 0.30, st["pi_by_type"]["T6"], 0.5, st["family"], st["a_B"], P.behaviour.sigma_tau,
                         P.behaviour.Theta, recalibrate=st["recalibrate"], t_bar=st["t_bar"], t_min_b=st["t_min_b"], prior=pr)
                if c is None:
                    continue
                acc.append(dict(prior_mean=m, prior=pr.label, P_mu_gt_half=P_half, rule=c["rule"], code=c["code"], tau=tau, t_min=0.30,
                                regime=c["regime"], sx=c["sx"], sd=c["sd"], P_change=c["P_change"], P_dec=c["P_dec"],
                                C1p=c["C1p"], C1R=c["C1R"], C2=c["C2"]))
    report.write_csv(acc, "viii1_TR23_T6_accession", P, scen,
                     note="TR23: T6 under illustrative accession priors (Beta with the variance of Beta(3,3), mean 0.50-0.80; the author "
                          "supplies the mean); E_full, t_min 0.30; tau 0.394 = the 2025 in-country Assembly-election turnout (1,463,314 / 3,713,761)")
    return rows


# ------------------------------------------------------------------ TR24
T5_MENU = [
    ("V4 on E at 1/2 (current law)", {"code": "V4", "q": 0.5}),
    ("V4 on R at 1/2 (P1)", {"code": "V4", "q": 0.5, "denominator": "R"}),
    ("V4 on R at 1/3, with N > Y", {"code": "V4", "q": 1 / 3, "denominator": "R"}),
    ("V4 on R at 0.25, with N > Y", {"code": "V4", "q": 0.25, "denominator": "R"}),
    ("V4 on R at 0.20, with N > Y", {"code": "V4", "q": 0.20, "denominator": "R"}),
    ("V1 (falls if No exceeds Yes)", {"code": "V1"}),
    ("V7 (stands only if Yes reaches 55%)", {"code": "V7", "qS": 0.45}),
]
T5_KEYS = ("regime", "sx", "sd", "P_change", "P_dec", "C1p", "C1R", "C2")


def _t5pkg(label, cfg):
    return _pkg(label, {"T5": cfg}, CUR)


def run_TR24(P, scen, n_draws=60, seed=7):
    cen, var, dr = [], [], []
    for lab, cfg in T5_MENU:
        pk = _t5pkg(lab, cfg)
        for vlab, kw, ov in VARIANTS:
            r = evaluate_package(P, pk, types=("T5",), overrides=ov, **kw)[0]
            row = dict(rule=lab, variant=vlab, feasible=r.get("feasible"), **{k: r.get(k) for k in T5_KEYS})
            var.append(row)
            if vlab == "central":
                cen.append(row)
        for d in draw_params(P, n_draws, seed):
            p = P.override(**{"electorate.R": d["R"]})
            if p.electorate.E_full / d["R"] * d["tau"] > d["t_bar"]:
                continue
            ov = dict(t_min_b=d["t_min_b"], t_bar=d["t_bar"], a_B=d["a_B"], mu_G=d["mu_G"],
                      pi_by_type={"T1": d["pi_12"], "T2": d["pi_12"], "T3": d["pi_36"], "T4": d["pi_36"], "T5": d["pi_36"], "T6": d["pi_36"]},
                      Psi_by_type={T: d["Psi"] for T in TYPES})
            r = evaluate_package(p, pk, tau=d["tau"], t_min=d["t_min"], sigma_tau=d["sigma_tau"], types=("T5",), overrides=ov)[0]
            if r.get("feasible"):
                dr.append(dict(rule=lab, draw=d["draw"], **{k: r.get(k) for k in T5_KEYS}))
    report.write_csv(cen, "viii1_TR24_T5_central", P, scen,
                     note="TR24: T5 menu at the calibrated centre; default = the adopted amendment; Pr[change] = Pr[rejection]")
    report.write_csv(var, "viii1_TR24_T5_variants", P, scen, note="TR24: T5 menu across the 15 one-at-a-time variants of TR11")
    report.write_csv(dr, "viii1_TR24_T5_draws", P, scen, note=f"TR24: T5 menu on the {n_draws} common draws (seed {seed})")
    summ = []
    for lab, _ in T5_MENU:
        rr = [r for r in dr if r["rule"] == lab]
        row = dict(rule=lab, n=len(rr), n_locked=sum(1 for r in rr if str(r["regime"]).startswith("lock")))
        for k in ("P_change", "C1p", "C1R", "C2"):
            v = [r[k] for r in rr if r[k] is not None]
            row.update({f"{k}_mean": float(np.mean(v)), f"{k}_min": float(np.min(v)), f"{k}_max": float(np.max(v))})
        summ.append(row)
    report.write_csv(summ, "viii1_TR24_T5_summary", P, scen, note="TR24 summary over the common draws")
    return summ


# ------------------------------------------------------------------ TR25
def run_TR25(P, scen, R_now=1.92e6, mults=(0.90, 0.95, 1.00, 1.05, 1.10)):
    st = scenario_settings(P)
    t0 = national(P, "E_full").t0(P.behaviour.tau_central)
    m = load_municipalities()
    units = [("municipality", m.names[i], float(m.E_hat[i]), float(m.R_hat(R_now)[i])) for i in range(len(m.names))]
    q = m.qarks(R_now)
    units += [("qark", k, float(v["E"]), float(v["R"])) for k, v in sorted(q.items())]
    rules = [("V3 at 1/3 of E_u (current law)", V3(q=1 / 3), 1.0), ("V1", V1(), 1.0), ("V7 at 0.55", V7(qS=0.55), 1.0)]
    for qq, lab in ((0.25, "1/4"), (1 / 3, "1/3")):
        for mu in mults:
            rules.append((f"V6d at {lab} of R_u, proxy x{mu:.2f}", V6(q=qq * mu, variant="d"), mu))
    pi_T = st["pi_by_type"].get("T7", st["pi_by_type"].get("T1", 0.25)) if st["pi_by_type"] else 0.0
    rows = []
    for level, name, Eu, Ru in units:
        el = Electorate(E=Eu, R=Ru, label=f"E_u {name}")
        for lab, r, mu in rules:
            c = cell(r, el, el.tau(t0), P.behaviour.t_min_central, pi_T, 0.5, st["family"], st["a_B"], P.behaviour.sigma_tau,
                     P.behaviour.Theta, recalibrate=st["recalibrate"], t_bar=st["t_bar"], t_min_b=st["t_min_b"])
            row = dict(level=level, unit=name, E_u=Eu, R_u=Ru, rho_u=Eu / Ru, rule=lab, proxy_multiplier=mu, t0_u=t0)
            if c is None:
                row.update(feasible=False)
            else:
                row.update(feasible=True, regime=c["regime"], P_change=c["P_change"], P_dec=c["P_dec"], C1p=c["C1p"],
                           C1R=c["C1R"], C2=c["C2"])
            rows.append(row)
    report.write_csv(rows, "viii1_TR25_local_units", P, scen,
                     note=f"TR25: local menu by unit (Appendix D, Table D.4); unit denominators; resident turnout t0 = {t0:.3f}; "
                          "V6d's bar counted on R_hat_u = (proxy multiplier) x R_u")
    summ = []
    for level in ("municipality", "qark"):
        for lab, _, mu in rules:
            rr = [x for x in rows if x["level"] == level and x["rule"] == lab and x.get("feasible")]
            if not rr:
                continue
            locked = [x for x in rr if str(x["regime"]).startswith("lock")]
            summ.append(dict(level=level, rule=lab, units=len(rr), locked=len(locked),
                             locked_units=", ".join(x["unit"] for x in locked)[:300],
                             median_P_change=float(np.median([x["P_change"] for x in rr])),
                             median_C1p=float(np.median([x["C1p"] for x in rr])), max_C1p=float(np.max([x["C1p"] for x in rr])),
                             median_C1R=float(np.median([x["C1R"] for x in rr])), max_C1R=float(np.max([x["C1R"] for x in rr]))))
    report.write_csv(summ, "viii1_TR25_local_summary", P, scen, note="TR25 summary by level and rule")
    return summ


# ------------------------------------------------------------------ TR26
def run_TR26(P, scen, q_w=0.10):
    Ln = math.log(1 / q_w)
    rows = []
    T_pres = (0.5, 0.75, 1.0, 1.25, 1.5, round(4 / Ln, 3), 2.0)
    T_posts = (0.5, 1.0, 1.5, 2.0, 2.12, 2.5, 3.0, 3.27, 4.0, 5.0, 6.0, 8.0)
    for Tp in T_pres:
        for Tq in T_posts:
            row = dict(T_pre=Tp, T_post=Tq, q_w=q_w, pre_side_needed_months=Tp * Ln, post_side_needed_months=Tq * Ln,
                       shortest_compliant_blocked_share=window.anchored_window_blocked_share(Tp, Tq, q_w),
                       recommended_pre_complies=Tp <= 4 / Ln)
            for y, dpost in window.D_POST.items():
                row[f"recommended_post_complies_{y}"] = Tq <= dpost / Ln
                row[f"current_pre_complies_{y}"] = Tp <= window.D_PRE[y] / Ln
            row["recommended_complies_all_years"] = row["recommended_pre_complies"] and all(
                row[f"recommended_post_complies_{y}"] for y in window.D_POST)
            rows.append(row)
    report.write_csv(rows, "viii1_TR26_window_grid", P, scen,
                     note="TR26: (5.31)-(5.33) at q_w = 0.10; recommended window = four months before Assembly polling day to three months "
                          "after the first sitting (post-side length from polling day by election year, window.D_POST)")
    yrs = []
    lo_pre, hi_pre, lo_post, hi_post = 0.5, 2.0, 0.5, 8.0
    for y in window.D_POST:
        dpost, dpre = window.D_POST[y], window.D_PRE[y]
        yrs.append(dict(year=y, post_side_months=dpost, pre_side_as_opened_months=dpre,
                        T_post_max=dpost / Ln, T_pre_max_current=dpre / Ln, T_pre_max_recommended=4 / Ln,
                        recommended_blocked_share=(4 + dpost) / 48.0,
                        share_of_T_post_range_complying=max(0.0, min(hi_post, dpost / Ln) - lo_post) / (hi_post - lo_post),
                        share_of_T_pre_range_complying_recommended=max(0.0, min(hi_pre, 4 / Ln) - lo_pre) / (hi_pre - lo_pre),
                        share_of_T_pre_range_complying_current=max(0.0, min(hi_pre, dpre / Ln) - lo_pre) / (hi_pre - lo_pre)))
    report.write_csv(yrs, "viii1_TR26_window_years", P, scen,
                     note="TR26: compliance bounds and blocked shares by election year; shares of the calibrated ranges T_pre [0.5, 2.0] and "
                          "T_post [0.5, 8] (uniform) that comply")
    return rows


# ------------------------------------------------------------------ TR27
LEVELS = (0.15, 0.20, 0.25, 0.30, 1 / 3)
BIAS_TOL = 0.005     # a cell is challenger-biased if C1' exceeds C1^R by more than this (slack cells differ by < 0.002)


def _runs(taus, step=0.01):
    """Contiguous runs of grid values, as '0.23-0.29; 0.41'."""
    if not taus:
        return ""
    t = sorted(taus)
    out, a, b = [], t[0], t[0]
    for x in t[1:]:
        if x - b <= step + 1e-9:
            b = x
        else:
            out.append(f"{a:.2f}" if a == b else f"{a:.2f}-{b:.2f}")
            a = b = x
    out.append(f"{a:.2f}" if a == b else f"{a:.2f}-{b:.2f}")
    return "; ".join(out)


def run_TR27(P, scen, step=0.01, tmins=TMINS):
    st = scenario_settings(P)
    pr = make_prior(st["family"], pi=st["pi_by_type"]["T1"], a_B=st["a_B"], mu_b=0.5)
    specs = []
    for qv in LEVELS:
        specs.append((f"V3 at {qv:.3g} of E_full", V3(q=qv), "E_full"))
        specs.append((f"V3 at {qv:.3g} of E_dom", V3(q=qv), "E_dom"))
        specs.append((f"V6d at {qv:.3g} of R", V6(q=qv, variant="d"), "E_full"))
    c2specs = []
    for den in ("E_full", "E_dom"):
        c2specs.append((f"V2 at 0.25 of {den}", V2(q=0.25), den))
        c2specs.append((f"V5p (q_T 0.5) on {den}", V5(qT=0.5, form="participation"), den))
    rows = []
    for lab, r, den in specs + c2specs:
        el = national(P, den)
        tmax = st["t_bar"] / el.rho
        taus = np.round(np.arange(0.20, tmax + 1e-9, step), 2)
        for tm in tmins:
            for tau in taus:
                if el.t0(tau) <= tm + 1e-6 or el.t0(tau) > st["t_bar"]:
                    continue
                c = cell(r, el, float(tau), tm, st["pi_by_type"]["T1"], 0.5, st["family"], st["a_B"], P.behaviour.sigma_tau,
                         P.behaviour.Theta, recalibrate=st["recalibrate"], t_bar=st["t_bar"], t_min_b=st["t_min_b"])
                if c is None:
                    continue
                rows.append(dict(rule=lab, denominator=den, t_min=tm, tau=float(tau), regime=c["regime"], P_change=c["P_change"],
                                 C1p=c["C1p"], C1R=c["C1R"], C2=c["C2"], sx=c["sx"], sd=c["sd"]))
    report.write_csv(rows, "viii1_TR27_grid", P, scen, note=f"TR27: tau grid step {step} from 0.20 to the feasible maximum t_bar/rho; T1 prior")
    summ = []
    for lab, r, den in specs + c2specs:
        el = national(P, den)
        for tm in tmins:
            rr = [x for x in rows if x["rule"] == lab and x["t_min"] == tm]
            if not rr:
                continue
            unl = [x["tau"] for x in rr if not str(x["regime"]).startswith("lock")]
            bias = [x["tau"] for x in rr if x["C1p"] - x["C1R"] > BIAS_TOL]
            slack = [x["tau"] for x in rr if str(x["regime"]).startswith("slack")]
            boyc = [x["tau"] for x in rr if "boycott" in str(x["regime"])]
            exact = None
            if r.code in ("V3", "V6d"):
                tl = tau_lock(r, el, tm, prior=pr, recalibrate=st["recalibrate"], pi=st["pi_by_type"]["T1"], t_bar=st["t_bar"])
                exact = "locked at every tau" if tl is None else float(tl)
            mixed = [x["tau"] for x in rr if str(x["regime"]).startswith("no pure")]
            summ.append(dict(rule=lab, denominator=den, t_min=tm, tau_from=rr[0]["tau"], tau_to=rr[-1]["tau"],
                             lowest_unlocked_tau=min(unl) if unl else None, tau_lock_exact=exact,
                             challenger_bias_band=_runs(bias), challenger_bias_cells=len(bias),
                             lowest_slack_tau=min(slack) if slack else None, mixed_cells=_runs(mixed),
                             boycott_cells=len(boyc), max_C2=max(x["C2"] for x in rr),
                             tau_at_max_C2=max(rr, key=lambda x: x["C2"])["tau"]))
    report.write_csv(summ, "viii1_TR27_boundaries", P, scen,
                     note="TR27: lowest unlocked tau, exact lock boundary (tau_lock), challenger-biased band (C1' - C1^R > 0.005), "
                          "lowest slack tau, cells without a pure equilibrium, and C2 by rule and t_min")
    return summ


# ------------------------------------------------------------------ TR28
def run_TR28(P, scen, kmax=100000):
    Q = P.electorate.E_full / P.electorate.R / 3
    rows = []
    for r_bar in (0.03, 0.07, 0.30):
        for beta in (0.2, 0.5, 1.0):
            for L in (30, 45, 60):
                for gc in (0.0, 0.4, 0.6, 0.8):
                    for mu_hi, slab in ((1.0, "support to 1"), (0.875, "support ends at 0.875")):
                        tb, tm = 0.90, 0.30
                        k_cur_reach = k_cur_rand = k_v7_rand = None
                        for k in range(1, kmax + 1):
                            A = campaign.A_I(L, k, beta, r_bar=r_bar)
                            Ae = A + gc * (1 - A)
                            if k_cur_reach is None and (tm + (tb - tm) * Ae) * mu_hi < Q:
                                k_cur_reach = k
                            if k_cur_rand is None and tb * (0.5 + Ae * (mu_hi - 0.5)) < Q:
                                k_cur_rand = k
                            if k_v7_rand is None and 0.5 + Ae * (mu_hi - 0.5) < 0.55:
                                k_v7_rand = k
                            if k_cur_reach and k_cur_rand and k_v7_rand:
                                break
                        nv = "never"
                        rows.append(dict(r_bar=r_bar, beta_I=beta, L_days=L, gamma_c=gc, support=slab, t_bar=tb, t_min=tm, Q=Q,
                                         current_reach=k_cur_reach or nv, current_random_vote=k_cur_rand or nv,
                                         V1_reach=nv, V1_random_vote=nv, V7_reach=nv if mu_hi > 0.55 else 1,
                                         V7_random_vote=k_v7_rand or nv,
                                         A_eff_needed_V7_random=0.05 / (mu_hi - 0.5)))
    report.write_csv(rows, "viii1_TR28_complexity", P, scen,
                     note="TR28: smallest k at which change becomes infeasible. Reach branch (Corollary 7): undecided voters abstain, reach "
                          "t_min + (t_bar - t_min) A_eff; random-vote branch (sensitivity): every reachable voter votes, a share "
                          "A_eff = A_I + gamma_c (1 - A_I) by preference and the rest evenly, so the Yes share is 1/2 + A_eff (mu - 1/2). "
                          "V1 never becomes infeasible in either branch; V7 at 0.55 needs A_eff >= 0.05/(mu_hi - 1/2) in the random-vote branch")
    return rows


# ------------------------------------------------------------------ TR29 (v1.1.2)
ACCESSION_MEANS = (0.50, 0.60, 0.70, 0.80)


def run_TR29(P, scen, means=ACCESSION_MEANS):
    """w* (8.2) of each rule against V1 for T6 under illustrative accession priors (Appendix H, H.15).

    Same cells as TR23's accession block (Beta with the variance of Beta(a_B, a_B) and the stated mean, pi_T6 for kappa,
    E_full, t_min 0.30). None of these rules has an advisory tier, so the binding criteria equal C1' and C1^R and
    w* is computed at full precision exactly as in TR22 (weight_vs). Mean 0.50 reproduces T1's weights (T6 = T1 under D13).
    """
    st = scenario_settings(P)
    el = national(P, "E_full")
    rows = []
    for m in means:
        pr = beta_shift(pi=1.0, a_B=st["a_B"], mu_b=m)
        P_half = float(pr.mass(0.5, np.inf))
        for tau in TAUS + (TAU_ANCHOR,):
            cells = []
            for r in (V1(), V7(qS=0.55), V3(q=1 / 3), V6(q=0.25, variant="d"), V6(q=1 / 3, variant="d")):
                c = cell(r, el, tau, 0.30, st["pi_by_type"]["T6"], 0.5, st["family"], st["a_B"], P.behaviour.sigma_tau,
                         P.behaviour.Theta, recalibrate=st["recalibrate"], t_bar=st["t_bar"], t_min_b=st["t_min_b"], prior=pr)
                if c is not None:
                    cells.append(c)
            ref = [c for c in cells if c["code"] == "V1"][0]
            for c in cells:
                w, how = weight_vs(dict(C1p_b=ref["C1p"], C1R_b=ref["C1R"]), dict(C1p_b=c["C1p"], C1R_b=c["C1R"]))
                rows.append(dict(prior_mean=m, prior=pr.label, P_mu_gt_half=P_half, tau=tau, t_min=0.30, rule=c["rule"],
                                 code=c["code"], regime=c["regime"], P_change=c["P_change"], C1p=c["C1p"], C1R=c["C1R"],
                                 C1sum=c["C1p"] + c["C1R"], C2=c["C2"], w_star_vs_V1=w, reading=how))
    report.write_csv(rows, "viii2_TR29_T6_accession_weights", P, scen,
                     note="TR29 (Appendix H): indifference weight w* against V1 (loss w C1' + C1^R, full precision) for T6 under "
                          "illustrative accession priors (Beta with the variance of Beta(3,3); the author supplies the mean); "
                          "E_full, t_min 0.30; tau 0.394 = the 2025 in-country Assembly-election turnout (1,463,314 / 3,713,761)")
    return rows


def run_all_design(P, scen, n_draws=60):
    for name, fn in (("TR26", run_TR26), ("TR28", run_TR28), ("TR22", run_TR22), ("TR23", run_TR23), ("TR27", run_TR27),
                     ("TR25", run_TR25), ("TR24", run_TR24), ("TR21", run_TR21), ("TR29", run_TR29)):
        t = time.time()
        fn(P, scen)
        _log(f"{name} done in {time.time() - t:.0f} s")
