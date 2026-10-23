"""Tool runs TR30-TR32 for Section IX (refsim v1.1.3), scenario calibrated_2026.

TR30  the boundary between V1 and V7 at 0.55 (indifference weight w*, (8.2), full precision, loss w*C1' + C1^R):
      (a) T1 at each of TR11's 15 one-at-a-time variants (digest §34), with the weight above which current law
          (V3 at one third of E_full) is preferred to V1, and C1' and C1^R of V1 and V7;
      (b) at the centre, T3, T4 and T5 (T5 with the adopted amendment as default, as TR24: V7 = 'stands only if Yes
          reaches 55%'), and every municipality and qark on TR25's unit denominators;
      (c) T1 at the centre with unequal stakes Omega = B_x/B_d in {1/2, 1, 2} (Section III, Proposition 8).
TR31  voting from abroad:
      (a) TS9 (O9) with the minimum lead of 91 days (Pv12, Pv19(2)) beside 42 days: Pr_vote(12/24/48) and mean months
          for a valid national initiative, prompt and latest-date President, regular terms; and the delay in entry into
          force of a T5 amendment (adoption to polling) under TS0 and TS9 at 42 and 91 days;
      (b) V1 (with V7 at 0.55 and current law for reference), T1 at the centre, with non-resident turnout
          tau_ab in {0, 0.05, 0.11} (share of A = E_full - R) and their split mu_A in {0.3, 0.5, 0.7}; resident
          turnout held at the centre's t0; C1' and C1^R measured against the residents' majority.
TR32  the first-year cost of Pv61 for the 2026 initiatives: T1, Law B (V1 with P5's procedure: TS9, R3, substance
      mooting rule) with Pv61's switch to Part IX's validity rule for votes polled within 12 months of entry into force;
      entry into force 1 Jan 2027 / 1 Jul 2027 / 1 Jan 2028 (hypothetical); deposits 30 Oct 2026, 30 Nov 2026,
      31 Mar 2027; prompt and latest-date President; with and without the initiators' option to wait (Neni 61(3));
      leads of 42 and 91 days.
Every run writes outputs/<scenario>/ix_TR*.csv (+ .meta.json).
"""
from __future__ import annotations

import time

import numpy as np

from . import pipeline, report
from .calibrated import VARIANTS, _log
from .design import CUR, PROC, _pkg, weight_vs
from .electorate import Electorate, load_municipalities, national
from .evaluate import cell, evaluate_package, scenario_settings
from .rules import V1, V3, V4, V7

CRIT = ("regime", "P_change", "P_dec", "C1p", "C1R", "C2")


def _w(ref, x):
    """w* of x against ref (V1) on (C1', C1^R); full precision."""
    return weight_vs(dict(C1p_b=ref["C1p"], C1R_b=ref["C1R"]), dict(C1p_b=x["C1p"], C1R_b=x["C1R"]))


def _t1_pkgs():
    v1 = {T: {"code": "V1"} for T in ("T1", "T2", "T3", "T4", "T5", "T6")}
    v7 = {T: {"code": "V7", "qS": 0.55} for T in ("T1", "T2", "T3", "T4", "T6")}
    v7["T5"] = {"code": "V7", "qS": 0.45}          # T5: the amendment stays in force only if Yes reaches 55%
    cur = {T: {"code": "V3", "q": 1 / 3} for T in ("T1", "T2", "T3", "T4", "T6")}
    cur["T5"] = {"code": "V4", "q": 0.5}
    return {"V1": _pkg("V1", v1, CUR), "V7 0.55": _pkg("V7 0.55", v7, CUR), "current law": _pkg("current law", cur, CUR)}


def _row_triplet(rows_by_rule, extra):
    v1, v7, cl = rows_by_rule["V1"], rows_by_rule["V7 0.55"], rows_by_rule["current law"]
    w7, how7 = _w(v1, v7)
    wc, howc = _w(v1, cl)
    out = dict(extra)
    for lab, r in (("V1", v1), ("V7", v7), ("current", cl)):
        for k in CRIT:
            out[f"{lab}_{k}"] = r.get(k)
    out.update(w_star_V7_vs_V1=w7, reading_V7=how7, w_star_current_vs_V1=wc, reading_current=howc)
    return out


# ------------------------------------------------------------------ TR30
def run_TR30(P, scen, parts=("a", "b", "c")):
    pk = _t1_pkgs()
    rows_a, rows_b, rows_c, summ = [], [], [], []
    st = scenario_settings(P)
    # (a) T1 across the 15 one-at-a-time variants of TR11
    for vlab, kw, ov in (VARIANTS if "a" in parts else ()):
        rr = {name: evaluate_package(P, d, types=("T1",), overrides=ov, **kw)[0] for name, d in pk.items()}
        rows_a.append(_row_triplet(rr, dict(variant=vlab)))
    if "a" in parts:
      report.write_csv(rows_a, "ix_TR30a_T1_variants", P, scen,
                     note="TR30(a): T1 at each of TR11's 15 one-at-a-time variants (digest §34): V1, V7 at 0.55 and current law "
                          "(V3 at 1/3 of E_full); w* (8.2) against V1 at full precision, loss w C1' + C1^R")
    # (b) T3, T4, T5 at the centre
    rows_b = []
    for T in ("T3", "T4", "T5"):
        rr = {name: evaluate_package(P, d, types=(T,))[0] for name, d in pk.items()}
        rows_b.append(_row_triplet(rr, dict(level="national", unit=T, type=T)))
    # local units on TR25's unit denominators (resident turnout = the national centre's t0)
    st = scenario_settings(P)
    t0 = national(P, "E_full").t0(P.behaviour.tau_central)
    m = load_municipalities()
    R_now = 1.92e6
    units = [("municipality", m.names[i], float(m.E_hat[i]), float(m.R_hat(R_now)[i])) for i in range(len(m.names))]
    units += [("qark", k, float(v["E"]), float(v["R"])) for k, v in sorted(m.qarks(R_now).items())]
    pi_T = st["pi_by_type"].get("T7", st["pi_by_type"].get("T1", 0.25)) if st["pi_by_type"] else 0.0
    for level, name, Eu, Ru in units:
        el = Electorate(E=Eu, R=Ru, label=f"E_u {name}")
        rr = {}
        for lab, r in (("V1", V1()), ("V7 0.55", V7(qS=0.55)), ("current law", V3(q=1 / 3))):
            c = cell(r, el, el.tau(t0), P.behaviour.t_min_central, pi_T, 0.5, st["family"], st["a_B"], P.behaviour.sigma_tau,
                     P.behaviour.Theta, recalibrate=st["recalibrate"], t_bar=st["t_bar"], t_min_b=st["t_min_b"])
            rr[lab] = c if c is not None else dict(regime="infeasible", P_change=np.nan, P_dec=np.nan, C1p=np.nan, C1R=np.nan, C2=np.nan)
        rows_b.append(_row_triplet(rr, dict(level=level, unit=name, type="T7" if level == "municipality" else "T8",
                                            rho_u=Eu / Ru)))
    report.write_csv(rows_b, "ix_TR30b_types_units", P, scen,
                     note="TR30(b): w* of V7 at 0.55 and of current law against V1 at the calibrated centre for T3, T4 and T5 (T5: default = "
                          "the adopted amendment; V7 = 'stands only if Yes reaches 55%', i.e. rejection wins at 45%; current law V4 at 1/2 "
                          f"of E) and for every municipality and qark on unit denominators (TR25; resident turnout t0 = {t0:.3f}; "
                          "current law V3 at 1/3 of E_u)")
    summ = []
    for level in ("municipality", "qark"):
        rr = [r for r in rows_b if r.get("level") == level]
        w7 = np.array([r["w_star_V7_vs_V1"] for r in rr if r["w_star_V7_vs_V1"] is not None], float)
        wc = np.array([r["w_star_current_vs_V1"] for r in rr if r["w_star_current_vs_V1"] is not None], float)
        rhos = np.array([r["rho_u"] for r in rr])
        med = rr[int(np.argsort(rhos)[len(rhos) // 2])]
        summ.append(dict(level=level, units=len(rr), w7_n=len(w7), w7_median=float(np.median(w7)), w7_min=float(w7.min()),
                         w7_max=float(w7.max()), wcur_n=len(wc), wcur_median=float(np.median(wc)) if len(wc) else None,
                         wcur_min=float(wc.min()) if len(wc) else None, wcur_max=float(wc.max()) if len(wc) else None,
                         readings_V7=";".join(sorted(set(r["reading_V7"] for r in rr))),
                         readings_current=";".join(sorted(set(r["reading_current"] for r in rr))),
                         median_rho_unit=med["unit"], median_rho=med["rho_u"], median_rho_w7=med["w_star_V7_vs_V1"],
                         median_rho_wcur=med["w_star_current_vs_V1"]))
    report.write_csv(summ, "ix_TR30b_units_summary", P, scen,
                     note="TR30(b) summary over units: w* of V7 at 0.55 and of current law against V1 (median, min, max) and the unit with "
                          "the median register ratio")
    # (c) unequal stakes
    rows_c = []
    el = national(P, "E_full")
    tau = P.behaviour.tau_central
    for Om in (0.5, 1.0, 2.0):
        rr = {}
        for lab, r in (("V1", V1()), ("V7 0.55", V7(qS=0.55)), ("current law", V3(q=1 / 3))):
            rr[lab] = cell(r, el, tau, P.behaviour.t_min_central, st["pi_by_type"]["T1"], 0.5, st["family"], st["a_B"],
                           P.behaviour.sigma_tau, P.behaviour.Theta, recalibrate=st["recalibrate"], t_bar=st["t_bar"],
                           t_min_b=st["t_min_b"], Omega=Om)
        row = _row_triplet(rr, dict(Omega=Om))
        row.update(V1_sx=rr["V1"]["sx"], V1_sd=rr["V1"]["sd"], V1_mu_hat=rr["V1"]["mu_hat_eq"],
                   V7_sx=rr["V7 0.55"]["sx"], V7_sd=rr["V7 0.55"]["sd"], V7_mu_hat=rr["V7 0.55"]["mu_hat_eq"])
        rows_c.append(row)
    report.write_csv(rows_c, "ix_TR30c_unequal_stakes", P, scen,
                     note="TR30(c): T1 at the calibrated centre with stake ratio Omega = B_x/B_d (challenger payoff Omega Pr[change] - cost; "
                          "Section III, Proposition 8); criteria against the residents' majority; w* against V1")
    return rows_a, rows_b, rows_c, summ


# ------------------------------------------------------------------ TR31
def _t5_delay(option, president, extra_weeks):
    """Adoption-to-polling delay of a T5 amendment (Section V Part 1's Assembly chain: request 2 weeks, Court 9 weeks),
    exact over adoption weeks of one mandate (Const. 65(4): no adoption in the last 9 weeks), regular terms."""
    Pm = pipeline._mod()
    rng = np.random.default_rng(0)
    cal = Pm.Calendar(Pm.gen_timeline(0.0, rng), option, -Pm.WY * 60, Pm.WY * 80)
    gap = Pm.BASE["DEC"] + Pm.BASE["GMIN"] + extra_weeks
    d = []
    for D in range(Pm.CYC, 2 * Pm.CYC):
        if (D - Pm.W_FS) % Pm.CYC >= Pm.CYC - 9:
            continue
        c = D + 2 + 9
        tp = Pm.schedule(cal, c, option, gap, president)
        d.append(tp - D)
    d = np.array(d) * 12 / Pm.WY
    return dict(mean_months=float(d.mean()), min_months=float(d.min()), p10_months=float(np.percentile(d, 10)),
                p90_months=float(np.percentile(d, 90)), max_months=float(d.max()))


def _clearance_to_polling(option, president, extra_weeks):
    """Clearance (Court ruling) to polling, exact over clearance weeks of one mandate, regular terms."""
    Pm = pipeline._mod()
    rng = np.random.default_rng(0)
    cal = Pm.Calendar(Pm.gen_timeline(0.0, rng), option, -Pm.WY * 60, Pm.WY * 80)
    gap = Pm.BASE["DEC"] + Pm.BASE["GMIN"] + extra_weeks
    d = np.array([Pm.schedule(cal, c, option, gap, president) - c for c in range(Pm.CYC, 2 * Pm.CYC)])
    return dict(mean_months_clearance_to_polling=float(d.mean() * 12 / Pm.WY),
                max_months_clearance_to_polling=float(d.max() * 12 / Pm.WY),
                Pr_within_26_weeks_of_clearance=float(np.mean(d <= Pm.O9_MAXW)))


def run_TR31(P, scen):
    Pm = pipeline._mod()
    FR = pipeline.kqz_false_rejection("R3")
    v1c = evaluate_package(P, _t1_pkgs()["V1"], types=("T1",))[0]      # V1 at the centre: Pr_dec for C4 (F104)
    rows_a = []
    for lead, extra in ((42, 0), (91, 7)):
        for pres in ("earliest", "latest"):
            p = dict(Pm.BASE)
            p.update(FR=FR, president=pres)
            r = Pm.exact_option(p, "O9", TA_post=extra)
            t5 = _t5_delay("O9", pres, extra)
            cp = _clearance_to_polling("O9", pres, extra)
            rows_a.append(dict(**cp, timing="TS9 (O9)", lead_days=lead, president=pres, Pr_vote_12=r["le52"], Pr_vote_24=r["le104"],
                               Pr_vote_48=r["le208"], C4_V1_P5_T1=pipeline.c4(r["le104"], v1c["P_dec"], v1c["P_change"], "substance"),
                               mean_months_filing_to_polling=r["E_months"],
                               p99_deposit_to_polling_months=r["p99_dep_months"], polled=r["polled"],
                               T5_delay_mean_months=t5["mean_months"], T5_delay_min=t5["min_months"], T5_delay_p10=t5["p10_months"],
                               T5_delay_p90=t5["p90_months"], T5_delay_max=t5["max_months"]))
    for pres in ("earliest", "latest"):
        t5 = _t5_delay("O0", pres, 0)
        rows_a.append(dict(timing="TS0 (O0), reference", lead_days=42, president=pres, T5_delay_mean_months=t5["mean_months"],
                           T5_delay_min=t5["min_months"], T5_delay_p10=t5["p10_months"], T5_delay_p90=t5["p90_months"],
                           T5_delay_max=t5["max_months"]))
    report.write_csv(rows_a, "ix_TR31a_TS9_lead91", P, scen,
                     note="TR31(a): TS9 (O9: polling within 26 weeks of clearance where a lawful week exists) with the decree-to-polling lead "
                          "of 42 days (GMIN 6 weeks) or 91 days (13 weeks; Pv12, Pv19(2)); valid national initiative filed in a uniform week "
                          "of the 2025-2029 mandate (exact enumeration, Section V Part 1 chain, R3 verification); regular terms "
                          "(lambda_ee = 0). T5: adoption to polling (request 2 weeks, Court 9 weeks), exact over adoption weeks; "
                          "TS0 rows are the reference for F51. C4_V1_P5_T1 = Pr_vote(24) x Pr_dec of V1 at the centre (substance mooting "
                          "rule; F104's 0.995 at 42 days)")
    # (b) votes from abroad
    st = scenario_settings(P)
    e = P.electorate
    t0 = national(P, "E_full").t0(P.behaviour.tau_central)
    rows_b = []
    for tab in (0.0, 0.05, 0.11):
        for muA in (0.3, 0.5, 0.7):
            el = Electorate(E=float(e.E_full), R=float(e.R), tau_ab=tab, mu_A=muA, label="E_full")
            rr = {}
            for lab, r in (("V1", V1()), ("V7 0.55", V7(qS=0.55)), ("current law", V3(q=1 / 3))):
                rr[lab] = cell(r, el, el.tau(t0), P.behaviour.t_min_central, st["pi_by_type"]["T1"], 0.5, st["family"], st["a_B"],
                               P.behaviour.sigma_tau, P.behaviour.Theta, recalibrate=st["recalibrate"], t_bar=st["t_bar"],
                               t_min_b=st["t_min_b"])
            row = _row_triplet(rr, dict(tau_ab=tab, mu_A=muA, t0=t0, tau_E=el.tau(t0), abroad_votes_share=el.n_ab / (t0 + el.n_ab),
                                        b_ab=el.b_ab()))
            row.update(V1_mu_hat=rr["V1"]["mu_hat_eq"])
            rows_b.append(row)
    report.write_csv(rows_b, "ix_TR31b_abroad", P, scen,
                     note="TR31(b): T1 at the calibrated centre with votes from abroad (A1, (3.12)): tau_ab = turnout among registered "
                          "non-residents as a share of A = E_full - R (0.11 ~ 202,187 ballots in 2025 / 1.79M), mu_A = their share for the "
                          "change; resident turnout held at the centre's t0; C1' and C1^R against the residents' majority (mu > 1/2)")
    return rows_a, rows_b


# ------------------------------------------------------------------ TR32
DEPOSITS = {"30.10.2026": (2026, 303), "30.11.2026": (2026, 334), "31.03.2027": (2027, 90)}
EIF = {"01.01.2027": (2027, 1), "01.07.2027": (2027, 182), "01.01.2028": (2028, 1)}


def _week(yr, doy, Pm):
    return (yr - 2025) * Pm.WY + (doy - 1) // 7


def _wlabel(k, Pm):
    return f"{2025 + k // Pm.WY}-w{k % Pm.WY}"


def run_TR32(P, scen):
    Pm = pipeline._mod()
    rng = np.random.default_rng(0)
    cals = {o: Pm.Calendar(Pm.gen_timeline(0.0, rng), o, -Pm.WY * 60, Pm.WY * 80) for o in ("O0", "O9")}
    st = scenario_settings(P)
    FR0, FR3 = pipeline.kqz_false_rejection("R0"), pipeline.kqz_false_rejection("R3")
    # validity cells at the centre (T1)
    tau = P.behaviour.tau_central
    t0 = national(P, "E_full").t0(tau)
    vc = {}
    for lab, r, den in (("V1", V1(), "E_full"), ("V3 1/3 E_full", V3(q=1 / 3), "E_full"), ("V3 1/3 E_dom", V3(q=1 / 3), "E_dom")):
        el = national(P, den)
        vc[lab] = cell(r, el, el.tau(t0), P.behaviour.t_min_central, st["pi_by_type"]["T1"], 0.5, st["family"], st["a_B"],
                       P.behaviour.sigma_tau, P.behaviour.Theta, recalibrate=st["recalibrate"], t_bar=st["t_bar"], t_min_b=st["t_min_b"])
    p = dict(Pm.BASE)
    rows, paths = [], []
    for eif_lab, (ey, ed) in EIF.items():
        t_eif = _week(ey, ed, Pm)
        for dep_lab, (yr, doy) in DEPOSITS.items():
            dep = _week(yr, doy, Pm)
            if dep % Pm.WY > Pm.W_DEP_LAST:
                dep = (dep // Pm.WY + 1) * Pm.WY
            for lead, extra in ((42, 0), (91, 7)):
                for pres in ("earliest", "latest"):
                    for wait in (False, True):
                        for old_den in ("E_full", "E_dom"):
                            acc = dict(first_year=0.0, old_law=0.0, P_change=0.0, C1R=0.0, C4_opp=0.0, C4_fr=0.0, months=0.0,
                                       waited=0.0)
                            weeks = []
                            for cure, pc in ((1, p["p_cure"]), (0, 1 - p["p_cure"])):
                                kqz = dep + p["KQZ"] + (p["CURE"] if cure else 0)
                                FR = FR0 if kqz < t_eif else FR3
                                c = kqz + p["CC"]
                                decree = c + p["DEC"]
                                if decree < t_eif:            # decreed before entry into force: the old law throughout (Neni 61(2))
                                    tp = Pm.schedule(cals["O0"], c, "O0", p["DEC"] + p["GMIN"], pres)
                                    regime, mooting, waited, fy = "old law (decree before entry into force)", "none", False, True
                                    v = vc["V3 1/3 " + old_den]
                                else:                          # Law B for the stages not completed (Neni 60); Neni 18 from clearance
                                    gap = p["DEC"] + p["GMIN"] + extra
                                    tp = Pm.schedule(cals["O9"], c, "O9", gap, pres)
                                    fy = tp < t_eif + Pm.WY
                                    waited = False
                                    if fy and wait:            # Neni 61(3): the first lawful day a year after entry into force
                                        tp = max(t_eif + Pm.WY, c + gap)
                                        while not cals["O9"].feasible(tp):
                                            tp += 1
                                        fy, waited = False, True
                                    regime = "Law B, first year (Pv61: Part IX validity)" if fy else "Law B (V1)"
                                    mooting = "substance"
                                    v = vc["V3 1/3 " + old_den] if fy else vc["V1"]
                                pv = 1.0 - FR
                                c4o = pipeline.c4(pv, v["P_dec"], v["P_change"], mooting, 1.0)
                                c4f = pipeline.c4(pv, v["P_dec"], v["P_change"], mooting, 0.0)
                                acc["first_year"] += pc * (1.0 if (fy and not regime.startswith("old")) else 0.0)
                                acc["old_law"] += pc * (1.0 if regime.startswith("old") else 0.0)
                                acc["P_change"] += pc * v["P_change"]
                                acc["C1R"] += pc * v["C1R"]
                                acc["C4_opp"] += pc * c4o
                                acc["C4_fr"] += pc * c4f
                                acc["months"] += pc * (tp - dep) * 12 / Pm.WY
                                acc["waited"] += pc * (1.0 if waited else 0.0)
                                weeks.append(f"{_wlabel(tp, Pm)} ({regime}; cure {cure})")
                                paths.append(dict(entry_into_force=eif_lab, deposit=dep_lab, lead_days=lead, president=pres,
                                                  option_to_wait=wait, old_rule_denominator=old_den, cure=cure, prob=pc,
                                                  kqz_decision=_wlabel(kqz, Pm), verification="R0" if kqz < t_eif else "R3",
                                                  clearance=_wlabel(c, Pm), decree=_wlabel(decree, Pm), polling=_wlabel(tp, Pm),
                                                  regime=regime, months_deposit_to_polling=(tp - dep) * 12 / Pm.WY,
                                                  P_change=v["P_change"], C1R=v["C1R"], P_dec=v["P_dec"], C4_opposed=c4o,
                                                  C4_friendly=c4f))
                            rows.append(dict(entry_into_force=eif_lab, deposit=dep_lab, lead_days=lead, president=pres,
                                             option_to_wait=wait, old_rule_denominator=old_den,
                                             Pr_decreed_before_EIF_old_law=acc["old_law"], Pr_polled_first_year_LawB=acc["first_year"],
                                             Pr_PartIX_validity=acc["old_law"] + acc["first_year"],
                                             Pr_waited=acc["waited"], Pr_change=acc["P_change"], C1R=acc["C1R"],
                                             C4_opposed=acc["C4_opp"], C4_friendly=acc["C4_fr"],
                                             mean_months_deposit_to_polling=acc["months"], polling=" | ".join(weeks)))
    report.write_csv(rows, "ix_TR32_first_year", P, scen,
                     note="TR32: T1, valid list; statutory maximum stage times (KQZ 13 weeks, cure 7 weeks with probability 1/2, Court 9 "
                          "weeks, decree 5 weeks); a decree issued before entry into force keeps the old law throughout (TS0, Part IX "
                          "validity, no mooting rule; Neni 61(2)); otherwise Neni 18 (TS9) from clearance with Law B's substance mooting "
                          "rule, and Part IX's validity rule (V3 at 1/3 of E_full or E_dom) if polling falls within 52 weeks of entry into "
                          "force (Pv61); with the option, initiators ask for the first lawful week a year after entry into force "
                          "(Neni 61(3)). Verification R0 if the KQZ decides before entry into force, else R3 (false rejection at the "
                          "baseline list). C4 = (1 - false rejection) x Pr_dec under a substance rule, x max(0, Pr_dec - Pr_change) "
                          "against an opposed majority without one. Validity at the calibrated centre (resident turnout t0 = "
                          f"{t0:.3f}). Entry-into-force dates are hypothetical. Regular terms (lambda_ee = 0).")
    report.write_csv(paths, "ix_TR32_paths", P, scen, note="TR32: every path (cure branch) with its dates and regime")
    return rows


def run_all_robustness(P, scen):
    for name, fn in (("TR31", run_TR31), ("TR32", run_TR32), ("TR30", run_TR30)):
        t = time.time()
        fn(P, scen)
        _log(f"{name} done in {time.time() - t:.0f} s")
