"""refsim v1.1.3: TR30-TR32 for Section IX (robustness of the V1/V7 boundary, voting from abroad, Pv61's first year)."""
import pytest

from refsim import pipeline, robustness
from refsim.design import weight_vs
from refsim.electorate import Electorate, national
from refsim.evaluate import cell, scenario_settings
from refsim.params import Params
from refsim.rules import V1, V3, V7


@pytest.fixture(scope="module")
def P():
    return Params.load("calibrated_2026")


def _t1(P, rule, Omega=1.0, el=None):
    st = scenario_settings(P)
    el = el or national(P, "E_full")
    t0 = national(P, "E_full").t0(P.behaviour.tau_central)
    return cell(rule, el, el.tau(t0), P.behaviour.t_min_central, st["pi_by_type"]["T1"], 0.5, st["family"], st["a_B"],
                P.behaviour.sigma_tau, P.behaviour.Theta, recalibrate=st["recalibrate"], t_bar=st["t_bar"], t_min_b=st["t_min_b"],
                Omega=Omega)


def test_equal_stakes_reproduce_the_centre(P):
    v1, v7 = _t1(P, V1()), _t1(P, V7(qS=0.55))
    assert abs(v1["P_change"] - 0.5) < 1e-6 and round(v1["C1p"], 4) == 0.0228
    w, how = weight_vs(dict(C1p_b=v1["C1p"], C1R_b=v1["C1R"]), dict(C1p_b=v7["C1p"], C1R_b=v7["C1R"]))
    assert how == "preferred to V1 iff w > w*" and abs(w - 3.30561) < 1e-4          # digest §45


def test_stakes_tilt_V1_towards_the_higher_stake_side(P):
    hi, lo = _t1(P, V1(), Omega=2.0), _t1(P, V1(), Omega=0.5)
    assert hi["P_change"] > 0.5 > lo["P_change"]
    assert hi["mu_hat_eq"] < 0.5 < lo["mu_hat_eq"]


def test_no_abroad_votes_and_an_even_split_leave_V1_unchanged(P):
    e = P.electorate
    for tab, muA in ((0.0, 0.7), (0.11, 0.5)):
        el = Electorate(E=float(e.E_full), R=float(e.R), tau_ab=tab, mu_A=muA, label="E_full")
        c = _t1(P, V1(), el=el)
        assert abs(c["P_change"] - 0.5) < 1e-6 and abs(c["C1p"] - c["C1R"]) < 1e-6


def test_abroad_skew_moves_V1_with_the_abroad_majority(P):
    e = P.electorate
    up = _t1(P, V1(), el=Electorate(E=float(e.E_full), R=float(e.R), tau_ab=0.11, mu_A=0.7, label="E_full"))
    dn = _t1(P, V1(), el=Electorate(E=float(e.E_full), R=float(e.R), tau_ab=0.11, mu_A=0.3, label="E_full"))
    assert up["P_change"] > 0.5 > dn["P_change"] and up["C1p"] > up["C1R"] and dn["C1R"] > dn["C1p"]


def test_TS9_at_42_days_reproduces_F76_and_91_days_is_slower():
    Pm = pipeline._mod()
    p = dict(Pm.BASE, FR=pipeline.kqz_false_rejection("R3"), president="latest")
    r42, r91 = Pm.exact_option(p, "O9"), Pm.exact_option(p, "O9", TA_post=7)
    assert round(r42["le104"], 3) == 0.995
    assert r91["le104"] < r42["le104"] and r91["E_months"] > r42["E_months"]


def test_T5_delay_reference_matches_F51():
    d = robustness._t5_delay("O0", "earliest", 0)
    assert abs(d["mean_months"] - 7.7) < 0.15 and round(d["min_months"]) == 5 and round(d["max_months"]) == 14


def test_entry_into_force_2028_leaves_every_2026_deposit_under_the_old_law(P):
    rows = robustness.run_TR32(P, "calibrated_2026")
    late = [r for r in rows if r["entry_into_force"] == "01.01.2028"]
    assert late and all(abs(r["Pr_decreed_before_EIF_old_law"] - 1.0) < 1e-12 and r["Pr_change"] < 1e-6 for r in late)
    early_wait = [r for r in rows if r["entry_into_force"] == "01.01.2027" and r["option_to_wait"]]
    assert all(abs(r["Pr_change"] - 0.5) < 1e-4 for r in early_wait)
