"""refsim v1.1.1: explicit priors in cell(), the T5 'Yes must reach 55%' rule, the binding-only screen,
noise on a resident denominator, and the window bounds of TR26."""
import math

import numpy as np
import pytest

from refsim import window
from refsim.design import binding_only, pareto, weight_vs
from refsim.electorate import national
from refsim.evaluate import cell, scenario_settings
from refsim.params import Params
from refsim.priors import beta_shift, make as make_prior
from refsim.rules import V1, V4, V6, V7


@pytest.fixture(scope="module")
def P():
    return Params.load("calibrated_2026")


def _cell(P, rule, el, tau, sig=0.04, prior=None):
    st = scenario_settings(P)
    return cell(rule, el, tau, 0.30, 0.25, 0.5, st["family"], st["a_B"], sig, 0.5, recalibrate=True, t_bar=st["t_bar"],
                t_min_b=st["t_min_b"], prior=prior)


def test_explicit_prior_equals_family(P):
    el = national(P, "E_full")
    st = scenario_settings(P)
    a = _cell(P, V6(q=1 / 3, variant="d"), el, 0.33)
    b = _cell(P, V6(q=1 / 3, variant="d"), el, 0.33, prior=make_prior(st["family"], pi=0.25, a_B=st["a_B"], mu_b=0.5))
    assert abs(a["P_change"] - b["P_change"]) < 1e-12 and abs(a["C1p"] - b["C1p"]) < 1e-12


def test_v7_045_mirrors_v7_055(P):
    el = national(P, "E_full")
    a = _cell(P, V7(qS=0.55), el, 0.33)
    b = _cell(P, V7(qS=0.45), el, 0.33)
    assert abs(a["P_change"] + b["P_change"] - 1.0) < 1e-6
    assert abs(a["C1p"] - b["C1R"]) < 1e-6 and abs(a["C1R"] - b["C1p"]) < 1e-6


def test_noise_scaled_on_R_makes_V4_on_R_equal_V6d(P):
    """V4 on residents at 1/3 with N > Y is V6d at 1/3 once the turnout noise is the same (sigma_tau x rho)."""
    rho = national(P, "E_full").rho
    elR = national(P, "R")
    a = _cell(P, V4(q=1 / 3), elR, 0.33 * rho, sig=0.04 * rho)
    b = _cell(P, V6(q=1 / 3, variant="d"), national(P, "E_full"), 0.33)
    assert abs(a["P_change"] - b["P_change"]) < 1e-6
    assert abs(a["C1R"] - b["C1R"]) < 1e-6


def test_binding_only_and_weights():
    r9 = dict(code="V9", P_change=0.0093, C1p=0.0, C1R=0.158)
    r11 = dict(code="V11", P_change=0.5, C1p=0.0286, C1R=0.0286)
    assert binding_only(r9, 0.5)["C1R_b"] == pytest.approx(0.5 - 0.0093)
    assert binding_only(r11, 0.5) == dict(P_bind=0.0, C1p_b=0.0, C1R_b=0.5)
    ref = dict(C1p_b=0.0228, C1R_b=0.0228)
    w, how = weight_vs(ref, dict(C1p_b=0.0014, C1R_b=0.0935))
    assert w == pytest.approx((0.0935 - 0.0228) / (0.0228 - 0.0014)) and how.endswith("w > w*")
    assert pareto({"a": (0.1, 0.0, 0.0), "b": (0.2, 0.0, 0.0), "c": (0.0, 0.1, 0.0)}) == ["a", "c"]


def test_window_bounds_match_F62():
    L = math.log(10)
    mx = [window.D_POST[y] / L for y in window.D_POST]
    assert min(mx) == pytest.approx(2.38, abs=0.01) and max(mx) == pytest.approx(3.27, abs=0.01)
    assert 4 / L == pytest.approx(1.737, abs=0.001)


def test_accession_prior_mean():
    pr = beta_shift(pi=1.0, a_B=3.0, mu_b=0.8)
    assert pr.mean() == pytest.approx(0.8)
    assert pr.a / (pr.a + pr.b) == pytest.approx(0.8)
