"""refsim v1.1: A36 floor split, calibrated prior families, calibrated scenario (Section VII Part 2)."""
import math

import pytest

from refsim.electorate import Electorate, national
from refsim.evaluate import evaluate_package
from refsim.game import Game, tau_lock
from refsim.params import Params
from refsim.priors import a_p_matched, beta_shift, make
from refsim.rules import V2, V3


def test_a36_floor_only_for_participation_rules():
    el = Electorate(E=1.934, R=1.0)
    g2 = Game(V2(q=0.40), el, tau=0.33, t_min=0.30, t_min_b=0.08)
    g3 = Game(V3(), el, tau=0.33, t_min=0.30, t_min_b=0.08)
    assert g2.floor_d == pytest.approx(0.08) and g2.floor_x == pytest.approx(0.30)
    assert g3.floor_d == pytest.approx(0.30)
    assert float(g2.cost(0.08)) == 0.0 and float(g2.cost(0.30)) == 0.0


def test_a36_off_reproduces_v10():
    el = Electorate(E=1.93, R=1.0)
    assert tau_lock(V3(), el, 0.2) == pytest.approx(tau_lock(V3(), el, 0.2, t_min_b=0.08), abs=1e-9)


def test_beta_shift_moments():
    p = beta_shift(pi=0.5, a_B=3.0, mu_b=2 / 3)
    m = p.mean()
    var = p.a * p.b / ((p.a + p.b) ** 2 * (p.a + p.b + 1))
    assert m == pytest.approx(0.5 + 0.5 * (2 / 3 - 0.5))
    assert var == pytest.approx(1 / (4 * 7))
    assert make("beta_shift", pi=0.25, a_B=3.0, mu_b=0.5).a == pytest.approx(3.0)


def test_zealot_variance_match_7_6():
    assert a_p_matched(3.014906638995449, 0.3) == pytest.approx(1.2223, abs=1e-4)
    assert 1 - math.sqrt(3 / (2 * 3.01 + 1)) == pytest.approx(0.346, abs=1e-3)


def test_calibrated_centre():
    P = Params.load("calibrated_2026")
    rows = {r["type"]: r for r in evaluate_package(P, "P0", types=("T1", "T4"))}
    assert rows["T1"]["regime"].startswith("lock") and rows["T4"]["regime"].startswith("lock")
    r1 = {r["type"]: r for r in evaluate_package(P, "P1", types=("T1",))}["T1"]
    assert r1["P_change"] == pytest.approx(0.613, abs=0.002)
    r3 = {r["type"]: r for r in evaluate_package(P, "P3c", types=("T1",))}["T1"]
    assert r3["regime"] == "lock (default boycott)" and r3["sd"] == pytest.approx(0.08)
