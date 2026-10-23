"""Regression tests against the printed values of Sections III and IV (D10: rounded rho where they print it)."""
import math

import numpy as np
import pytest

from refsim import signatures
from refsim.electorate import Electorate, national
from refsim.game import Game, pi_star, pi_star_capped, rho_lock, tau_lock
from refsim.params import Params
from refsim.priors import Prior, beta_matched, uniform
from refsim.rules import V1, V2, V3, V4, V6, V7

EL193 = Electorate(E=1.93, R=1.0)
EL181 = Electorate(E=1.81, R=1.0)
RF, RD = 3713761 / 1.92e6, 3467826 / 1.92e6


@pytest.mark.parametrize("tm,book", [(0.0, 0.294), (0.20, 0.351), (0.35, 0.389)])
def test_table_III3_pistar(tm, book):
    assert pi_star(V3(), EL193, 0.40, tm) == pytest.approx(book, abs=6e-4)
    assert pi_star_capped(1.93, 0.40, tm) == pytest.approx(book, abs=6e-4)


@pytest.mark.parametrize("el,tm,row", [
    (EL193, 0.0, [0.0, 0.0, 0.357, 0.357]), (EL193, 0.2, [0.0, 0.284, 0.357, 0.357]), (EL193, 0.35, [0.0, 0.231, 0.357, 0.357]),
    (EL181, 0.0, [0.0, 0.0, 0.397, 0.397]), (EL181, 0.2, [0.0, 0.281, 0.397, 0.397]), (EL181, 0.35, [0.0, 0.220, 0.397, 0.397])])
def test_table_III2_prchange(el, tm, row):
    got = [Game(V3(), el, tau=t, t_min=tm).criteria()["P_change"] for t in (0.2, 0.3, 0.4, 0.5)]
    assert got == pytest.approx(row, abs=6e-4)


@pytest.mark.parametrize("el,vals", [(EL193, (0.3062, 0.2974, 0.2971)), (EL181, (0.3062, 0.2971, 0.2978))])
def test_lock_boundary(el, vals):
    for tm, v in zip((0.0, 0.2, 0.35), vals):
        assert tau_lock(V3(), el, tm) == pytest.approx(v, abs=2e-4)


def test_prop18_closed_form():
    for pi, v in ((0.0, 0.3062), (0.3, 0.3907)):
        assert (1 / 3) / (2 * ((2 - pi) / 3) ** 1.5) == pytest.approx(v, abs=1e-4)


def test_prop1_slack_and_prop4_v1():
    c = Game(V1(), EL193, tau=0.4, t_min=0.2).criteria()
    assert c["regime"].startswith("slack")
    assert c["C1"] == pytest.approx(0, abs=1e-12) and c["C1p"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("qS", [0.55, 0.6, 2 / 3])
def test_prop6_v7(qS):
    g = Game(V7(qS=qS), EL193, tau=0.4, t_min=0.0)
    c = g.criteria()
    assert c["sx"] == pytest.approx(math.sqrt(4 * qS * (1 - qS)) * g.t0, abs=1e-4)
    assert c["mu_hat_eq"] == pytest.approx(qS, abs=1e-4)
    assert c["C1"] == pytest.approx(qS - 0.5, abs=1e-4)


def test_prop19_regime_b_thresholds():
    el = Electorate(E=RD, R=1.0)
    assert Game(V4(), el, tau=0.5, t_min=0.410).campaign_gain() <= 1e-12
    assert Game(V4(), el, tau=0.5, t_min=0.422).campaign_gain() > 0
    assert Game(V4(), el, tau=0.514, t_min=0.35).campaign_gain() <= 1e-12
    assert Game(V4(), el, tau=0.520, t_min=0.35).campaign_gain() > 0


def test_f29_v6d_bias_point():
    el = Electorate(E=1.93, R=1.0)
    assert Game(V6(q=1 / 3, variant="d"), el, t0=0.465, t_min=0.0).criteria()["mu_hat_eq"] > 0.5
    assert Game(V6(q=1 / 3, variant="d"), el, t0=0.478, t_min=0.0).criteria()["mu_hat_eq"] < 0.5


def test_v2_boycott_c2_positive():
    """F19 / Lemma 3: a participation quorum gives the default side a gain from abstaining."""
    c = Game(V2(q=0.40), Electorate(E=1.93, R=1.0), tau=0.45, t_min=0.0).criteria()
    assert c["C2"] > 0.01


def test_f38_noise():
    for el, v in ((EL193, 0.669), (EL181, 0.755)):
        c = Game(V3(), el, tau=0.40, t_min=0.2).criteria(sigma_tau=0.04)
        assert (c["P_change"] - c["C1p"]) / 0.5 == pytest.approx(v, abs=1e-3)


def test_half_variance_beta_lock():
    """F39: half-variance Beta moves the lock to 0.40-0.46 (kappa recalibrated)."""
    pr, a = beta_matched(0.0, concentrate=2.0)
    assert a == pytest.approx(2.5)
    vals = [tau_lock(V3(), EL193, tm, prior=pr) for tm in (0.0, 0.2, 0.35)]
    assert vals == pytest.approx([0.462, 0.425, 0.405], abs=2e-3)


def test_corollary4_rho_lock():
    assert rho_lock(0.60, 0.2) == pytest.approx(2.016, abs=1e-3)
    assert rho_lock(0.75, 0.2) == pytest.approx(2.418, abs=1e-3)
    assert rho_lock(0.90, 0.2) == pytest.approx(2.619, abs=1e-3)


def test_signatures():
    assert signatures.worst_case("R3", 50000, 0.6, W=60)[0] == pytest.approx(0.0091, abs=1e-4)
    assert signatures.n_min(0.6) == 2559
    e, _, _ = signatures.rule_exact("R0", 50000, 0.0, 0.85)
    assert 0.45 < e < 0.55          # the 5% rule errs with probability about one half at the threshold


def test_electorate_values():
    P = Params.load()
    assert national(P, "E_full").rho == pytest.approx(1.9343, abs=1e-4)
    assert national(P, "E_dom").rho == pytest.approx(1.8062, abs=1e-4)
    assert national(P, "E_full").E - national(P, "E_dom").E == 245935
