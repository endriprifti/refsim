"""Regression tests for the Section VII additions (refsim v1.0.3)."""
import math

import numpy as np

from refsim.cases import GameCo, implied_mu
from refsim.electorate import Electorate, national
from refsim.game import Game, tau_lock
from refsim.params import Params
from refsim.priors import uniform
from refsim.rules import V1, V2, V8x, V11, Votes, make_rule


def test_V8x_interval_view_matches_counts():
    el = Electorate(E=1.9343, R=1.0)
    rng = np.random.default_rng(3)
    for r in (V8x(q_P=0.40, q_A=0.30), V8x(q_P=0.30, q_A=0.25), V8x(q_P=0.5, q_S=0.55),
              make_rule({"code": "V11", "base": {"code": "V8x", "q_P": 0.3, "q_A": 0.25}})):
        for _ in range(3000):
            sx, sd = rng.uniform(0, 1, 2)
            mu = rng.uniform()
            iv = r.intervals(sx, sd, el)
            inside = any(float(p.a) <= mu <= float(p.b) for p in iv if p.kind == "change")
            out, _ = r.evaluate(Votes(Y=np.array([mu * sx]), N=np.array([(1 - mu) * sd]), E=el.rho, R=1.0))
            near = min(abs(mu - float(p.a)) for p in iv) < 1e-9
            assert inside == (out[0] == "change") or near


def test_V8x_participation_only_equals_V2():
    P = Params.load("baseline_2025")
    el = national(P, "E_full")
    for tm in (0.0, 0.20, 0.35):
        assert abs(tau_lock(V8x(q_P=0.40), el, tm) - tau_lock(V2(q=0.40), el, tm)) < 1e-9


def test_GameCo_V1_rate():
    el = Electorate(E=1.0, R=1.0)
    t0_sa, tm_sa, t_co = 0.40, 0.20, 0.70
    g = GameCo(V1(), el, t0_sa, tm_sa, t_co, uniform(0.0))
    eq = g.solve()
    s = 0.5 * (t_co + math.sqrt(t_co ** 2 + 4 * t0_sa * (t0_sa - tm_sa)))
    assert abs(eq.sx - s) < 1e-6 and abs(eq.sd - s) < 1e-6


def test_implied_mu_inverts_the_yes_share():
    sx, sd, mu = 0.83, 0.12, 0.61
    y = mu * sx / (mu * sx + (1 - mu) * sd)
    assert abs(implied_mu(y, sx, sd) - mu) < 1e-12
