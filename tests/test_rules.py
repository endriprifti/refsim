"""Rule modules: ex-post evaluation, default-monotonicity (Lemma 3) and agreement of the two views."""
import numpy as np
import pytest

from refsim.electorate import Electorate
from refsim.rules import V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, Votes, make_rule

E, R = 3713761.0, 1.92e6


def votes(Y, N, **kw):
    return Votes(Y=np.atleast_1d(float(Y)), N=np.atleast_1d(float(N)), E=E, R=R, T_ref=0.76 * R, **kw)


def test_v1():
    o, v = V1().evaluate(votes(10, 9))
    assert o[0] == "change" and v[0]


def test_v3_regime_a():
    bar = E / 3
    assert V3().evaluate(votes(bar + 1, 10))[0][0] == "change"
    o, v = V3().evaluate(votes(bar - 1, 10))
    assert o[0] == "default" and not v[0]          # Yes majority below one third of E: void (C1 event)
    o, v = V3().evaluate(votes(10, bar + 1))
    assert o[0] == "default" and v[0]               # default wins validly


def test_v4_regime_b_strict():
    assert V4().evaluate(votes(E / 2, 0))[0][0] == "default"      # "more than one half"
    assert V4().evaluate(votes(E / 2 + 1, 0))[0][0] == "change"


def test_v2_participation_and_v7():
    assert V2(q=0.4).evaluate(votes(0.3 * E, 0.09 * E))[0][0] == "default"
    assert V2(q=0.4).evaluate(votes(0.3 * E, 0.11 * E))[0][0] == "change"
    assert V7(qS=0.6).evaluate(votes(60, 40))[0][0] == "change"
    assert V7(qS=0.6).evaluate(votes(59, 41))[0][0] == "default"


def test_v6d_resident_indexed():
    assert V6(q=1 / 3, variant="d").evaluate(votes(R / 3 + 1, 0))[0][0] == "change"


def test_v9_tiers():
    r = V9(qa=0.2, qb=1 / 3, lam=0.5)
    assert r.evaluate(votes(0.25 * E, 0.1 * E))[0][0] == "advisory"
    assert r.evaluate(votes(0.34 * E, 0.1 * E))[0][0] == "change"
    assert r.evaluate(votes(0.15 * E, 0.1 * E))[0][0] == "default"


def test_v10_units():
    v = votes(60, 40, units_Y=np.array([[10, 10, 40]]), units_N=np.array([[20, 15, 5]]))
    assert V10(k_U=2, m_U=3).evaluate(v)[0][0] == "default"
    v = votes(60, 40, units_Y=np.array([[20, 20, 20]]), units_N=np.array([[10, 15, 15]]))
    assert V10(k_U=2, m_U=3).evaluate(v)[0][0] == "change"


DEFAULT_MONOTONE = [V1(), V3(), V4(), V5(form="approval"), V6(variant="d"), V7(qS=0.6), V8(), V9()]
NOT_MONOTONE = [V2(q=0.4), V5(form="participation"), V8(limb="participation"), V9(form="participation")]


@pytest.mark.parametrize("rule", DEFAULT_MONOTONE, ids=lambda r: r.code)
def test_lemma3_default_monotone(rule):
    """Adding default ballots never turns the default into change (Lemma 3)."""
    rng = np.random.default_rng(1)
    Y = rng.uniform(0, 0.6, 4000) * E
    N = rng.uniform(0, 0.6, 4000) * E
    o1, _ = rule.evaluate(Votes(Y=Y, N=N, E=E, R=R, T_ref=0.76 * R))
    o2, _ = rule.evaluate(Votes(Y=Y, N=N + 0.05 * E, E=E, R=R, T_ref=0.76 * R))
    assert not np.any((o1 == "default") & (o2 == "change"))
    assert rule.default_monotone


def _turnout_bar(rule):
    """Turnout (in votes) that the rule's participation limb requires."""
    if rule.code == "V5p":
        return rule.params["qT"] * 0.76 * R
    if rule.code == "V9":
        return rule.params["qb"] * E
    return rule.params["q"] * E


@pytest.mark.parametrize("rule", NOT_MONOTONE, ids=lambda r: r.code)
def test_lemma3_not_monotone(rule):
    """Participation-type rules: added default ballots can turn a void vote into change (a boycott pays)."""
    assert not rule.default_monotone
    B = _turnout_bar(rule)
    Y = np.array([0.80 * B])
    N = np.array([0.05 * B])
    o1, _ = rule.evaluate(Votes(Y=Y, N=N, E=E, R=R, T_ref=0.76 * R))
    o2, _ = rule.evaluate(Votes(Y=Y, N=N + 0.20 * B, E=E, R=R, T_ref=0.76 * R))
    assert o1[0] != "change" and o2[0] == "change"   # void or advisory -> binding change


@pytest.mark.parametrize("rule", [V1(), V2(q=0.4), V3(), V4(), V6(variant="d"), V7(qS=0.6), V8(), V9(), V5(form="approval")],
                         ids=lambda r: r.describe())
def test_interval_view_matches_counts(rule):
    """mu in the change set (equilibrium view) iff the counts give change (ex-post view)."""
    el = Electorate(E=E, R=R)
    rng = np.random.default_rng(3)
    n = 3000
    sx = rng.uniform(0.05, 1, n)
    sd = rng.uniform(0.05, 1, n)
    mu = rng.uniform(0, 1, n)
    inside = np.zeros(n, bool)
    for p in rule.intervals(sx, sd, el):
        if p.kind == "change":
            inside |= (mu >= p.a) & (mu < p.b)
    Y = mu * sx * R
    N = (1 - mu) * sd * R
    o, _ = rule.evaluate(Votes(Y=Y, N=N, E=E, R=R, T_ref=0.76 * R))
    agree = inside == (o == "change")
    # disagreements only on measure-zero boundaries (ties); allow none in a random sample
    assert agree.mean() > 0.999


def test_make_rule():
    assert make_rule("V6d").code == "V6d"
    assert make_rule({"code": "V11", "base": "V3", "lam": 0.4}).base.code == "V3"
    assert make_rule({"code": "V5p", "qT": 0.5}).code == "V5p"
