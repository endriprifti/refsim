"""Monte Carlo layer: ex-post rule evaluation on simulated counts agrees with the exact quadrature."""
import pytest

from refsim.electorate import Electorate
from refsim.game import Game
from refsim.rules import V1, V2, V3, V6, V7
from refsim.simulate import convergence

EL = Electorate(E=3713761 / 1.92e6, R=1.0)


@pytest.mark.parametrize("rule,tau,tm", [(V3(), 0.40, 0.2), (V3(), 0.30, 0.2), (V1(), 0.40, 0.2), (V2(q=0.4), 0.45, 0.0),
                                         (V6(variant="d"), 0.35, 0.2), (V7(qS=0.6), 0.40, 0.0)],
                         ids=lambda x: getattr(x, "code", str(x)))
def test_mc_matches_exact(rule, tau, tm):
    g = Game(rule, EL, tau=tau, t_min=tm)
    cv = convergence(g, g.solve(), n=20000, sigma_tau=0.04, seed=11)
    assert cv["max_z"] < 4.0          # within four Monte Carlo standard errors
    assert cv["max_half_diff"] < 0.02
