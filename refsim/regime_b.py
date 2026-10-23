"""Regime B (V4, T5) under general priors: Proposition 30 (F28 qualifier; Consistency check 2, Check 8).

With the reform side at its floor (Lemma 4(a)), the T5 opposition's payoff from rate s is
    S_mu(Q_R/s) - f_cal (s - t_min)^2 / (8 t0 (t0 - t_min)),
and its floor payoff is S_mu(Q_R/t_min) = 0 when Q_R > t_min.  At the cap s = t_bar it campaigns iff
    S_mu(Q_R/t_bar) / f_cal >= (t_bar - t_min)^2 / (8 t0 (t0 - t_min))            (4.4) applied to V4.
Proposition 30 (proof in docs/PROPOSITION30.md):
 (i)  for every symmetric unimodal prior on [0, 1] (pi = 0) and either kappa convention, the opposition's
      campaign gain at every s is at most its value under A2; so wherever F28 holds under A2
      (no campaign within D3), it holds under every symmetric unimodal prior;
 (ii) under U-shaped Beta(a, a) priors (a < 1) the clause can fail: the opposition campaigns at
      registered turnout above tau*(a) = [t_min + sqrt(t_min^2 + (t_bar - t_min)^2/(2K))] / (2 rho),
      K = S(Q_R/t_bar)/f_cal (cap condition), confirmed by the general best response.
"""
from __future__ import annotations

import math

from .electorate import Electorate
from .game import Game, tau_lock
from .priors import Prior, kappa_calibration
from .rules import V4


def cap_ratio(prior: Prior, rho: float, q_R=0.5, t_bar=1.0, recalibrate=True, pi=0.0) -> float:
    """K = S(Q_R/t_bar) / f_cal."""
    QR = q_R * rho
    return float(prior.sf(QR / t_bar)) / kappa_calibration(prior, recalibrate, pi)


def tau_threshold_cap(prior: Prior, rho: float, t_min: float, q_R=0.5, t_bar=1.0, recalibrate=True):
    """Registered turnout above which the opposition campaigns at the cap (None if never within t0 <= t_bar)."""
    K = cap_ratio(prior, rho, q_R, t_bar, recalibrate)
    if K <= 0:
        return None
    t0 = 0.5 * (t_min + math.sqrt(t_min ** 2 + (t_bar - t_min) ** 2 / (2 * K)))
    if t0 > t_bar:
        return None
    return t0 / rho


def tau_threshold_solver(prior: Prior, rho: float, t_min: float, q_R=0.5, recalibrate=True):
    """Same threshold from the general best response (any rate, not only the cap)."""
    el = Electorate(E=rho, R=1.0)
    return tau_lock(V4(q=q_R), el, t_min, prior=prior, recalibrate=recalibrate)


def dominance_check(prior: Prior, rho: float, t_min: float, tau: float, q_R=0.5, recalibrate=True, n=2001):
    """Proposition 30(i) numerically: max_s [gain under `prior` - gain under A2] <= 0 (units of f_cal)."""
    import numpy as np
    from .priors import uniform
    el = Electorate(E=rho, R=1.0)
    g = Game(V4(q=q_R), el, tau=tau, t_min=t_min, prior=prior, recalibrate=recalibrate)
    u = Game(V4(q=q_R), el, tau=tau, t_min=t_min, prior=uniform(0.0))
    s = np.linspace(t_min + 1e-6, 1.0, n)
    diff = g.u_x(s, t_min) / g.fcal - u.u_x(s, t_min) / u.fcal
    return float(np.max(diff))
