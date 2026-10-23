"""Priors on resident support for change (Specification module 4).

A2: mu = pi*mu_b + (1 - pi)*iota, iota ~ U[0,1]: uniform on [pi mu_b, pi mu_b + 1 - pi].
A12: general prior F_mu; kappa calibrated at f_mu(1/2) ("recalibrated") or held at the
     uniform calibration 1/(1 - pi) ("fixed").
Stress families: Beta(a_B, a_B) on [0,1] (matched to A2's variance, or half of it);
zealot family (5.27): iota ~ Beta(a_x, a_d), scaled to [pi mu_b, pi mu_b + 1 - pi] (needs A25').
"""
from __future__ import annotations

import numpy as np
from scipy import special, stats


class Prior:
    """Scaled Beta(a, b) on [lo, hi]; a = b = 1 gives the uniform prior of A2."""

    def __init__(self, a: float = 1.0, b: float | None = None, lo: float = 0.0, hi: float = 1.0, label: str = ""):
        self.a = float(a)
        self.b = float(a if b is None else b)
        self.lo, self.hi = float(lo), float(hi)
        self.label = label or (f"uniform[{lo:.3g},{hi:.3g}]" if self.a == 1 == self.b else f"Beta({self.a:g},{self.b:g})")
        self._d = stats.beta(self.a, self.b)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def is_uniform(self) -> bool:
        return self.a == 1.0 and self.b == 1.0

    def _x(self, mu):
        return np.clip((np.asarray(mu, float) - self.lo) / self.width, 0.0, 1.0)

    def cdf(self, mu):
        x = self._x(mu)
        if self.is_uniform:
            return x
        return special.betainc(self.a, self.b, x)

    def sf(self, mu):
        return 1.0 - self.cdf(mu)

    S = sf  # survival function S_mu (written F-bar in the book)

    def pdf(self, mu):
        mu = np.asarray(mu, float)
        x = (mu - self.lo) / self.width
        inside = (x > 0) & (x < 1)
        if self.is_uniform:
            return np.where(inside, 1.0 / self.width, 0.0)
        xs = np.clip(x, 1e-300, 1 - 1e-16)
        return np.where(inside, self._d.pdf(xs) / self.width, 0.0)

    f = pdf

    def mass(self, a, b):
        """Pr[a <= mu <= b], vectorised; empty when a >= b."""
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        return np.clip(self.cdf(b) - self.cdf(a), 0.0, None) * (b > a)

    def mean(self) -> float:
        return self.lo + self.width * self.a / (self.a + self.b)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.is_uniform:
            return self.lo + self.width * rng.random(n)
        return self.lo + self.width * rng.beta(self.a, self.b, n)

    def f_half(self) -> float:
        return float(self.pdf(0.5))


def uniform(pi: float = 0.0, mu_b: float = 0.5) -> Prior:
    lo = pi * mu_b
    return Prior(1.0, 1.0, lo, lo + 1.0 - pi, label=f"A2 uniform (pi={pi:g})")


def zealot(pi: float = 0.0, a_x: float = 1.0, a_d: float | None = None, mu_b: float = 0.5) -> Prior:
    """(5.27): scaled Beta(a_x, a_d) on [pi mu_b, pi mu_b + 1 - pi] (A25')."""
    lo = pi * mu_b
    a_d = a_x if a_d is None else a_d
    return Prior(a_x, a_d, lo, lo + 1.0 - pi, label=f"zealot Beta({a_x:g},{a_d:g}), pi={pi:g}")


def beta_matched(pi: float = 0.0, concentrate: float = 1.0) -> tuple[Prior, float]:
    """A12 stress family: Beta(a, a) on [0, 1] with A2's variance (1-pi)^2/12 divided by `concentrate`."""
    var = (1.0 - pi) ** 2 / 12.0 / concentrate
    a = (1.0 / (4.0 * var) - 1.0) / 2.0
    return Prior(a, a, 0.0, 1.0, label=f"matched Beta({a:.3f},{a:.3f})"), a


def a_p_matched(a_obs: float, pi: float) -> float:
    """(7.6): flexible-unit parameter of the zealot family matching the variance of Beta(a_obs, a_obs)."""
    return ((1.0 - pi) ** 2 * (2.0 * a_obs + 1.0) - 1.0) / 2.0


def beta_shift(pi: float = 0.0, a_B: float = 3.0, mu_b: float = 0.5) -> Prior:
    """Calibrated full-support family (v1.1, D13): Beta on [0, 1] with the variance of Beta(a_B, a_B)
    (the cross-question variance, Section VII Part 2) and the type mean 1/2 + pi (mu_b - 1/2) of A2."""
    m = 0.5 + pi * (mu_b - 0.5)
    v = 1.0 / (4.0 * (2.0 * a_B + 1.0))
    s = m * (1.0 - m) / v - 1.0
    return Prior(m * s, (1.0 - m) * s, 0.0, 1.0, label=f"Beta-shift(a_B={a_B:g}, mean={m:.3f})")


def make(family: str = "uniform", pi: float = 0.0, a_B: float = 1.0, mu_b: float = 0.5,
         concentrate: float = 1.0) -> Prior:
    if family == "uniform":
        return uniform(pi, mu_b)
    if family == "beta_shift":
        return beta_shift(pi, a_B, mu_b)
    if family == "zealot_vm":
        ap = a_p_matched(a_B, pi)
        if ap <= 0:
            raise ValueError(f"zealot_vm: a_p({a_B:g}, pi={pi:g}) = {ap:.3f} <= 0")
        return zealot(pi, ap, ap, mu_b)
    if family == "zealot":
        return zealot(pi, a_B, a_B, mu_b)
    if family == "beta":
        return Prior(a_B, a_B, 0.0, 1.0)
    if family == "matched":
        return beta_matched(pi, concentrate)[0]
    raise ValueError(family)


def kappa_calibration(prior: Prior, recalibrate: bool = True, pi: float = 0.0) -> float:
    """f_cal in the cost f_cal*(s - t_min)^2 / (8 t0 (t0 - t_min)) (3.4), in units of B.

    recalibrate=True: kappa set at f_mu(1/2) (A12); False: held at the uniform calibration 1/(1 - pi).
    """
    if recalibrate:
        return prior.f_half()
    return 1.0 / (1.0 - pi)


def f_half_beta(a: float) -> float:
    """Density of Beta(a, a) at 1/2: 2 Gamma(a + 1/2) / (sqrt(pi) Gamma(a)) (Corollary 9(i))."""
    return float(2.0 * special.gamma(a + 0.5) / (np.sqrt(np.pi) * special.gamma(a)))
