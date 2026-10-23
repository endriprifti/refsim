"""Drift (Specification module 11; Section IV Corollary 4, Section V Part 2 Proposition 29, Corollary 10; D11).

A29 deterministic path rho(t) = (E0 + gamma_E t) / (R0 e^{-d_R t}); lock year at fixed resident turnout t0
(rho >= rho_lock(t0)); inverse-Gaussian hitting times with ln rho a Brownian motion with drift gamma_rho
and volatility sigma: mean ln(rho_L/rho_0)/gamma, shape ln^2(rho_L/rho_0)/sigma^2 (5.36);
Pr[ever falls to rho_L < rho_0] = exp(-2 gamma ln(rho_0/rho_L)/sigma^2).
D11: d_R central 0.0085 (adults), fast 0.0116 (all residents); municipal proxies rebased.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
from scipy import stats

from .electorate import gamma_rho, load_municipalities, rho_path, t_cross
from .game import rho_lock


def lock_table(E0, R0, gamma_E, d_R, t0_grid=(0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90), t_min=0.20, pi=0.0,
               base=dt.date(2025, 4, 1)):
    rows = []
    for t0 in t0_grid:
        rl = rho_lock(t0, t_min, pi=pi)
        t = t_cross(rl, E0, R0, gamma_E, d_R)
        rows.append(dict(t0=t0, rho_lock=rl, years=t, year=int((base.year + (base.timetuple().tm_yday - 1) / 365.25) + t)))
    return rows


def ig_band(rho0, level, gamma, sigma, p=(0.05, 0.95)):
    """Inverse-Gaussian hitting-time quantiles (years) of rho_L = level from rho0 (Proposition 29)."""
    a = math.log(level / rho0)
    if a <= 0:
        return (0.0, 0.0)
    mean = a / gamma
    lam = a * a / (sigma * sigma)
    d = stats.invgauss(mean / lam, scale=lam)
    return tuple(float(d.ppf(q)) for q in p)


def pr_ever_falls(rho0, level, gamma, sigma):
    return math.exp(-2 * gamma * math.log(rho0 / level) / sigma ** 2)


def review_trigger(rho_hat, t0_hat, gamma_hat, t_min=0.20):
    """(5.37): review iff rho_hat >= rho_lock(t0_hat) * exp(-2 gamma_hat)."""
    rl = rho_lock(t0_hat, t_min)
    return dict(rho_lock=rl, threshold=rl * math.exp(-2 * gamma_hat), review=rho_hat >= rl * math.exp(-2 * gamma_hat))


def municipal_drift(gamma, years=(0, 2, 5, 10, 15, 20), R_now=1.92e6, rebase=True, q=1 / 3, map=None):
    m = load_municipalities(map=map)
    base = m.rho_u(R_now if rebase else None)
    counts = {y: int(np.sum(q * base * math.exp(gamma * y) > 1.0)) for y in years}
    joiners = sorted([(math.log(1 / (q * base[i])) / gamma, m.names[i]) for i in range(len(base))
                      if q * base[i] <= 1.0 and math.log(1 / (q * base[i])) / gamma <= 2.5])
    exits = sorted([(math.log(1.5 / base[i]) / gamma, m.names[i]) for i in range(len(base)) if base[i] < 1.5])
    return dict(counts=counts, joiners=joiners, remark1_exits=exits, basis="2025 (rebased, D11)" if rebase else "Table C.4 as printed")
