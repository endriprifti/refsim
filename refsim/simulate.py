"""Monte Carlo layer: ex-post rule evaluation on simulated vote counts.

For an equilibrium of the noiseless game (strategies as in Section IV), draw
  mu ~ prior;  (Z_x, Z_d) ~ BVN(0, 1, Theta);  s~_j = min{t_bar, max{0, s_j + rho sigma_tau Z_j}}
  (a zero rate stays zero);  Y = (mu s~_x + mu_A n_ab) R,  N = ((1 - mu) s~_d + (1 - mu_A) n_ab) R,
and evaluate the rule on the counts through its common interface (rules.Rule.evaluate).
Estimates carry Wilson 95% intervals; `convergence` compares them with the exact quadrature of game.py.
V10 adds unit heterogeneity: mu_u = clip(mu + sigma_u eps_u), unit electorates from Table C.4 (qarks).
"""
from __future__ import annotations

import math

import numpy as np

from .game import Equilibrium, Game
from .rules import V10, Votes


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def simulate(game: Game, eq: Equilibrium, n: int = 20000, seed: int = 20261012, sigma_tau: float = 0.04,
             Theta: float = 0.5, unit_weights=None, unit_rho=None, sigma_u: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    el, pr = game.el, game.prior
    mu = pr.sample(rng, n)
    sig = el.rho * sigma_tau
    if eq.pure:
        sx = np.full(n, eq.sx)
        sd = np.full(n, eq.sd)
    else:
        sx = rng.choice(eq.grid, size=n, p=eq.mix_x)
        sd = rng.choice(eq.grid_d if eq.grid_d is not None else eq.grid, size=n, p=eq.mix_d)
    Z1 = rng.standard_normal(n)
    Z2 = Theta * Z1 + math.sqrt(max(1 - Theta ** 2, 0)) * rng.standard_normal(n)
    X = np.where(sx > 0, np.clip(sx + sig * Z1, 0, game.t_bar), 0.0)
    Dd = np.where(sd > 0, np.clip(sd + sig * Z2, 0, game.t_bar), 0.0)
    R = 1.0e6  # counts in units of a million residents (rules are scale-free)
    E = el.rho * R
    nab = el.n_ab
    Y = (mu * X + el.mu_A * nab) * R
    N = ((1 - mu) * Dd + (1 - el.mu_A) * nab) * R
    v = Votes(Y=Y, N=N, E=E, R=R, T_ref=el.tauR_ref * R)
    if isinstance(game.rule, V10) and unit_weights is not None:
        w = np.asarray(unit_weights, float)
        w = w / w.sum()
        m = len(w)
        mu_u = np.clip(mu[:, None] + sigma_u * rng.standard_normal((n, m)), 0, 1)
        Ru = R * w
        v.units_Y = mu_u * X[:, None] * Ru
        v.units_N = (1 - mu_u) * Dd[:, None] * Ru
        v.Y = v.units_Y.sum(1)
        v.N = v.units_N.sum(1)
        Y, N = v.Y, v.N
    out, valid = game.rule.evaluate(v)
    change = out == "change"
    adv = out == "advisory"
    default = out == "default"
    maj = Y > N
    res = {}
    for key, ind in (("P_change", change), ("P_advisory", adv), ("P_dec", valid | change | adv),
                     ("C1", maj & default), ("C1p", change & (mu < 0.5)), ("C1R", default & (mu > 0.5))):
        k = int(ind.sum())
        lo, hi = wilson(k, n)
        res[key] = k / n
        res[key + "_lo"] = lo
        res[key + "_hi"] = hi
    res["turnout_E"] = float(np.mean((Y + N) / E))
    res["n"] = n
    return res


def convergence(game: Game, eq: Equilibrium, n: int = 20000, sigma_tau=0.04, Theta=0.5, seed=20261012) -> dict:
    """Monte Carlo against the exact quadrature: max |difference| in units of the MC standard error,
    and the difference between the n/2 and n runs."""
    exact = game.criteria(eq, sigma_tau=sigma_tau, Theta=Theta)
    full = simulate(game, eq, n, seed, sigma_tau, Theta)
    half = simulate(game, eq, n // 2, seed + 1, sigma_tau, Theta)
    worst_z = 0.0
    worst_half = 0.0
    for k in ("P_change", "P_dec", "C1", "C1p", "C1R"):
        p = full[k]
        se = math.sqrt((p * (1 - p) + 1.0 / n) / n)   # floored at the 1/n resolution of the estimate
        worst_z = max(worst_z, abs(p - exact[k]) / se)
        worst_half = max(worst_half, abs(p - half[k]))
    return dict(max_z=worst_z, max_half_diff=worst_half, exact=exact, mc=full)
