"""Mobilisation window (Specification module 10; Section V Part 2, V.5, Proposition 28, Remark 6).

A28: election-induced premium X(t) in resident turnout capacity, an OU process with
E[X(t_el + d)] = X_0 e^{-d/T_post} after, X_0 e^{-d/T_pre} before Assembly polling day.
(5.31) window test; (5.32) thresholds; (5.33) blocked share of the polling-day-anchored window;
(5.34) two-point bound on T_post; A5 floor path t_min(d_el) = t_min + (t_min^co - t_min) e^{-d_el/T_post}
with kappa held at the stand-alone calibration (the construction that reproduces Proposition 12).
"""
from __future__ import annotations

import math

import numpy as np


D_PRE = {2013: 3.55, 2017: 3.55, 2021: 1.54, 2025: 2.04}
D_POST = {2013: 5.55, 2017: 5.49, 2021: 7.52, 2025: 7.06}


def residual_share(d_months: float, T: float) -> float:
    return math.exp(-d_months / T) if T > 0 else 0.0


def window_complies(T_pre: float, T_post: float, d_pre: float, d_post: float, q_w=0.10) -> dict:
    """(5.32): T_pre <= d_pre/ln(1/q_w) and T_post <= d_post/ln(1/q_w)."""
    L = math.log(1 / q_w)
    return dict(pre=T_pre <= d_pre / L, post=T_post <= d_post / L, T_pre_max=d_pre / L, T_post_max=d_post / L)


def anchored_window_blocked_share(T_pre, T_post, q_w=0.10, cycle_months=48.0):
    """(5.33): (T_pre + T_post) ln(1/q_w) / 48."""
    return (T_pre + T_post) * math.log(1 / q_w) / cycle_months


def two_point_bound(t_L: float, t_25=0.4027, d=5.98, peak=0.76) -> float:
    """(5.34): T_post <= d / ln((peak - t_L)/(t_25 - t_L)); inf (no bound) if t_L >= t_25."""
    if t_L >= t_25:
        return float("inf")
    return d / math.log((peak - t_L) / (t_25 - t_L))


def floor_path(d_el_months, t_min=0.20, t_min_co=0.60, T_post=2.2):
    return t_min + (t_min_co - t_min) * np.exp(-np.asarray(d_el_months, float) / T_post)


def pi_star_fixed_kappa(Q, t0_sa, t_min_sa, t_min_now, t_bar=1.0, n=40001):
    """(3.9) with the cost held at the stand-alone calibration (A5): payoff l - (s - t_min_now)^2 /
    (8 t0_sa (t0_sa - t_min_sa)); floor t_min_now.  Ported from legacy v5_ou.py."""
    c = 1.0 / (8 * t0_sa * (t0_sa - t_min_sa))
    s_ = np.linspace(t_min_now + 1e-9, t_bar, n)
    thr = np.maximum(Q / s_, t_min_now / (s_ + t_min_now))
    j = int(np.argmax(-thr - c * (s_ - t_min_now) ** 2))
    return 2 * (1 - thr[j] - c * (s_[j] - t_min_now) ** 2)


def pi_star_at_distance(rho, tau_sa=0.30, t_min=0.20, t_min_co=0.60, T_post=2.2, q=1 / 3,
                        d_grid=(0, 1, 2, 3, 4, 5.5, 7.5, 12)):
    """Table V.24: lock bound pi* at distance d_el (months) from Assembly polling day (floor decays, kappa fixed)."""
    t0 = rho * tau_sa
    return [(d, max(0.0, pi_star_fixed_kappa(q * rho, t0, t_min, float(floor_path(d, t_min, t_min_co, T_post)))))
            for d in d_grid]
