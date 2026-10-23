"""Campaign and learning (Specification module 7; Section V Part 1, V.6, Proposition 24, Corollary 7).

c_I(t) = 1/2 + (1/pi) arctan sqrt(r_I t / k^beta)                (5.10)
A_I(L) = exp(-k^beta x* / (r_bar L_eff)),  x* = tan^2(pi (c* - 1/2))   (5.11), r_I ~ Exp(mean r_bar)
L*     = k^beta x* / (r_bar ln(1/A*))                            (5.12)
L_eff  = m_C L + m_C (m_I - 1) d_I                               (5.13)
Reach t_bar(L, k) = t_min + (1 - t_min) A_I(L)  (Corollary 7: a sensitivity; Section III keeps t_bar = 1).
Calibration (illustrative): a 30-day campaign informs 95% on a one-issue question; c* = 0.75.
Ported from legacy/section5_part1_B2/learning.py.
"""
from __future__ import annotations

import math

C_STAR = 0.75
X_STAR = math.tan(math.pi * (C_STAR - 0.5)) ** 2
L0, A0 = 30.0, 0.95
R_BAR = X_STAR / (L0 * math.log(1 / A0))
L_BLK = 5.0


def accuracy(r, t, k, beta):
    return 0.5 + math.atan(math.sqrt(r * t / k ** beta)) / math.pi


def A_I(L_eff, k, beta, r_bar=R_BAR):
    return math.exp(-k ** beta * X_STAR / (r_bar * L_eff))


def L_star(k, beta, A_target, r_bar=R_BAR):
    return k ** beta * X_STAR / (r_bar * math.log(1 / A_target))


def L_eff(L, m_C=1.0, m_I=1.0, d_I=0.0):
    return m_C * L + m_C * (m_I - 1) * d_I


def reach(L, k, beta, t_min=0.0):
    """t_bar(L, k) (5.14), sensitivity only."""
    return t_min + (1 - t_min) * A_I(L, k, beta)


def k_min(Q, pi, t_min, beta, L=30.0, mu_b=0.5, kmax=100000):
    """Smallest k at which change becomes infeasible: t_bar(L, k) mu_hi < Q (Corollary 7)."""
    mu_hi = pi * mu_b + 1 - pi
    for kk in range(1, kmax):
        if reach(L, kk, beta, t_min) * mu_hi < Q:
            return kk
    return None


def gamma_c(k, Q, pi=0.0, t_min=0.0, beta=1.0, L=30.0):
    """Cue-taking share needed to restore feasibility: (A* - A_I)/(1 - A_I), A* = (Q/mu_hi - t_min)/(1 - t_min)."""
    mu_hi = pi * 0.5 + 1 - pi
    Astar = (Q / mu_hi - t_min) / (1 - t_min)
    A = A_I(L, k, beta)
    return max(0.0, (Astar - A) / (1 - A))


def blackout_rho(L_eff_, Lb_eff, k, beta):
    x = R_BAR * L_eff_ / k ** beta
    xb = R_BAR * (L_eff_ - Lb_eff) / k ** beta
    return 1 - (xb / (1 + xb)) / (x / (1 + x))
