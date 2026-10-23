"""Opinion dynamics (Specification module 9; Section V Part 2, V.3-V.4).

Lemma 16 (mean-field voter model with zealots): stationary BetaBinomial, Corr(mu_0, mu_t) = e^{-t/T_rel},
T_rel = 1/(r_v o) = R/(r_v A_B).
Proposition 25: the bar c exceeds the fidelity horizon iff c > T_rel ln(1/q_chi)                  (5.20)
Lemma 17 / (5.22)-(5.23): Ochiai overlap of the paragraphs at stake; same issue iff same kind of legal
effect and overlap > 1/2 (by paragraph; limb (b) a rebuttable presumption).
Proposition 26: Jacobi diffusion in units of T_rel with stationary Beta(a_Bx, a_Bd)               (5.25)-(5.27)
Proposition 27: mean first passage to b                                                           (5.28)-(5.29)
Exact transition laws by the finite-volume solver of B.51 (legacy/section5_part2_B2/jacobi_fv.py).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, special

from ._legacy import load


def fidelity_horizon(T_rel: float, q_chi: float) -> float:
    """Years over which a past result predicts opinion above q_chi: T_rel ln(1/q_chi)."""
    return T_rel * math.log(1 / q_chi)


def bar_exceeds_fidelity(c_years: float, T_rel: float, q_chi: float) -> bool:
    return c_years > fidelity_horizon(T_rel, q_chi)


def critical_T_rel(c_years: float, q_chi: float) -> float:
    return c_years / math.log(1 / q_chi)


def ochiai(P1: set, P2: set) -> float:
    if not P1 or not P2:
        return 0.0
    return len(P1 & P2) / math.sqrt(len(P1) * len(P2))


def same_issue(P1: set, P2: set, same_effect: bool, threshold=0.5) -> bool:
    """(5.23): same kind of legal effect and Ochiai overlap > 1/2 of the paragraphs at stake."""
    return bool(same_effect and ochiai(P1, P2) > threshold)


def predicted_years(chi_txt: float, T_rel: float, q_chi: float) -> float:
    """Lemma 17: Q2 is predicted above q_chi for T_rel ln(chi_txt/q_chi)^+ years."""
    return T_rel * max(0.0, math.log(chi_txt / q_chi)) if chi_txt > 0 else 0.0


def voter_model_T_rel(R: float, r_v: float, A_B: float) -> float:
    return R / (r_v * A_B)


def r_v_needed(R: float, A_B: float, T_rel: float) -> float:
    """Revisions per resident-year needed for a given T_rel (Lemma 16: 0.96M at T_rel = 1 for A_B = 2)."""
    return R / (A_B * T_rel)


def mean_first_passage(a_x: float, a_d: float, b: float, x: float = 0.5) -> float:
    """(5.28) in units of T_rel: E[T_b | p0 = x] = A int_x^b y^-a_x (1-y)^-a_d Beta(a_x,a_d) I_y(a_x,a_d) dy."""
    A = a_x + a_d
    B = special.beta(a_x, a_d)
    f = lambda y: A * y ** (-a_x) * (1 - y) ** (-a_d) * B * special.betainc(a_x, a_d, y)
    return float(integrate.quad(f, x, b, limit=200)[0])


def d_first_passage_d_rho(a_x: float, a_d: float, b: float, q_A=1 / 3, t_bar=1.0) -> float:
    """(5.29): derivative of (5.28) in rho through b = q_A rho / t_bar."""
    A = a_x + a_d
    return (q_A / t_bar) * A * b ** (-a_x) * (1 - b) ** (-a_d) * special.beta(a_x, a_d) * special.betainc(a_x, a_d, b)


def jacobi(a_x: float, a_d: float, N: int = 8000):
    """Finite-volume Jacobi solver (B.51), exact up to discretisation; see legacy jacobi_fv.py."""
    return load("v2", "jacobi_fv").Jacobi(a_x, a_d, N=N)


def prob_reach_within(a_x, a_d, x0, b, t_years, T_rel, N=4000, dt=1e-3):
    """Pr[T_b <= t | p0 = x0] for a fixed barrier b (Table V.21 first columns)."""
    jf = load("v2", "jacobi_fv")
    tg = np.array([t_years / T_rel])
    surv = jf.survival_curve(a_x, a_d, x0, tg, lambda t: b, N=N, dt=dt)
    return float(1 - np.asarray(surv)[-1])
