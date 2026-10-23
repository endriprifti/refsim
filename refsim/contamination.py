"""Plebiscitary contamination C3 (eq. (I.9); Section III Part 2, Proposition 16; A9).

Voters weigh the government (approval mu_G) with weight Psi: the challenger wins iff
(1 - Psi) mu + Psi mu_G^x >= mu_hat, i.e. mu >= mu_hat_C = (mu_hat - Psi mu_G^x)/(1 - Psi).
C3 = Pr[mu between mu_hat and mu_hat_C] (= Psi |mu_hat - mu_G^x| / ((1 - Psi)(1 - pi)) under A2).
mu_G^x = mu_G when the challenger is the government's side, 1 - mu_G when it is the opposition's.
"""
from __future__ import annotations

from .priors import Prior


def mu_hat_C(mu_hat: float, Psi: float, mu_Gx: float) -> float:
    return (mu_hat - Psi * mu_Gx) / (1 - Psi)


def C3(prior: Prior, mu_hat: float, Psi: float, mu_Gx: float) -> float:
    if Psi <= 0:
        return 0.0
    m = mu_hat_C(mu_hat, Psi, mu_Gx)
    lo, hi = min(m, mu_hat), max(m, mu_hat)
    return float(prior.mass(lo, hi))
