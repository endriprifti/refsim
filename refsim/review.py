"""Review triggers (Specification module 12; Section V Part 2, Table V.28, (5.37); F66).

Indicators I1-I10 with their sources and thresholds; the trigger (5.37) review iff
rho_hat >= rho_lock(t0_hat) exp(-2 gamma_hat); even-year reporting (no national or local window under
regular terms); amendments of fundamental rules apply only to votes held at least one year after entry
into force, except amendments required by a Constitutional Court decision, and never to a referendum
already called by decree (Revised Code 2022, Guideline II.3.b; CDL-AD(2024)027).
"""
from __future__ import annotations

from .drift import review_trigger  # noqa: F401  (re-exported)

INDICATORS = [
    ("I1", "rho_full and rho_dom = E/R, R = resident adults", "KQZ final lists; INSTAT residents 18+ on 1 January",
     "(5.37); or a change of 0.10 or more since the last review; or a purge or change of denominator"),
    ("I2", "Stand-alone resident turnout t0_hat", "KQZ results of the latest referendum or minimum-competition vote, net of the residual premium (5.30)",
     "t0_hat/rho_hat within 0.03 of tau_lock"),
    ("I3", "Lock status of the rule in force", "this simulator at t0_hat, t_min in {0.20, 0.35}, pi in {0, 0.3}",
     "locked at the median observed stand-alone turnout"),
    ("I4", "Support needed for a 0.9 chance of passing, mu_0.9 (Lemma 12)", "computed from I1 and sigma_tau", "above 2/3"),
    ("I5", "Authentic-signature and duplicate rates", "KQZ verification records under R3",
     "authentic share below 0.60 in any list, or worst-case decision error above 0.01"),
    ("I6", "Municipalities with q_A rho_u > 1", "KQZ lists by municipality; INSTAT residents 18+ by municipality",
     "any, under a register-indexed local rule"),
    ("I7", "Municipalities crossing the 200,000-voter branch of Art. 132(1)", "KQZ lists by municipality", "any crossing"),
    ("I8", "Void votes with a Yes majority of valid votes (C1 events)", "KQZ result decisions", "any; review within six months"),
    ("I9", "Stage durations against statutory deadlines", "KQZ and Constitutional Court decisions",
     "any stage over its deadline in two consecutive cases"),
    ("I10", "Decay of mobilisation T_post_hat", "partial-election turnout by distance from the Assembly election",
     "T_post_hat above d_post/ln 10 for the shortest recent post-election buffer"),
]


def review_years(first=2026, last=2040):
    """Even years carry neither the national nor the local window under regular terms."""
    return [y for y in range(first, last + 1) if y % 2 == 0]
