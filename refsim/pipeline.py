"""Pipeline (Specification module 5) and C4 (eq. (I.10), Remark 4).

Wraps the lifted absorbing chain of Section V Part 1 (legacy/section5_part1_B2/pipeline.py, exact
enumeration of Appendix B.43) for a valid national initiative filed in a uniform week:
Pr_vote(h) = Pr[polling day within h], expected months to polling, by timing scenario TS0-TS9
(code O0-O9), presidential behaviour (prompt 'earliest' / 'latest') and verification rule.

C4 combines the pipeline with the vote (Remark 4, F52):
  with a substance mooting rule or an indirect initiative:  C4 = Pr_vote * Pr_dec
  without one, a majority opposed on polling day (prob. Pr_opp) moots every initiative that would pass:
                                                            C4 = Pr_vote * (Pr_dec - Pr_opp * Pr_ch)
"""
from __future__ import annotations

from functools import lru_cache

from . import signatures
from ._legacy import load

TS_LABELS = {"O0": "TS0 current law", "O1": "TS1 election-day anchor", "O2": "TS2 shorter windows",
             "O3": "TS3 co-scheduling allowed", "O4": "TS4 co-scheduling required", "O5": "TS5 two slots (θ_const)",
             "O6": "TS6 corridor", "O7": "TS7 two corridors (θ_const)", "O8": "TS8 postal voting from abroad",
             "O9": "TS9 statutory maximum (repeal 119(6); polling within six months of clearance)"}


def _mod():
    return load("v1", "pipeline")


@lru_cache(maxsize=256)
def pr_vote(option: str = "O0", president: str = "earliest", FR: float = 0.0, W: int = 31,
            phi_f: float = 0.25, cutoff: bool = True):
    """Exact Pr_vote within 12/24/48 months and the mean (months) for a valid national initiative."""
    P = _mod()
    p = dict(P.BASE)
    p.update(FR=FR, president=president, W=W, phi_f=phi_f, cutoff=cutoff)
    r = P.exact_option(p, option)
    return dict(option=option, label=TS_LABELS.get(option, option), president=president,
                Pr_vote_12=r["le52"], Pr_vote_24=r["le104"], Pr_vote_48=r["le208"],
                mean_months=r["E_months"], polled=r["polled"])


def kqz_false_rejection(rule: str = "R0") -> float:
    """National list at a 5% margin and v_L = 0.85 (the Section V Part 1 baseline)."""
    return signatures.false_rejection_kqz(rule, 50000, 0.85, 0.05)


def c4(Pr_vote: float, Pr_dec: float, Pr_ch: float, mooting: str = "none", Pr_opp: float = 1.0) -> float:
    if mooting == "none":
        return Pr_vote * max(0.0, Pr_dec - Pr_opp * Pr_ch)
    return Pr_vote * Pr_dec
