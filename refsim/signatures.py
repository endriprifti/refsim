"""Signatures and verification (Specification module 8).

Verification rules (Section III Lemma 10, Section IV Proposition 20, Lemma 14, Remark 3; F26, F44):
  R0  current 5% random sample, projection against S (State Election Commissioner decision 2026, Art. 7(2))
  R1  California-style 3% sample (min 500): accept if projection >= 110% of S, reject below 95%, else count
  R3  band rule: n_smp = 1,500 (every line if J <= 3,000); accept >= (1+a)S, reject < (1-a)S, count between;
      statutory option (F44): a = 0.02 if >= 92% of sampled lines are authentic, else 0.05
  R4  sequential (Lemma 14): draws with replacement, Hoeffding boundary, n_seq = 6,000, alpha = 0.01
  R5  full count (no error by construction; workload J)
A list has V_L valid lines of J; the valid share v_L = V_L/J is held fixed as the margin varies.
Errors: false rejection FR (V_L >= S rejected) and false admission FA (V_L < S admitted; C4').
Exact hypergeometric computations; R4 by simulation with a fixed seed.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def setup(S: int, Delta: float, v: float):
    V = int(round(S * (1 + Delta)))
    J = int(round(S * (1 + Delta) / v))
    return V, J


def band_error(S, V, J, n, lo, hi):
    """Exact error of a band rule (accept if projection >= hi*S, reject if < lo*S, else count every line).
    lo = hi = 1 is a sample-only rule (R0).  Returns (error, Pr[full count])."""
    n = min(n, J)
    if n >= J:
        return 0.0, 1.0
    hg = stats.hypergeom(J, V, n)
    k_acc = math.ceil(hi * S * n / J - 1e-9)
    k_rej = math.ceil(lo * S * n / J - 1e-9) - 1
    p_acc, p_rej = float(hg.sf(k_acc - 1)), float(hg.cdf(k_rej))
    return (p_rej if V >= S else p_acc), max(0.0, 1 - p_acc - p_rej)


def n_R0(J, frac=0.05):
    return int(np.ceil(frac * J))


def n_R1(J):
    return max(500, int(np.ceil(0.03 * J)))


def rule_exact(rule: str, S: int, Delta: float, v: float, n_smp=1500, a=0.05, full_if=3000, frac=0.05):
    """(error, expected lines examined, Pr[full count]) for R0, R1, R3 (fixed band) or R5."""
    V, J = setup(S, Delta, v)
    if rule == "R5":
        return 0.0, J, 1.0
    if rule == "R3":
        if J <= full_if:
            return 0.0, J, 1.0
        n, lo, hi = n_smp, 1 - a, 1 + a
    elif rule == "R0":
        n, lo, hi = n_R0(J, frac), 1.0, 1.0
    elif rule == "R1":
        n, lo, hi = n_R1(J), 0.95, 1.10
    else:
        raise ValueError(rule)
    n = min(J, n)
    err, pfull = band_error(S, V, J, n, lo, hi)
    return err, n + pfull * (J - n), pfull


def r3_statutory(S: int, v: float, Delta: float, n_smp=1500, narrow=0.02, wide=0.05, thr=0.92, full_if=3000):
    """F44 statutory option: band +-narrow if the sampled authentic share >= thr, else +-wide.
    Returns (error, expected lines)."""
    V, J = setup(S, Delta, v)
    if J <= full_if:
        return 0.0, J
    n = min(n_smp, J)
    x = np.arange(0, n + 1)
    pmf = stats.hypergeom.pmf(x, J, V, n)
    a = np.where(x / n >= thr, narrow, wide)
    proj = J * x / n
    acc = proj >= (1 + a) * S
    rej = proj < (1 - a) * S
    band = ~(acc | rej)
    err = pmf[rej].sum() if V >= S else pmf[acc].sum()
    return float(err), float(n + J * pmf[band].sum())


def worst_case(rule: str, S: int, v: float, W=600, **kw):
    """Maximum error over every integer V_L in [S - W, S + W] at fixed v_L (supremum at V_L = S - 1 or S)."""
    best = (0.0, 0)
    for V in range(S - W, S + W + 1):
        e = rule_exact(rule, S, V / S - 1, v, **kw)[0]
        if e > best[0]:
            best = (e, V - S)
    return best


def seq_hoeffding(S, Delta, v, n_seq=6000, alpha=0.01, reps=2000, full_if=3000, seed=11):
    """R4 (Lemma 14): with replacement; accept if sum(1_i - S/J) >= Y, reject if <= -Y,
    Y = sqrt(n_seq ln(1/alpha)/2); undecided by n_seq -> full count.  Returns (error, lines, Pr[full])."""
    rng = np.random.default_rng(seed)
    V, J = setup(S, Delta, v)
    if J <= max(full_if, n_seq):
        return 0.0, J, 1.0
    p, c = V / J, S / J
    Y = np.sqrt(n_seq * np.log(1 / alpha) / 2)
    errs = works = fulls = 0
    for _ in range(reps):
        dev = np.cumsum((rng.random(n_seq) < p) - c)
        up = np.flatnonzero(dev >= Y)
        dn = np.flatnonzero(dev <= -Y)
        iu = up[0] if len(up) else None
        idn = dn[0] if len(dn) else None
        if iu is None and idn is None:
            fulls += 1
            works += J
            continue
        if idn is None or (iu is not None and iu < idn):
            works += iu + 1
            errs += V < S
        else:
            works += idn + 1
            errs += V >= S
    return errs / reps, works / reps, fulls / reps


def n_min(v_min: float, a=0.05, alpha=0.01) -> int:
    """(4.20): sample size for which the band rule's Hoeffding bound is at most alpha."""
    return int(math.ceil(math.log(1 / alpha) / (2 * a * a * v_min * v_min)))


def duplicates(J, dfrac, f):
    """Remark 3: expected duplicate pairs found within a sample (fraction f) and against the full list."""
    d = dfrac * J
    n = f * J
    lam1 = d * (n / J) * ((n - 1) / (J - 1))
    lam2 = d * (1 - (1 - n / J) * (1 - (n - 1) / (J - 1)))
    return dict(expected_within=lam1, P_any_within=1 - np.exp(-lam1), expected_vs_full=lam2)


def error_table(units=(("National", 50000), ("Korce", 11030), ("Pustec", 444)),
                v_grid=(0.70, 0.85), margins=(-0.10, -0.05, -0.02, 0.02, 0.05, 0.10)):
    """Rows: unit, v_L, margin, R0 error and lines, R3 (statutory) error and lines, R5 lines."""
    rows = []
    for unit, S in units:
        for v in v_grid:
            for D in margins:
                e0, w0, _ = rule_exact("R0", S, D, v)
                e3, w3 = r3_statutory(S, v, D)
                rows.append(dict(unit=unit, S=S, v_L=v, margin=D,
                                 kind="false admission" if D < 0 else "false rejection",
                                 R0_error=e0, R0_lines=w0, R3_error=e3, R3_lines=w3, R5_lines=setup(S, D, v)[1]))
    return rows


def false_rejection_kqz(rule="R0", S=50000, v=0.85, margin=0.05):
    """Probability that a list with S*(1+margin) valid lines is rejected (feeds the pipeline's KQZ stage)."""
    if rule == "R3":
        return r3_statutory(S, v, margin)[0]
    return rule_exact(rule, S, margin, v)[0]
