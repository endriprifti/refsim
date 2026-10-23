"""V10 double majority: per-voter power (Section III Part 2, Lemma 11, Table III.11; F36).

Penrose-Banzhaf power of a voter in unit u under V10 (national majority and a majority in k_U of m_U units),
relative to a simple national majority, under impartial culture with normal unit margins of variance
proportional to the unit electorate w_u:
    ratio_u = Pr[>= k_U units Yes | national tie] + sqrt(W/w_u) * Pr[unit u pivotal among units, national Yes]
(the national-channel term is exactly 1/2 for odd m_U and k_U = (m_U + 1)/2).
Ported from the Appendix B1 Part 2 check (b1_part2_check.py, function bz); Monte Carlo with a fixed seed.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def banzhaf_ratios(w, k, N=1_000_000, chunk=200_000, seed=12345):
    rng = np.random.default_rng(seed)
    w = np.asarray(w, float)
    m = len(w)
    W = w.sum()
    sd = np.sqrt(w)
    hits = 0
    for _ in range(N // chunk):
        Z = rng.standard_normal((chunk, m)) * sd
        Mm = Z - np.outer(Z.sum(1), w / W)
        hits += np.count_nonzero((Mm > 0).sum(1) >= k)
    cnat = hits / N
    p = np.zeros(m)
    for u in range(m):
        oth = [v for v in range(m) if v != u]
        h = 0
        for _ in range(N // chunk):
            Z = rng.standard_normal((chunk, m - 1)) * sd[oth]
            h += np.count_nonzero((Z.sum(1) > 0) & ((Z > 0).sum(1) == k - 1))
        p[u] = h / N
    ratio = cnat + np.sqrt(W / w) * p
    se = np.sqrt((W / w) * p * (1 - p) / N + cnat * (1 - cnat) / N)
    return dict(national_term=cnat, p=p, ratio=ratio, se=se, spread=float(ratio.max() / ratio.min()),
                share_below_1=float(w[ratio < 1].sum() / W))


def shapley_shubik(w, k=None):
    """Exact Shapley-Shubik index of units (feasible for m_U <= 14)."""
    w = np.asarray(w, float)
    n = len(w)
    W = w.sum()
    f = [math.factorial(i) for i in range(n + 1)]
    ss = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            for T in itertools.combinations(others, r):
                wt = w[list(T)].sum() if T else 0.0
                win0 = (wt > W / 2) and (k is None or r >= k)
                win1 = (wt + w[i] > W / 2) and (k is None or r + 1 >= k)
                if win1 and not win0:
                    ss[i] += f[r] * f[n - r - 1] / f[n]
    return ss
