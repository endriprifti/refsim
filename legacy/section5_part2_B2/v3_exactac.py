"""Exact stationary autocorrelation of the finite voter model with zealots,
from the generator matrix (check of Lemma 16(ii))."""
import numpy as np
from scipy.linalg import expm
from scipy import stats
for (M, Zx, Zd) in [(400, 8, 8), (300, 2, 6), (200, 1, 1)]:
    N = M + Zx + Zd; rv = 1.0
    G = np.zeros((M + 1, M + 1))
    for n in range(M + 1):
        b = rv * (M - n) * (Zx + n) / N
        d = rv * n * (Zd + M - n) / N
        if n < M: G[n, n + 1] = b
        if n > 0: G[n, n - 1] = d
        G[n, n] = -(b + d)
    pi = stats.betabinom(M, Zx, Zd).pmf(np.arange(M + 1))
    mu = (Zx + np.arange(M + 1)) / N
    m = pi @ mu; v = pi @ (mu - m) ** 2
    T = N / (rv * (Zx + Zd))
    out = []
    for k in (0.5, 1.0, 2.0):
        P = expm(G * k * T)
        cov = (pi * (mu - m)) @ (P @ (mu - m))
        out.append('lag %.1f T_rel: %.6f vs %.6f' % (k, cov / v, np.exp(-k)))
    # stationarity check
    print('M=%d Zx=%g Zd=%g |pi G| max = %.1e;' % (M, Zx, Zd, np.abs(pi @ G).max()), '; '.join(out))
