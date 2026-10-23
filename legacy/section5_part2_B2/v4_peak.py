"""Corollary 9(i) proof check [REVISED B2]: Beta(a, a) is ordered by peakedness.
(2X - 1)^2 ~ Beta(1/2, a) for X ~ Beta(a, a); f_a(1/2) = 2 Gamma(a+1/2)/(sqrt(pi) Gamma(a))
is increasing; hence S_a(mu) = Pr[X >= mu] decreases in a for every mu > 1/2."""
import numpy as np
from scipy import stats, special
grid_a = [0.5, 0.75, 1, 1.5, 2.5, 4, 8, 16]
v = np.linspace(0.01, 0.99, 99)
err = 0.0
for a in grid_a:
    # Pr[(2X-1)^2 <= v] = Pr[|2X-1| <= sqrt v] = F(1/2 + sqrt(v)/2) - F(1/2 - sqrt(v)/2)
    F = stats.beta(a, a).cdf
    lhs = F(0.5 + np.sqrt(v) / 2) - F(0.5 - np.sqrt(v) / 2)
    rhs = stats.beta(0.5, a).cdf(v)
    err = max(err, np.abs(lhs - rhs).max())
print('max |CDF((2X-1)^2) - Beta(1/2,a) CDF| over grid: %.1e' % err)
f12 = [2 * special.gamma(a + 0.5) / (np.sqrt(np.pi) * special.gamma(a)) for a in grid_a]
print('f_a(1/2) formula:', ['%.3f' % x for x in f12])
print('f_a(1/2) from pdf:', ['%.3f' % stats.beta(a, a).pdf(0.5) for a in grid_a])
for mu in (0.55, 0.603, 0.645, 0.7, 0.8):
    S = [stats.beta(a, a).sf(mu) for a in grid_a]
    print('mu=%.3f S_a decreasing in a: %s  %s' % (mu, all(np.diff(S) < 0), ['%.3f' % x for x in S]))
