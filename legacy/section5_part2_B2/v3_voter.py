"""V.3: voter model with zealots (A24). Exact checks and tables."""
import numpy as np
from scipy import stats, special

rng = np.random.default_rng(52)

# ---------------------------------------------------------------- (a) exact law
def bd_stationary(M, Zx, Zd, N):
    """Birth-death chain on n = flexible residents for change, n = 0..M."""
    b = lambda n: (M - n) * (Zx + n) / N
    d = lambda n: n * (Zd + M - n) / N
    logp = np.zeros(M + 1)
    for n in range(1, M + 1):
        logp[n] = logp[n - 1] + np.log(b(n - 1)) - np.log(d(n))
    p = np.exp(logp - logp.max())
    return p / p.sum()

for (M, Zx, Zd) in [(200, 3, 5), (500, 1, 1), (1000, 0.5 * 2, 4)]:
    N = M + Zx + Zd
    p = bd_stationary(M, Zx, Zd, N)
    q = stats.betabinom(M, Zx, Zd).pmf(np.arange(M + 1))
    print('Lemma 16(i) check M=%d Zx=%g Zd=%g: max|bd - betabinom| = %.2e' %
          (M, Zx, Zd, np.abs(p - q).max()))

# ---------------------------------------------------------------- (b) relaxation
def gillespie(M, Zx, Zd, rv, T, n0, dt_grid):
    N = M + Zx + Zd
    t, n = 0.0, n0
    out = np.empty(len(dt_grid)); k = 0
    while k < len(dt_grid):
        b = rv * (M - n) * (Zx + n) / N
        d = rv * n * (Zd + M - n) / N
        tot = b + d
        tau = rng.exponential(1 / tot)
        while k < len(dt_grid) and dt_grid[k] < t + tau:
            out[k] = (Zx + n) / N; k += 1
        t += tau
        n += 1 if rng.random() < b / tot else -1
    return out

M, Zx, Zd, rv = 400, 8, 8, 1.0
N = M + Zx + Zd
Trel = 1.0 / (rv * (Zx + Zd) / N)
grid = np.arange(0, 400 * Trel, 0.05 * Trel)
path = gillespie(M, Zx, Zd, rv, None, M // 2, grid)
x = path[len(path) // 10:] - path[len(path) // 10:].mean()
for lag_T in (0.5, 1.0, 2.0):
    L = int(round(lag_T / 0.05))
    ac = np.dot(x[:-L], x[L:]) / np.dot(x, x)
    print('autocorr at %.1f T_rel: sim %.3f  theory %.3f' % (lag_T, ac, np.exp(-lag_T)))
print('T_rel = R/(r_v*(Zx+Zd)) = %.2f (r_v units)' % Trel)

# ---------------------------------------------------------------- (c) literal model SD
R = 1.92e6
print('\nTable V.13: literal model, R = 1.92 million residents, o_x = o_d = o/2')
print('o_total   committed per side   SD(mu) stationary   central 90% band of mu')
for o in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.10, 0.30):
    Zs = R * o / 2
    Mf = R * (1 - o)
    a = b = Zs
    var_p = a * b * (a + b + Mf) / (Mf * (a + b) ** 2 * (a + b + 1))
    sd_mu = (Mf / R) * np.sqrt(var_p)
    # 90% band of p via beta approximation (exact for Mf -> inf)
    if Zs >= 0.5:
        lo, hi = stats.beta(a, b).ppf([0.05, 0.95])
    else:
        lo, hi = stats.beta(a, b).ppf([0.05, 0.95])
    mlo = o / 2 + (1 - o) * lo; mhi = o / 2 + (1 - o) * hi
    print('%-8g  %14.1f  %16.5f   [%.4f, %.4f]' % (o, Zs, sd_mu, mlo, mhi))

# ---------------------------------------------------------------- (d) bar table
print('\nTable V.14: the three-year bar against the opinion relaxation time')
print('T_rel  corr(3y)=e^{-3/T}  t(q=0.5) t(q=0.25) t(q=0.10)  SD change 3y (uniform) (Beta2.5)')
for T in (0.5, 1, 2, 3, 3/np.log(2), 5, 10, 20):
    c3 = np.exp(-3 / T)
    ts = [T * np.log(1 / q) for q in (0.5, 0.25, 0.10)]
    sd_u = np.sqrt(2 * (1 / 12) * (1 - c3))
    sd_b = np.sqrt(2 * (1 / (4 * 6)) * (1 - c3))
    print('%5.2f  %6.3f   %6.2f %6.2f %6.2f   %6.3f %6.3f' % (T, c3, *ts, sd_u, sd_b))
# T_rel at which c = 3 years exactly matches t(q)
for q in (0.5, 1/np.e, 0.25, 0.10):
    print('bar matches when T_rel = 3/ln(1/q) = %.2f years (q=%.3f)' % (3 / np.log(1 / q), q))

# ---------------------------------------------------------------- (e) Ochiai examples
def ochiai(P1, P2):
    P1, P2 = set(P1), set(P2)
    return len(P1 & P2) / np.sqrt(len(P1) * len(P2))
def jaccard(P1, P2):
    P1, P2 = set(P1), set(P2)
    return len(P1 & P2) / len(P1 | P2)
ex = {
 'identical repeal, 4 provisions': (range(4), range(4)),
 'narrower repeal, 2 of the 4': (range(4), range(2)),
 'wider repeal, 4 + 4 new': (range(4), range(8)),
 'half overlap, 4 and 4 sharing 2': (range(4), range(2, 6)),
 'one shared of 10 and 10': (range(10), range(9, 19)),
 'renumbered law, same content (mapped)': (range(6), range(6)),
}
print('\nTable V.15 inputs: Ochiai and Jaccard overlaps')
for k, (a, b) in ex.items():
    o = ochiai(a, b)
    print('%-40s Ochiai %.3f  Jaccard %.3f' % (k, o, jaccard(a, b)))
# months of bar justified: t* = T_rel ln(chi_txt / q_chi), for q_chi = 0.5
print('\nMonths after which a new question stops being predicted (q_chi=0.5):')
for T in (1, 2, 3, 5):
    row = []
    for o in (1.0, 0.707, 0.5, 0.25):
        t = T * np.log(o / 0.5) if o > 0.5 else 0.0
        row.append('%5.1f' % (12 * t))
    print('T_rel=%g:  chi_txt=1.0,0.707,0.5,0.25 ->' % T, ' '.join(row))

# ---------------------------------------------------------------- [REVISED B2]
# (f) counting unit: articles against paragraphs (granularity example for (5.23))
# earlier question repeals articles 10 (one paragraph) and 11 (four paragraphs);
# the new one repeals article 10 alone
q1_art, q2_art = {'a10', 'a11'}, {'a10'}
q1_par = {'a10.1', 'a11.1', 'a11.2', 'a11.3', 'a11.4'}; q2_par = {'a10.1'}
print('\nGranularity: Ochiai by article %.3f, by paragraph %.3f' %
      (ochiai(q1_art, q2_art), ochiai(q1_par, q2_par)))
# (g) literal reading with A2: one committed resident per side, T_rel = R/(r_v * a)
for T in (1, 5):
    print('A2 literal (a = R*o = 2): T_rel = %d y needs r_v = %.3f million revisions per resident-year' %
          (T, R / (2 * T) / 1e6))
