"""V.7: long-run drift of E, R and rho; hitting times and municipal counts."""
import numpy as np
from scipy import stats, optimize
from game import Prior, campaigns, tau_lock, RHO_FULL, RHO_DOM

E0, GAMMA_E = 3713761.0, 31200.0
EDOM0 = 3467826.0
R0, D_R = 1.92e6, 0.0116
t = np.arange(0, 31)

def rho_det(t, E=E0):
    return (E + GAMMA_E * t) / (R0 * np.exp(-D_R * t))

print('A29 deterministic path: gamma_rho(t) = gamma_E/E_t + d_R')
for tt in (0, 5, 10, 20, 30):
    g = GAMMA_E / (E0 + GAMMA_E * tt) + D_R
    print(' t=%2d (%d): rho_full %.3f rho_dom %.3f gamma_rho %.4f ; const-rate rho_full %.3f' %
          (tt, 2025 + tt, rho_det(tt), rho_det(tt, EDOM0), g, RHO_FULL * np.exp(0.02 * tt)))

def t_cross(level, E=E0):
    if rho_det(0, E) >= level:
        return 0.0
    return optimize.brentq(lambda x: rho_det(x, E) - level, 0, 200)

# ----------------------------------------------------- rho_lock(t0)
def rho_lock(t0, tmin, pi=0.0):
    pr_fn = lambda: Prior(1.0, lo=pi / 2, hi=1 - pi / 2)
    def camp(rho):
        return campaigns(rho / 3, t0, tmin, pr_fn(), 1.0 / (1 - pi))[0]
    lo, hi = 1.0, 3.0
    if not camp(lo):
        return None
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if camp(m): lo = m
        else: hi = m
    return lo

print('\nTable V.25: rho_lock at fixed resident turnout t0 (pi = 0) and crossing years')
print('t0     tmin=0      tmin=0.20   tmin=0.35  | crossing (tmin=0.20): A29 path / const 2.0% / 1.5% / 2.5%')
RL = {}
for t0 in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90):
    vals = [rho_lock(t0, tm) for tm in (0.0, 0.20, 0.35)]
    RL[t0] = vals
    r = vals[1]
    tc = t_cross(r)
    cc = [np.log(r / RHO_FULL) / g if r > RHO_FULL else 0.0 for g in (0.020, 0.015, 0.025)]
    print('%.2f   %s | %.1f (%d) / %.1f / %.1f / %.1f' %
          (t0, '  '.join('%.3f' % v for v in vals), tc, 2025 + int(np.floor(tc)), *cc))

# ----------------------------------------------------- IG hitting times
print('\nInverse-Gaussian hitting time of ln rho_lock (drift 0.020, sigma_lnrho):')
for t0 in (0.60, 0.75):
    r = RL[t0][1]
    d = np.log(r / RHO_FULL)
    for sig in (0.005, 0.01, 0.02):
        m = d / 0.020; lam = d ** 2 / sig ** 2
        ig = stats.invgauss(mu=m / lam, scale=lam)
        q = ig.ppf([0.05, 0.5, 0.95])
        print(' t0=%.2f rho_lock=%.3f d=%.4f sigma=%.3f: mean %.2f y, 5%%/50%%/95%% = %.2f/%.2f/%.2f y' %
              (t0, r, d, sig, ig.mean(), *q))
# parameter uncertainty in gamma_rho
print('\nParameter uncertainty: gamma_rho uniform on [0.015, 0.025]:')
for t0 in (0.60, 0.75):
    r = RL[t0][1]; d = np.log(r / RHO_FULL)
    g = np.random.default_rng(1).uniform(0.015, 0.025, 200000)
    th = d / g
    print(' t0=%.2f: t_hit 5/50/95%% = %.1f/%.1f/%.1f y' % (t0, *np.percentile(th, [5, 50, 95])))

# ----------------------------------------------------- lock boundary in tau as rho drifts
print('\nTable V.26: lock boundary in tau (share of E) and in resident turnout as rho drifts (pi=0)')
U = Prior(1.0)
for rho in (1.806, 1.934, 2.0, 2.1, 2.2, 2.3, 2.42):
    row = []
    for tm in (0.0, 0.20, 0.35):
        tl = tau_lock(rho, tm, U)
        row.append('%s (%s)' % ('%.3f' % tl if tl else 'none', '%.3f' % (rho * tl) if tl else '-'))
    print('rho=%.3f  ' % rho + '   '.join(row))

# ----------------------------------------------------- T5 clause (F28/F41)
print('\nT5 clause: (3.11) rho > (7 + tmin)/4 ; Proposition 19 tmin*(rho) at tau = 0.50')
def tmin_star(rho):
    """smallest tmin at which the T5 opposition campaigns at tau = 0.50 (4.18)"""
    if rho >= 2.0:
        return None
    f = lambda tm: (1 - rho / 2) - (1 - tm) ** 2 / (2 * rho * (rho - 2 * tm))
    g = np.linspace(0, min(0.999, rho / 2 - 1e-6), 20001)
    v = np.array([f(x) for x in g])
    idx = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    return optimize.brentq(f, g[i], g[i + 1])
for rho in (1.78, 1.7803, 1.806, 1.84, 1.90, 1.934, 1.97):
    ts = tmin_star(rho)
    print(' rho=%.4f tmin* = %s; (3.11) holds for tmin < %.3f' %
          (rho, '%.3f' % ts if ts else 'none', 4 * rho - 7))
for lev, lab in ((1.8375, '(3.11) for every tmin<=0.35'), (2.0, '(3.11) for tmin<=1; tau=0.50 infeasible')):
    print(' E_dom reaches %.4f [%s] after %.2f y (A29 path)' % (lev, lab, t_cross(lev, EDOM0)))
# reverse hitting probability (register purge as diffusion is NOT modelled)
d = np.log(RHO_DOM / 1.7803)
for sig in (0.005, 0.01, 0.02):
    print(' Pr[ln rho_dom ever falls to ln 1.7803 | drift 0.02, sigma %.3f] = %.4f' %
          (sig, np.exp(-2 * 0.02 * d / sig ** 2)))

# ----------------------------------------------------- municipalities
# [REVISED B2] exact unit ratios rho_u = E_u / R_u from Section II, Table C.4 (Ê_u, R̂_u)
C4 = {'Tirane': (647195, 480610), 'Durres': (256353, 123423), 'Elbasan': (175943, 92479),
 'Shkoder': (171668, 82302), 'Fier': (167767, 81923), 'Vlore': (165906, 67236), 'Korce': (110291, 48813),
 'Lushnje': (108901, 50726), 'Kamez': (107358, 77242), 'Lezhe': (90791, 41261), 'Berat': (84493, 50001),
 'Pogradec': (77743, 37015), 'Kruje': (68028, 41130), 'Kavaje': (67984, 24113), 'Diber': (67457, 40796),
 'Kurbin': (62370, 27643), 'Maliq': (55258, 24914), 'Kukes': (50754, 29025), 'Kucove': (47250, 24969),
 'Malesi e Madhe': (46075, 17422), 'Divjake': (45609, 19992), 'Gjirokaster': (44482, 18696),
 'Sarande': (43308, 18169), 'Vau i Dejes': (41843, 15475), 'Cerrik': (39866, 20217), 'Librazhd': (37754, 18730),
 'Shijak': (37688, 17723), 'Mallakaster': (36773, 12725), 'Patos': (36521, 14645), 'Devoll': (36222, 20807),
 'Rrogozhine': (34766, 10097), 'Bulqize': (34372, 21554), 'Dimal': (33798, 22605), 'Finiq': (33374, 9170),
 'Peqin': (33242, 13321), 'Mat': (32998, 13984), 'Mirdite': (31946, 10947), 'Gramsh': (31024, 13284),
 'Vore': (30960, 17372), 'Selenice': (30587, 7697), 'Prrenjas': (28226, 15079), 'Belsh': (27678, 13758),
 'Roskovec': (26959, 13122), 'Tropoje': (24112, 11400), 'Himare': (23114, 6691), 'Dropull': (19766, 6636),
 'Memaliaj': (18344, 5285), 'Has': (18156, 9388), 'Permet': (17348, 6412), 'Kolonje': (17022, 6041),
 'Skrapar': (16298, 8637), 'Polican': (15673, 7040), 'Delvine': (15448, 4954), 'Konispol': (15157, 3935),
 'Puke': (14387, 4999), 'Klos': (14201, 9780), 'Tepelene': (13990, 5432), 'Kelcyre': (10654, 3535),
 'Fushe-Arrez': (9956, 3919), 'Libohove': (6117, 2222), 'Pustec': (4436, 1481)}
assert len(C4) == 61
rho_u = {k: e / r for k, (e, r) in C4.items()}
r = np.array(list(rho_u.values()))
print('\nTable V.27: municipalities with q_A rho_u > 1 (local Regime A bar above all residents)')
rng = np.random.default_rng(7)
for yrs in (0, 2, 5, 10, 15, 20):
    det = int(np.sum(r * np.exp(0.02 * yrs) > 3))
    bands = []
    for s in (0.05, 0.10):
        c = rng.normal(0, s, (20000, 1)); i = rng.normal(0, s, (20000, r.size))
        cnt = np.sum(r * np.exp(0.02 * yrs + c + i) > 3, axis=1)
        bands.append('%d-%d' % tuple(np.percentile(cnt, [5, 95])))
    print(' %d: %2d (90%% band sigma 5%%: %s; 10%%: %s); bar > 1/2 (rho_u >= 1.5): %d' %
          (2025 + yrs, det, *bands, int(np.sum(r * np.exp(0.02 * yrs) >= 1.5))))
print('Remark 1 units (rho_u < 1.5) and crossing times at 2% a year (base 2025):')
for nm, v in sorted(rho_u.items(), key=lambda kv: kv[1]):
    if v < 1.5:
        print('  %-8s rho_u %.4f reaches 1.5 after %.2f y (%.1f)' % (nm, v, np.log(1.5 / v) / 0.02, 2025 + np.log(1.5 / v) / 0.02))
print('Units with q_A rho_u > 1 in 2025:', sorted(k for k, v in rho_u.items() if v > 3))
print('Next joiners (years to cross 3.0):')
for nm, v in sorted(((k, v) for k, v in rho_u.items() if v <= 3), key=lambda kv: -kv[1])[:8]:
    print('  %-12s rho_u %.4f -> %.2f y' % (nm, v, np.log(3 / v) / 0.02))
gE = GAMMA_E / E0
for nm, Eu in (('Elbasan', 175943), ('Shkoder', 171668), ('Fier', 167767), ('Vlore', 165906)):
    print('  %-8s E_u %d reaches 200,000 in %.1f y at the national growth rate %.4f' %
          (nm, Eu, np.log(200000 / Eu) / gE, gE))
