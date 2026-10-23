"""V.5: Ornstein-Uhlenbeck mobilisation premium around the Assembly election
(A28) and the national window of Code 2003 Art. 119(2)."""
import numpy as np, datetime as dt
from dateutil.relativedelta import relativedelta as rd
from game import pi_star_uniform, RHO_FULL

MONTH = 30.4375
D = dt.date
# (t_el, t_end, t_1); t_end = previous first sitting + 4 years
cyc = {
 2013: (D(2013, 6, 23), D(2013, 9, 7), D(2013, 9, 9)),
 2017: (D(2017, 6, 25), D(2017, 9, 9), D(2017, 9, 9)),    # first sitting 9 Sep 2017 (30th Kuvendi) [REVISED B2]
 2021: (D(2021, 4, 25), D(2021, 9, 9), D(2021, 9, 10)),
 2025: (D(2025, 5, 11), D(2025, 9, 10), D(2025, 9, 12)),
}
print('Table V.22: distance of the window edges from Assembly polling day (months)')
dpre, dpost, gend, g1 = {}, {}, {}, {}
for y, (el, te, t1) in cyc.items():
    a = te - rd(months=6); b = t1 + rd(months=3)
    dpre[y] = (el - a).days / MONTH; dpost[y] = (b - el).days / MONTH
    gend[y] = (te - el).days / MONTH; g1[y] = (t1 - el).days / MONTH
    print(y, 'window', a, b, ' d_pre %.2f  d_post %.2f  t_end-t_el %.2f  t_1-t_el %.2f' %
          (dpre[y], dpost[y], gend[y], g1[y]))
dpre_rng = (min(dpre.values()), max(dpre.values()))
dpost_rng = (min(dpost.values()), max(dpost.values()))

print('\nTable V.23: residual share of the peak premium at the window edges e^{-d/T}')
print('T (months)  pre-edge [%.2f, %.2f]   post-edge [%.2f, %.2f]' % (*dpre_rng, *dpost_rng))
for T in (0.5, 1.0, 1.5, 2.0, 2.2, 3.0, 4.0, 6.0):
    print('%5.1f   %.3f-%.3f   %.3f-%.3f' % (T, np.exp(-dpre_rng[1] / T), np.exp(-dpre_rng[0] / T),
                                            np.exp(-dpost_rng[1] / T), np.exp(-dpost_rng[0] / T)))

print('\nTable V.23 [REVISED B2]: span T ln(1/q_w), w_post* (q_w = 0.10), polling-day-anchored blocked share')
g1min, g1max = min(g1.values()), max(g1.values())
for T in (0.5, 1.0, 1.5, 2.2, 3.0, 4.0, 6.0):
    L10, L05 = T * np.log(10), T * np.log(20)
    wpost = (max(0.0, L10 - g1max), max(0.0, L10 - g1min))
    print('T=%.1f: span %.2f / %.2f; w_post* %.2f-%.2f; blocked share %.3f / %.3f' %
          (T, L10, L05, *wpost, 2 * L10 / 48, 2 * L05 / 48))
print('(5.32): q_w=0.10: T_pre <= %.2f-%.2f, T_post <= %.2f-%.2f; q_w=0.05: T_pre <= %.2f-%.2f, T_post <= %.2f-%.2f' %
      (dpre_rng[0] / np.log(10), dpre_rng[1] / np.log(10), dpost_rng[0] / np.log(10), dpost_rng[1] / np.log(10),
       dpre_rng[0] / np.log(20), dpre_rng[1] / np.log(20), dpost_rng[0] / np.log(20), dpost_rng[1] / np.log(20)))
print('polling-day anchor: start 4 months before t_el; T_pre <= %.2f months meets q_w = 0.10; blocked share %.3f-%.3f' %
      (4 / np.log(10), (4 + g1min + 3) / 48, (4 + g1max + 3) / 48))
print('Current phi_B = (6 + 3 + delta)/48 = %.4f (delta = 0)' % (9 / 48))

# ---------------------------------------------------------------- calibration
print('\nRemark 6 [REVISED B2]: two-point bound on T_post')
# 2025 partial local elections, 9 Nov 2025: Vlore, Berat, Cerrik, Mat, Tepelene
E_list = 334539; turnout = 0.1845
Rhat = {'Vlore': 67236, 'Berat': 50001, 'Cerrik': 20217, 'Mat': 13984, 'Tepelene': 5432}
Ehat = {'Vlore': 165906, 'Berat': 84493, 'Cerrik': 39866, 'Mat': 32998, 'Tepelene': 13990}
R5 = sum(Rhat.values()); E5 = sum(Ehat.values())
lag25 = (D(2025, 11, 9) - D(2025, 5, 11)).days / MONTH
for adj, lab in ((1.0, 'census 2023 basis'), (0.977, 'residents -2.3% to 2025')):
    print('rho_hat(5 units) = %.3f (proxy %.3f); resident turnout = %.3f (%s)' %
          (E_list / (R5 * adj), E5 / (R5 * adj), turnout * E_list / (R5 * adj), lab))
t25 = (turnout * E_list / R5, turnout * E_list / (R5 * 0.977))
# [REVISED B2] consistent bound: the same stationary level L enters residual and peak
def bound(t25, L, peak_level=0.76):
    r = t25 - L
    if r <= 0:
        return 0.0                      # data imply no residual: degenerate bound
    return lag25 / np.log((peak_level - L) / r)
print('lag 2025 partial = %.4f months' % lag25)
print('consistent maximum (L = 0.38, t25 = %.4f): T_post <= %.3f months' % (t25[1], bound(t25[1], 0.38)))
print('  (with t25 rounded to 0.403: %.3f)' % bound(0.403, 0.38))
for L in (0.30, 0.35, 0.38, 0.40, 0.4027):
    print('  L = %.4f: T_post <= %.2f (t25 = %.4f), %.2f (t25 = %.4f)' %
          (L, bound(t25[1], L), t25[1], bound(t25[0], L), t25[0]))
vals = [bound(t * f, 0.38) for t in t25 for f in (0.95, 1.0, 1.05)]
print('  +-5%% error in rho_hat (L = 0.38): range %.2f-%.2f months' % (min(vals), max(vals)))
for T in (1.0, 2.12, 3.0, 6.0):
    print('  T=%.2f: residual in the 2025 figure = %.3f-%.3f (peak 0.35-0.38); 2025 floor net of it = %.3f-%.3f; 2019 residual %.4f' %
          (T, 0.35 * np.exp(-lag25 / T), 0.38 * np.exp(-lag25 / T),
           t25[0] - 0.38 * np.exp(-lag25 / T), t25[1] - 0.35 * np.exp(-lag25 / T), 0.38 * np.exp(-24 / T)))

# ---------------------------------------------------------------- residual and the lock
def pi_star_fixed_kappa(Q, t0_sa, tmin_sa, tmin_now, tbar=1.0):
    """(3.9) with the cost held at the stand-alone calibration: payoff
    l - (s - tmin_now)^2 / (8 t0_sa (t0_sa - tmin_sa)); floor tmin_now."""
    c = 1.0 / (8 * t0_sa * (t0_sa - tmin_sa))
    s_ = np.linspace(tmin_now + 1e-9, tbar, 40001)
    thr = np.maximum(Q / s_, tmin_now / (s_ + tmin_now))
    j = int(np.argmax(-thr - c * (s_ - tmin_now) ** 2))
    return 2 * (1 - thr[j] - c * (s_[j] - tmin_now) ** 2)

print('\nCheck: Proposition 12 reproduced at tau = 0.40 (rho 1.93 / 1.81), tmin_sa in {0, 0.20, 0.35}, tmin_co in {0.60, 0.76}')
for rho in (1.93, 1.81):
    vals = [pi_star_fixed_kappa(rho / 3, rho * 0.40, ts, tc) for ts in (0.0, 0.20, 0.35) for tc in (0.60, 0.76)]
    print('  rho %.2f: pi*_co range %.3f-%.3f' % (rho, min(vals), max(vals)))

print('\nTable V.24 [REVISED B2]: lock bound pi* at distance d from Assembly polling day; kappa at the')
print('stand-alone calibration (tau = 0.30, t_min = 0.20, E_full); floor t_min(d) = 0.20 + 0.40 e^{-d/T}')
Q = RHO_FULL / 3
t0_sa = RHO_FULL * 0.30
for T in (1.0, 2.2, 4.0):
    row = [pi_star_fixed_kappa(Q, t0_sa, 0.20, 0.20 + 0.40 * np.exp(-d / T)) for d in (0, 1, 2, 3, 4, 5.5, 7.5, 12)]
    print('T=%.1f m: d = 0,1,2,3,4,5.5,7.5,12 -> %s' % (T, ' '.join('%5.2f' % max(x, 0.0) for x in row)))

# [REVISED B2] Proposition 28: a national referendum may follow general local
# elections at once (Art. 119(1) bars only their polling day); share of the local
# premium left at distance d (months) at the consistent decay time of Remark 6
T_CONS = bound(t25[1], 0.38)
print('\nLocal-election premium at T_post = %.3f months: share left after 1 month %.3f, after 3 months %.3f;'
      ' distance needed for q_w = 0.10: %.2f months' %
      (T_CONS, np.exp(-1 / T_CONS), np.exp(-3 / T_CONS), T_CONS * np.log(10)))
