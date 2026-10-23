"""Worked illustration (illustrative parameters): the same issue returning
after a failed vote, through opinion dynamics, the bar, the calendar and
register drift."""
import numpy as np, datetime as dt
from dateutil.relativedelta import relativedelta as rd
from game import Prior, campaigns, tau_lock, best_response
from calendar_al import first_feasible_on_or_after, poll_after_clearance, FEAS, D
def rho_det(t, E=3713761.0):
    return (E + 31200.0 * t) / (1.92e6 * np.exp(-0.0116 * t))

MONTH = 30.4375

# --- first vote: Court ruling before 15 March 2027, prompt President
clear1 = D(2027, 3, 10)
poll1 = poll_after_clearance(clear1, 77)
rho1 = rho_det((poll1 - D(2025, 5, 11)).days / 365.25)
t0, tmin, mu0 = 0.58, 0.20, 0.58
U = Prior(1.0)
Q1 = rho1 / 3
camp1 = campaigns(Q1, t0, tmin, U, 1.0)
print('First vote %s; rho %.3f; tau = t0/rho = %.3f; Regime A bar Q = %.3f; campaign: %s' %
      (poll1, rho1, t0 / rho1, Q1, camp1[0]))
print('  lock boundary tau at this rho (tmin 0.20): %.3f' % tau_lock(rho1, tmin, U))
print('  outcome: only the floor votes; turnout %.3f of E; Yes share of valid votes = mu0 = %.2f; void' %
      (tmin / rho1, mu0))
# V6d and V1 at the same vote
s6, u6 = best_response(1/3, t0, tmin, U, 1.0)
c6 = campaigns(1/3, t0, tmin, U, 1.0)
print('  V6d (1/3 of residents): campaign %s, s_x %.3f, threshold mu_hat %.3f -> %s at mu0' %
      (c6[0], c6[1], max((1/3) / c6[1], tmin / (c6[1] + tmin)),
       'passes' if mu0 >= max((1/3) / c6[1], tmin / (c6[1] + tmin)) else 'fails'))
print('  V1: mu_hat = 1/2 -> passes at mu0 = %.2f' % mu0)

# --- repeat dates
anniv = poll1 + rd(years=3)
B1 = first_feasible_on_or_after(anniv)
# BF (bar on filing): collection 21 whole weeks (median 20.5 weeks conditional on
# qualifying within 31 weeks), statutory 200 days
dep = anniv + dt.timedelta(weeks=21)
clear = dep + dt.timedelta(days=200)
B2 = poll_after_clearance(clear, 77)
# VR counterfactual (bar keyed to a valid result; needs an amendment of Art. 151(3)):
# request filed 4 weeks after the first poll
dep_h2 = poll1 + dt.timedelta(weeks=4 + 21)
clear_h2 = dep_h2 + dt.timedelta(days=200)
H2 = poll_after_clearance(clear_h2, 77)
for lab, d in (('VR: valid-result bar (counterfactual)', H2), ('H1-BP: bar on the polling day', B1),
               ('H1-BF: bar on filing', B2)):
    t = (d - D(2025, 5, 11)).days / 365.25
    r = rho_det(t)
    tl = tau_lock(r, tmin, U)
    print('%-32s repeat %s (%.1f months later): rho %.3f, tau = %.3f, lock boundary %.3f (resident %.3f), Q %.3f'
          % (lab, d, (d - poll1).days / MONTH, r, t0 / r, tl, tl * r, r / 3))

# --- opinion at the repeat: exact (finite-volume) Jacobi transition from mu0  [REVISED B2]
from jacobi_fv import Jacobi, legendre_tail

def tail_mass(J, g, c):
    idx = J.x > c + 1e-12
    on = np.isclose(J.x, c)
    return float(np.dot(J.m[idx], g[idx]) + 0.5 * np.dot(J.m[on], g[on]))

print('\nOpinion at the repeat (start mu0 = 0.58; mean reverts to 1/2), exact:')
paths = (('VR (valid-result bar, counterfactual)', H2), ('H1-BP', B1), ('H1-BF', B2))
for a in (1.0, 2.5):
    J = Jacobi(a, a, 8000)
    for Trel in (1.0, 3.0):
        yrs = [(d - poll1).days / 365.25 for _, d in paths]
        S, gs = J.evolve(J.point_mass(mu0), max(yrs) / Trel, 1e-3, barrier=None,
                         record=sorted(y / Trel for y in yrs))
        for (lab, d), y, g in zip(paths, yrs, gs):
            r = rho_det((d - D(2025, 5, 11)).days / 365.25)
            chk = ('; Legendre series Pr[mu>1/2] %.4f' % legendre_tail(mu0, 0.5, y / Trel)) if a == 1.0 else ''
            print(' a=%.1f T_rel=%.0f %-38s (%.2f y): E[mu] %.4f (theory %.4f); Pr[mu>1/2] %.4f; '
                  'Pr[mu > rho/3=%.3f] %.4f%s' %
                  (a, Trel, lab, y, float(np.dot(J.m * g, J.x)), 0.5 + 0.08 * np.exp(-y / Trel),
                   tail_mass(J, g, 0.5), r / 3, tail_mass(J, g, r / 3), chk))
