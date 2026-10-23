"""V.4(b): lock boundary in tau as a function of the stationary Beta shape.
Table V.18 (symmetric Beta on [0,1], pi = 0) and Table V.19 (zealot family:
scaled Beta on A2's support, pi = 0.3)."""
import numpy as np
from game import Prior, bloc_prior, tau_lock, RHO_FULL, RHO_DOM, campaigns
from scipy import optimize

A_GRID = [0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 8.0, 16.0]
TMINS = [0.0, 0.20, 0.35]

def fmt(x):
    return 'none' if x is None else '%.3f' % x

print('Table V.18: tau_lock, symmetric Beta(a,a) on [0,1], pi=0 (recalibrated kappa / uniform kappa)')
res = {}
for a in A_GRID:
    pr = Prior(a)
    row = []
    for tmin in TMINS:
        cells = []
        for rho in (RHO_FULL, RHO_DOM):
            r1 = tau_lock(rho, tmin, pr, True)
            r2 = tau_lock(rho, tmin, pr, False)
            res[(a, tmin, rho, 'r')] = r1
            res[(a, tmin, rho, 'f')] = r2
            cells.append(fmt(r1) + '|' + fmt(r2))
        row.append(' / '.join(cells))
    print('a=%5.2f f(1/2)=%.3f S(0.643)=%.3f :: ' % (a, pr.f(0.5), pr.S(RHO_FULL/3)) + ' || '.join(row))

# critical a: largest a with a lock boundary at or below tau = 0.50 (D3 top)
print('\nCritical a_B above which Regime A is locked at every tau in D3 (tau<=0.50)')
for rho, nm in ((RHO_FULL, 'E_full'), (RHO_DOM, 'E_dom')):
    for tmin in TMINS:
        for recal in (True, False):
            def g(a):
                t = tau_lock(rho, tmin, Prior(a), recal)
                return (0.6 if t is None else t) - 0.50
            # bracket
            lo, hi = 1.0, 60.0
            if g(hi) < 0:
                print(nm, tmin, recal, '> 60'); continue
            a_c = optimize.brentq(g, lo, hi, xtol=1e-3)
            print('%s tmin=%.2f recal=%s a_crit=%.2f' % (nm, tmin, recal, a_c))

print('\nTable V.19: zealot family, scaled Beta(a,a) on [pi/2, 1-pi/2], pi=0.3')
for a in (0.5, 1.0, 2.5, 4.0):
    row = []
    for tmin in TMINS:
        cells = []
        for rho in (RHO_FULL, RHO_DOM):
            pr = bloc_prior(0.3, a)
            r1 = tau_lock(rho, tmin, pr, True, pi=0.3)
            r2 = tau_lock(rho, tmin, pr, False, pi=0.3)
            cells.append(fmt(r1) + '|' + fmt(r2))
        row.append(' / '.join(cells))
    print('a=%4.2f :: ' % a + ' || '.join(row))

# Pr[change] at the cap for each a (Proposition 17(ii)): S(Q/tbar)
print('\nS_mu(Q/tbar) at the cap (pi=0): E_full Q=%.4f, E_dom Q=%.4f' % (RHO_FULL/3, RHO_DOM/3))
for a in A_GRID:
    pr = Prior(a)
    print('a=%5.2f  %.3f  %.3f' % (a, pr.S(RHO_FULL/3), pr.S(RHO_DOM/3)))
