"""V.4(c): first passage of support to Regime A's capped threshold Q_A/tbar
under the Wright-Fisher (Jacobi) diffusion with committed units (A25, A25').
Table V.20 by quadrature of (5.28).

Time unit: T_rel = 1/(r_v (o_x + o_d)).  In these units the generator is
   L f = (1/(a_x+a_d)) [ p(1-p) f'' + (a_x - (a_x+a_d) p) f' ].
Expected hitting time of b from x < b (0 is an entrance/regular boundary,
reflecting):
   E[T_b | x] = (a_x+a_d) * int_x^b  y^{-a_x} (1-y)^{-a_d} B(y; a_x, a_d) dy
with B(y; a, b) the (unregularised) incomplete beta function.
"""
import numpy as np
from scipy import integrate, special

def ET(x, b, ax, ad):
    f = lambda y: y ** (-ax) * (1 - y) ** (-ad) * special.betainc(ax, ad, y) * special.beta(ax, ad)
    v, _ = integrate.quad(f, x, b, limit=200)
    return (ax + ad) * v

def ET_stationary_below(b, ax, ad, n=400):
    """E[T_b] starting from the stationary law conditioned on p < b."""
    from scipy import stats
    d = stats.beta(ax, ad)
    xs = d.ppf(np.linspace(0.0005, d.cdf(b) - 0.0005, n))
    return np.mean([ET(x, b, ax, ad) for x in xs])

RHOS = [1.806, 1.934, 2.0, 2.1, 2.2, 2.42, 2.62]
print('Table V.20: E[T_b | p0] in units of T_rel, b = Q_A/tbar = rho/3 (pi = 0, tau_ab = 0)')
print('a_B   start      ' + '  '.join('rho=%.3f' % r for r in RHOS))
for a in (0.5, 1.0, 2.5, 5.0):
    for lab, fn in (('p0=0.50', lambda b: ET(0.5, b, a, a)),
                    ('stat<b', lambda b: ET_stationary_below(b, a, a))):
        row = []
        for r in RHOS:
            b = r / 3
            row.append('%8.2f' % fn(b))
        print('%4.1f  %-8s ' % (a, lab) + ' '.join(row))

# derivative wrt rho at the 2025 value (dE/drho = (1/3) dE/db)
for a in (1.0, 2.5):
    b = 1.934 / 3
    db = 1e-4
    dE = (ET(0.5, b + db, a, a) - ET(0.5, b - db, a, a)) / (2 * db) / 3
    print('a=%.1f: dE[T]/drho at rho=1.934 from p0=0.5: %.2f T_rel per unit rho' % (a, dE))

# [REVISED B2] The Monte Carlo of the first version is replaced by the exact
# finite-volume computation in v4_exact.py (Table V.21) and jacobi_fv.py.
