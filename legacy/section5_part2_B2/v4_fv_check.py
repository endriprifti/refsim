"""Validation of jacobi_fv.py [REVISED B2]: (i) transition probabilities against the
Legendre series (a_B = 1); (ii) E[T_b] from the survival curve against (5.28)."""
import numpy as np
from scipy import special, integrate
from jacobi_fv import Jacobi, legendre_tail

def ET(x, b, ax, ad):
    f = lambda y: y ** (-ax) * (1 - y) ** (-ad) * special.betainc(ax, ad, y) * special.beta(ax, ad)
    v, _ = integrate.quad(f, x, b, limit=200)
    return (ax + ad) * v

J = Jacobi(1.0, 1.0, 8000)
ts = [0.5, 1.0, 1.5]
_, gs = J.evolve(J.point_mass(0.58), ts[-1], 1e-3, barrier=None, record=ts)
for t, g in zip(ts, gs):
    idx = J.x > 0.5 + 1e-12; on = np.isclose(J.x, 0.5)
    fv = float(np.dot(J.m[idx], g[idx]) + 0.5 * np.dot(J.m[on], g[on]))
    print('t~=%.1f Pr[p>1/2]: FV %.6f  Legendre series %.6f' % (t, fv, legendre_tail(0.58, 0.5, t)))
for a in (1.0, 2.5):
    J = Jacobi(a, a, 8000)
    b = 1.934 / 3
    tg = np.arange(0.01, 60.0001, 0.01)
    S, _ = J.evolve(J.point_mass(0.5), tg[-1], 2e-3, barrier=lambda t: b, record=list(tg))
    E_fv = np.trapezoid(np.concatenate(([1.0], S)), np.concatenate(([0.0], tg)))
    print('a=%.1f E[T_b]: FV %.4f  (5.28) %.4f' % (a, E_fv, ET(0.5, b, a, a)))
