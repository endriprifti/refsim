"""Table V.21 with exact (finite-volume) values [REVISED B2]: probability that
support reaches Q_A/tbar = rho/3 within three years from p0 = 1/2; fixed and
drifting barriers (median, Pr within 10 years, Pr before the barrier reaches 1).
Replaces the Monte Carlo of v4_passage.py."""
import numpy as np
from jacobi_fv import Jacobi

N, DT = 8000, 1e-3
GAMMA = 0.02
T_ONE = np.log(3 / 1.934) / GAMMA          # barrier rho0 e^{0.02 t}/3 reaches 1

def median_from(S, tg):
    i = np.searchsorted(-S, -0.5)           # S decreasing
    if i >= len(S):
        return np.inf
    return float(np.interp(0.5, [S[i], S[i - 1]], [tg[i], tg[i - 1]]))

print('Table V.21 (exact): Pr[T_b <= 3 years | p0 = 1/2]')
for a in (1.0, 2.5):
    J = Jacobi(a, a, N)
    for Trel in (1.0, 3.0, 5.0):
        row = []
        for rho in (1.806, 1.934, 2.2):
            b = rho / 3
            S, _ = J.evolve(J.point_mass(0.5), 3.0 / Trel, DT, barrier=lambda t, b=b: b,
                            record=[3.0 / Trel])
            row.append(1 - S[0])
        # fixed barrier at rho = 1.934 and drifting barrier
        tg = np.arange(0.01, 25.0001, 0.01)                       # years
        Sfix, _ = J.evolve(J.point_mass(0.5), 25.0 / Trel, DT,
                           barrier=lambda t: 1.934 / 3, record=list(tg / Trel))
        Smov, _ = J.evolve(J.point_mass(0.5), 25.0 / Trel, DT,
                           barrier=lambda t, Trel=Trel: min(1.0, 1.934 * np.exp(GAMMA * t * Trel) / 3),
                           record=list(tg / Trel))
        Sfix, Smov = np.array(Sfix), np.array(Smov)
        i10 = int(round(10 / 0.01)) - 1
        iT1 = int(np.searchsorted(tg, T_ONE)) - 1
        print('a=%.1f T_rel=%.0f: within 3y %.3f / %.3f / %.3f | fixed: median %.2f y, Pr<=10y %.3f '
              '| drifting: median %.2f y, Pr<=10y %.3f, Pr before b=1 (%.1f y) %.3f' %
              (a, Trel, *row, median_from(Sfix, tg), 1 - Sfix[i10], median_from(Smov, tg),
               1 - Smov[i10], T_ONE, 1 - Smov[iT1]))
