"""Exact (finite-volume) solver for the Jacobi / Wright-Fisher diffusion of
Proposition 26, used for Tables V.21 and V.29 [REVISED B2].

In units of T_rel (time t~), with abar = a_Bx + a_Bd,
    L f = (1/abar) * w(x)^-1 * d/dx [ D(x) df/dx ],
    w(x) = x^(a_Bx-1) (1-x)^(a_Bd-1),  D(x) = x^a_Bx (1-x)^a_Bd.
The process is reversible with speed density w, so the forward equation for
g = density / w has the same operator: M dg/dt = -(1/abar) K g, with M the
(exact) speed-measure mass of each control volume and K the symmetric
stiffness matrix of the fluxes D(x_{i+1/2}) (g_{i+1}-g_i)/h.  Absorption at a
(possibly moving) barrier b is imposed by g = 0 at nodes x_i >= b.
Time stepping: Crank-Nicolson after four half-steps of backward Euler
(Rannacher start).
"""
import numpy as np
from scipy import special
from scipy.linalg import solve_banded


class Jacobi:
    def __init__(self, ax, ad, N=8000):
        self.ax, self.ad, self.abar = ax, ad, ax + ad
        self.N = N
        self.x = np.linspace(0.0, 1.0, N + 1)
        h = 1.0 / N
        self.h = h
        xm = np.concatenate(([0.0], 0.5 * (self.x[:-1] + self.x[1:]), [1.0]))
        cdf = special.betainc(ax, ad, xm) * special.beta(ax, ad)
        self.m = np.diff(cdf)                           # exact speed mass per CV
        xf = 0.5 * (self.x[:-1] + self.x[1:])           # faces
        self.Df = xf ** ax * (1 - xf) ** ad / h         # flux coefficient

    def _bands(self, n_active, theta, dt):
        """Banded matrix for M + theta*dt/abar*K on the first n_active nodes
        (the rest are absorbed: g = 0)."""
        n = n_active
        c = theta * dt / self.abar
        Df = np.concatenate((self.Df, [0.0]))[:n]      # face i+1/2 (none beyond x=1)
        diag = self.m[:n].copy()
        # flux between i and i+1 for i = 0..n-2 (both active); face to an
        # absorbed node n-1 -> n couples to g_n = 0 (Dirichlet)
        diag[:n] += c * Df[:n]                          # right faces
        diag[1:n] += c * Df[:n - 1]                     # left faces
        off = -c * Df[:n - 1]
        ab = np.zeros((3, n))
        ab[0, 1:] = off
        ab[1, :] = diag
        ab[2, :-1] = off
        return ab

    def _apply_K(self, g, n):
        """(K g) on the first n nodes, g_n = 0 beyond."""
        Df = np.concatenate((self.Df, [0.0]))[:n]
        gpad = np.concatenate((g[:n], [0.0]))
        flux = Df * (gpad[1:n + 1] - gpad[:n])          # face i+1/2, i=0..n-1
        Kg = -flux.copy()
        Kg[1:] += flux[:-1]
        return Kg

    def n_active(self, b):
        """number of nodes strictly below the barrier b (b >= 1: all)."""
        if b >= 1.0:
            return self.N + 1
        return int(np.searchsorted(self.x, b, side='left'))

    def evolve(self, g0, t_end, dt, barrier=None, record=None):
        """Forward evolution of g = density/w.  barrier(t~) -> b, or None for
        no absorption.  record: sorted times t~ at which the surviving mass
        and g are returned."""
        g = g0.copy()
        t = 0.0
        out_S, out_g = [], []
        rec = list(record) if record is not None else []
        steps = [(dt / 2, 1.0)] * 4                     # Rannacher start
        while t < t_end - 1e-12:
            if steps:
                step, theta = steps.pop(0)
            else:
                step, theta = min(dt, t_end - t), 0.5
            b = barrier(t + step) if barrier is not None else 1.0
            n = self.n_active(b)
            n = min(n, self.N + 1)
            g[n:] = 0.0
            rhs = self.m[:n] * g[:n]
            if theta < 1.0:
                rhs = rhs - (1 - theta) * step / self.abar * self._apply_K(g, n)
            g[:n] = solve_banded((1, 1), self._bands(n, theta, step), rhs)
            t += step
            while rec and rec[0] <= t + 1e-12:
                out_S.append(float(np.dot(self.m, g)))
                out_g.append(g.copy())
                rec.pop(0)
        return out_S, out_g

    def point_mass(self, x0):
        j = int(round(x0 * self.N))
        g = np.zeros(self.N + 1)
        g[j] = 1.0 / self.m[j]
        return g


def survival_curve(ax, ad, x0, t_grid, barrier, N=8000, dt=1e-3):
    J = Jacobi(ax, ad, N)
    S, _ = J.evolve(J.point_mass(x0), t_grid[-1], dt, barrier=barrier, record=t_grid)
    return np.array(S)


def legendre_tail(x0, c, t, nmax=400):
    """Exact Pr[p_t > c | p_0 = x0] for a_Bx = a_Bd = 1 (uniform stationary law):
    eigenfunctions P_n(2x-1), eigenvalues n(n+1)/2 in T_rel units."""
    from scipy.special import eval_legendre
    u0, uc = 2 * x0 - 1, 2 * c - 1
    tot = (1 - c)
    for n in range(1, nmax):
        integ = -(eval_legendre(n + 1, uc) - eval_legendre(n - 1, uc)) / (2 * n + 1) / 2
        tot += np.exp(-n * (n + 1) / 2 * t) * (2 * n + 1) * eval_legendre(n, u0) * integ
    return tot
