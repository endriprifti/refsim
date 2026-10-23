"""Strategy and equilibria (Specification module 3) and the criteria at equilibrium.

Section III's group-mobilisation game with a sincere floor (A1-A2, D9, A12):
  side rates s_j = t_min + t_j in [t_min, t_bar] (resident units), chosen before mu is realised;
  challenger payoff  Omega * Pr[change | s_x, s_d] - cost(s_x),
  default payoff     1 - Pr[change | s_x, s_d] - cost(s_d),
  cost(s) = f_cal (s - t_min)^2 / (8 t0 (t0 - t_min))          (3.4)-(3.5), units of B_d,
  with f_cal = f_mu(1/2) (kappa recalibrated, A12) or 1/(1 - pi) (held at the uniform calibration).
Pr[change] is the prior mass of the rule's change set (rules.intervals).  Advisory outcomes (V9, V11)
enter with weight lambda_adv.

Criteria (Section III, eq. (3.6), operational reading of (I.7)-(I.8)):
  C1   = Pr[mu_V < mu and outcome = default]            (voters' majority overridden; joint)
  C1'  = Pr[binding change and mu < 1/2]                 (false validation)
  C1^R = Pr[mu > 1/2 and outcome = default]              (diagnostic)
  C2   = [Pr(default | s_d = t_min) - Pr(default | s_d = t0)]^+ with s_x at equilibrium
  P_dec = Pr[valid result for one side]                   (C4's last stage)
A36 (v1.1, calibrated scenario): the floor depends on the side's strategy.  With t_min_b set, a default side
under a participation-type rule (rule.default_monotone False) may call to abstain: its rate can fall to t_min_b
< t_min at no cost (cost(s) = f_cal max(s - t_min, 0)^2 / ...); every other side keeps the contested floor t_min.
Turnout noise (A11): realised s~_j = min{t_bar, max{0, s_j + sigma_s Z_j}}, corr(Z_x, Z_d) = Theta,
sigma_s = rho * sigma_tau; strategies are those of the noiseless game (Section IV convention).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from .priors import Prior, kappa_calibration, uniform
from .rules import INF, Rule, make_rule

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(80)
_GH_W = _GH_W / _GH_W.sum()


def noise_nodes(sx, sd, sigma_s, Theta=0.5, tbar=1.0, n_nodes=80):
    """Gauss-Hermite nodes for (s~_x, s~_d); a zero rate stays zero (Section IV, B.23)."""
    if sigma_s <= 0:
        return np.array([sx]), np.array([sd]), np.array([1.0])
    if n_nodes == 80:
        x, w = _GH_X, _GH_W
    else:
        x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
        w = w / w.sum()
    Z1, Z2 = np.meshgrid(x, x, indexing="ij")
    W = np.outer(w, w)
    ux = sigma_s * Z1
    ud = sigma_s * (Theta * Z1 + np.sqrt(max(1 - Theta * Theta, 0.0)) * Z2)
    X = np.clip(sx + ux, 0.0, tbar) if sx > 0 else np.zeros_like(ux)
    D = np.clip(sd + ud, 0.0, tbar) if sd > 0 else np.zeros_like(ud)
    return X.ravel(), D.ravel(), W.ravel()


@dataclass
class Equilibrium:
    sx: float
    sd: float
    pure: bool = True
    approx: bool = False
    mix_x: np.ndarray | None = None
    mix_d: np.ndarray | None = None
    grid: np.ndarray | None = None
    grid_d: np.ndarray | None = None      # default side's grid when its floor differs (A36)
    others: list = field(default_factory=list)
    exploitability: float = 0.0


class Game:
    def __init__(self, rule, el, tau: float | None = None, t0: float | None = None, t_min: float = 0.0,
                 prior: Prior | None = None, pi: float = 0.0, recalibrate: bool = True,
                 fcal: float | None = None, t_bar: float = 1.0, Omega: float = 1.0, t_min_b: float | None = None):
        self.rule: Rule = make_rule(rule)
        self.el = el
        self.t0 = float(el.t0(tau) if t0 is None else t0)
        self.tau = float(el.tau(self.t0))
        self.t_min = float(t_min)
        # A36: boycott floor for the default side under participation-type rules only
        a36 = t_min_b is not None and not getattr(self.rule, "default_monotone", True)
        self.floor_x = self.t_min
        self.floor_d = float(min(t_min_b, self.t_min)) if a36 else self.t_min
        self.pi = pi
        self.prior = prior if prior is not None else uniform(pi)
        self.fcal = kappa_calibration(self.prior, recalibrate, pi) if fcal is None else fcal
        self.t_bar = t_bar
        self.Omega = Omega
        self.feasible = (self.t0 > self.t_min + 1e-9) and (self.t0 <= t_bar + 1e-12)

    # ------------------------------------------------------------ payoffs
    def cost(self, s):
        return self.fcal * np.maximum(np.asarray(s, float) - self.t_min, 0.0) ** 2 / (8 * self.t0 * (self.t0 - self.t_min))

    def change_mass(self, sx, sd):
        tot = 0.0
        for p in self.rule.intervals(sx, sd, self.el):
            tot = tot + p.weight * self.prior.mass(p.a, p.b)
        return tot

    def u_x(self, sx, sd):
        return self.Omega * self.change_mass(sx, sd) - self.cost(sx)

    def u_d(self, sx, sd):
        return 1.0 - self.change_mass(sx, sd) - self.cost(sd)

    def grid(self, n, side="x"):
        lo = self.floor_d if side == "d" else self.floor_x
        return np.linspace(lo, self.t_bar, n)

    def _br(self, other, side, n=801):
        g = self.grid(n, side)
        u = self.u_x(g, other) if side == "x" else self.u_d(other, g)
        i = int(np.argmax(u))
        best_s, best_u = g[i], float(u[i])
        lo, hi = g[max(i - 1, 0)], g[min(i + 1, n - 1)]
        if hi > lo:
            f = (lambda s: -float(self.u_x(s, other))) if side == "x" else (lambda s: -float(self.u_d(other, s)))
            r = optimize.minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-11})
            if -r.fun > best_u + 1e-13:
                best_s, best_u = float(r.x), float(-r.fun)
        return best_s, best_u

    def br_x(self, sd, n=801):
        return self._br(sd, "x", n)

    def br_d(self, sx, n=801):
        return self._br(sx, "d", n)

    def is_equilibrium(self, sx, sd, n=4001, tol=1e-7):
        gxg, gdg = self.grid(n, "x"), self.grid(n, "d")
        ux = float(self.u_x(sx, sd))
        ud = float(self.u_d(sx, sd))
        gx = float(np.max(self.u_x(gxg, sd))) - ux
        gd = float(np.max(self.u_d(sx, gdg))) - ud
        return (gx <= tol and gd <= tol), max(gx, gd, 0.0)

    # ------------------------------------------------------------ equilibrium
    def solve(self, n=801, iters=120) -> Equilibrium:
        if not self.feasible:
            raise ValueError(f"infeasible cell: t0={self.t0:.4f} <= t_min={self.t_min} or > t_bar")
        t0c = min(self.t0, self.t_bar)
        starts = [(t0c, t0c), (self.floor_x, self.floor_d), (self.t_bar, self.floor_d), (self.t_bar, self.t_bar)]
        if self.floor_d < self.t_min:
            starts.append((self.t_bar, self.t_min))
        found = []
        for sx, sd in starts:
            hist = []
            for _ in range(iters):
                sx2, _ = self.br_x(sd, n)
                sd2, _ = self.br_d(sx2, n)
                if abs(sx2 - sx) < 1e-9 and abs(sd2 - sd) < 1e-9:
                    break
                key = (round(sx2, 7), round(sd2, 7))
                if hist.count(key) > 2:
                    break
                hist.append(key)
                sx, sd = sx2, sd2
            ok, gap = self.is_equilibrium(sx, sd)
            if ok and not any(abs(sx - a) < 1e-5 and abs(sd - b) < 1e-5 for a, b in found):
                found.append((sx, sd))
        if found:
            sx, sd = found[0]
            return Equilibrium(sx=sx, sd=sd, pure=True, others=found[1:])
        return self.solve_mixed()

    def solve_mixed(self, n=121, iters=6000) -> Equilibrium:
        """No pure equilibrium (e.g. Z2): approximate mixed equilibrium by fictitious play on a grid."""
        g = self.grid(n, "x")
        gd = self.grid(n, "d")
        SX, SD = np.meshgrid(g, gd, indexing="ij")
        M = self.change_mass(SX, SD)
        Ux = self.Omega * M - self.cost(SX)
        Ud = 1.0 - M - self.cost(SD)
        cx = np.zeros(n)
        cd = np.zeros(n)
        i, j = n // 2, n // 2
        for _ in range(iters):
            cx[i] += 1
            cd[j] += 1
            i = int(np.argmax(Ux @ (cd / cd.sum())))
            j = int(np.argmax((cx / cx.sum()) @ Ud))
        px, pd = cx / cx.sum(), cd / cd.sum()
        vx = px @ Ux @ pd
        vd = px @ Ud @ pd
        expl = max(float(np.max(Ux @ pd) - vx), float(np.max(px @ Ud) - vd))
        return Equilibrium(sx=float(px @ g), sd=float(pd @ gd), pure=False, approx=True,
                           mix_x=px, mix_d=pd, grid=g, grid_d=gd, exploitability=expl)

    # ------------------------------------------------------------ outcomes and criteria
    def outcome(self, sx, sd):
        """Criteria at given (possibly array-valued) realised rates; returns dict of arrays."""
        el, pr = self.el, self.prior
        sx = np.asarray(sx, float)
        sd = np.asarray(sd, float)
        den = sx + sd
        mu_V = np.where(den > 0, (sd + el.b_ab()) / np.where(den > 0, den, 1.0), INF)
        P_maj = pr.mass(mu_V, INF)
        P_half = pr.mass(0.5, INF)
        P_change = 0.0
        P_adv = 0.0
        maj_ok = 0.0
        half_ok = 0.0
        C1p = 0.0
        a_min = np.full(np.broadcast(sx, sd).shape, INF)
        for p in self.rule.intervals(sx, sd, el):
            m = pr.mass(p.a, p.b)
            if p.kind == "change":
                P_change = P_change + m
                C1p = C1p + pr.mass(p.a, np.minimum(p.b, 0.5))
                a_min = np.minimum(a_min, p.a)
            else:
                P_adv = P_adv + m
            maj_ok = maj_ok + pr.mass(np.maximum(p.a, mu_V), p.b)
            half_ok = half_ok + pr.mass(np.maximum(p.a, 0.5), p.b)
        dv_a, dv_b = self.rule.default_valid(sx, sd, el)
        P_def_valid = pr.mass(dv_a, dv_b)
        Emu = pr.mean()
        turnout_E = (Emu * sx + (1 - Emu) * sd + el.n_ab) / el.rho
        return dict(P_change=P_change, P_advisory=P_adv, P_default=1 - P_change - P_adv,
                    C1=P_maj - maj_ok, C1p=C1p, C1R=P_half - half_ok,
                    P_dec=P_change + P_adv + P_def_valid, P_void=1 - (P_change + P_adv + P_def_valid),
                    mu_V=mu_V, mu_hat=a_min, turnout_E=turnout_E)

    def _avg(self, d, w):
        return {k: float(np.sum(np.asarray(v) * w)) if np.ndim(v) else float(v) for k, v in d.items()}

    def criteria(self, eq: Equilibrium | None = None, sigma_tau: float = 0.0, Theta: float = 0.5) -> dict:
        eq = eq or self.solve()
        sig = self.el.rho * sigma_tau
        if eq.pure:
            pairs = [(eq.sx, eq.sd, 1.0)]
        else:
            idx = np.argwhere(np.outer(eq.mix_x, eq.mix_d) > 1e-4)
            wsum = sum(eq.mix_x[i] * eq.mix_d[j] for i, j in idx)
            gd = eq.grid_d if eq.grid_d is not None else eq.grid
            pairs = [(eq.grid[i], gd[j], eq.mix_x[i] * eq.mix_d[j] / wsum) for i, j in idx]
        acc = {}
        for sx, sd, w in pairs:
            X, D, W = noise_nodes(sx, sd, sig, Theta, self.t_bar)
            o = self._avg(self.outcome(X, D), W)
            # C2 (3.6): fixed-opponent gain of stopping the default campaign (noiseless rates)
            pd_floor = self._avg(self.outcome(*noise_nodes(sx, self.floor_d, sig, Theta, self.t_bar)[:2]),
                                 noise_nodes(sx, self.floor_d, sig, Theta, self.t_bar)[2])["P_default"]
            pd_base = self._avg(self.outcome(*noise_nodes(sx, min(self.t0, self.t_bar), sig, Theta, self.t_bar)[:2]),
                                noise_nodes(sx, min(self.t0, self.t_bar), sig, Theta, self.t_bar)[2])["P_default"]
            o["C2"] = max(0.0, pd_floor - pd_base)
            for k, v in o.items():
                acc[k] = acc.get(k, 0.0) + w * v
        o0 = self.outcome(eq.sx, eq.sd)
        acc.update(sx=eq.sx, sd=eq.sd, pure=eq.pure, approx=eq.approx,
                   mu_hat_eq=float(o0["mu_hat"]), mu_V_eq=float(o0["mu_V"]),
                   regime=self.regime(eq), t0=self.t0, tau=self.tau, rho=self.el.rho)
        return acc

    def regime(self, eq: Equilibrium) -> str:
        if not eq.pure:
            return "no pure equilibrium (mixed, approx.)"
        tol = 1e-6
        x_floor = abs(eq.sx - self.floor_x) < tol
        d_floor = abs(eq.sd - self.floor_d) < tol
        if x_floor and d_floor:
            return "lock (default boycott)" if self.floor_d < self.t_min - tol else "lock (mutual abstention)"
        if d_floor:
            lab = "default boycotts" if self.floor_d < self.t_min - tol else "default demobilises"
            return lab + (" (capped)" if abs(eq.sx - self.t_bar) < tol else "")
        if x_floor:
            return "challenger demobilises"
        if abs(eq.sx - eq.sd) < 1e-5 and abs(eq.sx - self.t0) < 1e-4:
            return "slack (V1 outcome)"
        return "contested"

    # ------------------------------------------------------------ campaign tests with the default at its floor
    def campaign_gain(self, sd=None, n=4001) -> float:
        """Best payoff from campaigning (s > t_min) minus the payoff at the floor, with the default at
        `sd` (its floor by default).  > 0 iff the challenger campaigns; negative values measure how far
        the cell is inside the lock (used for pi* and tau_lock by root-finding)."""
        sd = self.floor_d if sd is None else sd
        g = self.grid(n, "x")[1:]
        u = self.u_x(g, sd)
        i = int(np.argmax(u))
        best = float(u[i])
        lo, hi = g[max(i - 1, 0)], g[min(i + 1, len(g) - 1)]
        if hi > lo:
            r = optimize.minimize_scalar(lambda s: -float(self.u_x(s, sd)), bounds=(lo, hi), method="bounded",
                                         options={"xatol": 1e-11})
            best = max(best, -float(r.fun))
        return best - float(self.u_x(self.t_min, sd))


# ---------------------------------------------------------------- lock bound and lock boundary (Regime A)
def pi_star(rule, el, tau, t_min, t_bar=1.0, mu_b=0.5) -> float:
    """Lock bound pi* (3.9): bloc alignment at which the challenger stops campaigning
    (uniform prior A2, default at its floor; nan if locked at every pi, 1 if never locked)."""
    def g(pi):
        gm = Game(rule, el, tau=tau, t_min=t_min, prior=uniform(pi, mu_b), pi=pi, t_bar=t_bar)
        return gm.campaign_gain()
    if g(0.0) <= 0:
        return float("nan")
    if g(0.999) > 0:
        return 1.0
    return float(optimize.brentq(g, 0.0, 0.999, xtol=1e-7))


def pi_star_capped(rho, tau, t_min, q=1 / 3, t_bar=1.0):
    """Closed form of (3.9) on the capped branch: 2[1 - Q/t_bar - (t_bar - t_min)^2/(8 t0 (t0 - t_min))]."""
    t0 = rho * tau
    return 2 * (1 - q * rho / t_bar - (t_bar - t_min) ** 2 / (8 * t0 * (t0 - t_min)))


def tau_lock(rule, el, t_min, prior=None, recalibrate=True, pi=0.0, t_bar=1.0, lo=0.05, t_min_b=None):
    """Smallest registered turnout tau at which the challenger campaigns (default at its floor)."""
    prior = prior or uniform(pi)
    fcal = kappa_calibration(prior, recalibrate, pi)
    hi = (t_bar + el.n_ab) / el.rho - 1e-9

    def camp(tau):
        t0 = el.t0(tau)
        if t0 <= t_min + 1e-9:
            return False
        return Game(rule, el, tau=tau, t_min=t_min, prior=prior, fcal=fcal, t_bar=t_bar,
                    t_min_b=t_min_b).campaign_gain() > 1e-12

    if not camp(hi):
        return None
    a, b = max(lo, (t_min + el.n_ab) / el.rho + 1e-6), hi
    if camp(a):
        return a
    for _ in range(50):
        m = 0.5 * (a + b)
        if camp(m):
            b = m
        else:
            a = m
    return b


def rho_lock(t0, t_min, rule="V3", pi=0.0, lo=1.0, hi=3.0, tau_ab=0.0, prior=None, recalibrate=True, t_bar=1.0):
    """Register ratio at which Regime A locks at fixed resident turnout t0 (Corollary 4, numeric)."""
    from .electorate import Electorate
    pr = prior if prior is not None else uniform(pi)

    def camp(rho):
        el = Electorate(E=rho, R=1.0, tau_ab=tau_ab)
        return Game(rule, el, t0=t0, t_min=t_min, prior=pr, pi=pi, recalibrate=recalibrate,
                    t_bar=t_bar).campaign_gain() > 1e-12

    if not camp(lo):
        return lo
    if camp(hi):
        return float("inf")
    return _bisect(camp, lo, hi)


def _bisect(camp, lo, hi, n=60):
    for _ in range(n):
        m = 0.5 * (lo + hi)
        if camp(m):
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)
