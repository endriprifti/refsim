"""Section III/IV challenger problem, reused in Section V Part 2.

Regime A (V3) with the default side at its floor (Remark 1, Q >= tbar/2).
Payoff of the challenger in units of B:
    Pr[win | s] - f_cal * (s - tmin)^2 / (8 t0 (t0 - tmin))
with Pr[win | s] = S_mu( max(Q/s, tmin/(s+tmin)) ).
f_cal = f_mu(1/2) (kappa recalibrated) or the uniform calibration 1/(1-pi).
The challenger campaigns iff the maximum over s in (tmin, tbar] exceeds the
payoff at s = tmin (zero when Q >= tmin * mu_hi).
"""
import numpy as np
from scipy import stats, optimize

TBAR = 1.0
RHO_FULL = 3713761 / 1.92e6   # 1.934
RHO_DOM = 3467826 / 1.92e6    # 1.806


class Prior:
    """Scaled Beta(a, b) on [lo, hi]; a = b = 1 is A2's uniform."""
    def __init__(self, a=1.0, b=None, lo=0.0, hi=1.0):
        self.a, self.b = a, (a if b is None else b)
        self.lo, self.hi = lo, hi
        self.d = stats.beta(self.a, self.b)

    def S(self, mu):
        x = (np.asarray(mu, float) - self.lo) / (self.hi - self.lo)
        return self.d.sf(np.clip(x, 0.0, 1.0))

    def f(self, mu):
        x = (mu - self.lo) / (self.hi - self.lo)
        if x <= 0 or x >= 1:
            return 0.0
        return self.d.pdf(x) / (self.hi - self.lo)


def bloc_prior(pi, a=1.0, b=None, mub=0.5):
    lo = pi * mub
    return Prior(a, b, lo, lo + 1 - pi)


def payoff(s, Q, t0, tmin, prior, fcal):
    thr = np.maximum(Q / s, tmin / (s + tmin))
    return prior.S(thr) - fcal * (s - tmin) ** 2 / (8 * t0 * (t0 - tmin))


def best_response(Q, t0, tmin, prior, fcal, tbar=TBAR, n=4001):
    s = np.linspace(tmin + 1e-9, tbar, n)
    u = payoff(s, Q, t0, tmin, prior, fcal)
    i = int(np.argmax(u))
    lo, hi = s[max(i - 1, 0)], s[min(i + 1, n - 1)]
    if hi > lo:
        r = optimize.minimize_scalar(lambda x: -payoff(x, Q, t0, tmin, prior, fcal),
                                     bounds=(lo, hi), method='bounded',
                                     options={'xatol': 1e-12})
        if -r.fun > u[i]:
            return float(r.x), float(-r.fun)
    return float(s[i]), float(u[i])


def campaigns(Q, t0, tmin, prior, fcal, tbar=TBAR):
    base = float(prior.S(Q / tmin)) if tmin > 0 else 0.0
    s, u = best_response(Q, t0, tmin, prior, fcal, tbar)
    return u > base + 1e-12, s, u


def tau_lock(rho, tmin, prior, recal=True, pi=0.0, qA=1/3, tau_ab=0.0,
             lo=0.05, hi=None):
    """Smallest registered turnout tau (share of E) at which the challenger
    campaigns.  t0 = rho*tau - tau_ab*(rho-1).  Returns None if never within
    t0 <= tbar."""
    Q = qA * rho
    fcal = prior.f(0.5) if recal else 1.0 / (1.0 - pi)
    if hi is None:
        hi = (TBAR + tau_ab * (rho - 1)) / rho - 1e-9

    def camp(tau):
        t0 = rho * tau - tau_ab * (rho - 1)
        if t0 <= tmin + 1e-9:
            return False
        return campaigns(Q, t0, tmin, prior, fcal)[0]

    if not camp(hi):
        return None
    a, b = lo, hi
    while camp(a):
        a /= 2
    for _ in range(60):
        m = 0.5 * (a + b)
        if camp(m):
            b = m
        else:
            a = m
    return b


def pi_star_uniform(Q, t0, tmin, mub=0.5, tbar=TBAR):
    """(3.9) with the floor: pi* = 2[1 - mu_hat - cost] (mu_b = 1/2)."""
    fcal = 1.0  # units of B f_pi: the payoff is l - t^2/(8 t0 (t0 - tmin))
    s = np.linspace(tmin + 1e-9, tbar, 20001)
    thr = np.maximum(Q / s, tmin / (s + tmin))
    cost = (s - tmin) ** 2 / (8 * t0 * (t0 - tmin))
    # objective independent of pi while thr inside support: maximise -thr - cost
    j = int(np.argmax(-thr - cost))
    return 2 * (1 - thr[j] - cost[j]), s[j], thr[j]


if __name__ == '__main__':
    # Reproduction checks against Sections III-IV
    for rho in (1.93, RHO_FULL):
        Q = rho / 3
        for tmin in (0.0, 0.20, 0.35):
            print('pi* tau=0.40 rho=%.3f tmin=%.2f -> %.3f' %
                  (rho, tmin, pi_star_uniform(Q, rho * 0.40, tmin)[0]))
    U = Prior(1.0)
    B25 = Prior(2.5)
    for name, rho in (('full', RHO_FULL), ('dom', RHO_DOM), ('1.93', 1.93), ('1.81', 1.81)):
        row = []
        for tmin in (0.0, 0.20, 0.35):
            row.append((tau_lock(rho, tmin, U), tau_lock(rho, tmin, B25, True),
                        tau_lock(rho, tmin, B25, False)))
        print(name, ['%.3f/%.3f/%.3f' % r for r in row])
