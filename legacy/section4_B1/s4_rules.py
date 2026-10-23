"""Validity rules V1-V11 as change sets in mu; equilibrium (pure, best-response on a grid); noisy pass probabilities (A11)."""
import numpy as np
from s4_core import *

INF = 9.0
def iv_major(sx, sd):            # mu > sd/(sx+sd)
    return (sd/(sx+sd), INF) if sx+sd > 0 else (INF, INF)
def iv_appr(Q, sx):              # sx mu >= Q
    return (Q/sx, INF) if sx > 0 else (INF, INF)
def iv_part(Q, sx, sd):          # sd + (sx-sd) mu >= Q
    if abs(sx-sd) < 1e-12: return (-INF, INF) if sd >= Q else (INF, INF)
    if sx > sd: return ((Q-sd)/(sx-sd), INF)
    return (-INF, (sd-Q)/(sd-sx))
def iv_super(q, sx, sd):         # Yes/(Yes+No) >= q
    den = (1-q)*sx + q*sd
    return (q*sd/den, INF) if den > 0 and sx > 0 else (INF, INF)
def inter(*ivs):
    a = max(i[0] for i in ivs); b = min(i[1] for i in ivs); return (a, b) if a < b else (INF, INF)

def make_rule(code, rho, **kw):
    """Returns change(sx, sd) -> list of (interval, weight) for the challenger's payoff and pass."""
    if code == 'V1':  return lambda sx, sd: [(iv_major(sx, sd), 1.0)]
    if code == 'V2':  Q = kw.get('q', .40)*rho; return lambda sx, sd: [(inter(iv_major(sx, sd), iv_part(Q, sx, sd)), 1.0)]
    if code == 'V3':  Q = kw.get('q', QA)*rho;  return lambda sx, sd: [(inter(iv_major(sx, sd), iv_appr(Q, sx)), 1.0)]
    if code == 'V4':  Q = kw.get('q', .5)*rho;  return lambda sx, sd: [((Q/sx if sx > 0 else INF, INF), 1.0)]  # rejection; opposition = side x
    if code == 'V5a': Q = kw.get('qT', .5)*kw.get('tauref', .76); return lambda sx, sd: [(inter(iv_major(sx, sd), iv_appr(Q, sx)), 1.0)]
    if code == 'V6d': Q = kw.get('q', QA);      return lambda sx, sd: [(inter(iv_major(sx, sd), iv_appr(Q, sx)), 1.0)]
    if code == 'V7':  q = kw.get('qS', .6);     return lambda sx, sd: [(iv_super(q, sx, sd), 1.0)]
    if code == 'V8':
        Q, q = kw.get('q', QA)*rho, kw.get('qS', .55)
        return lambda sx, sd: [(inter(iv_super(q, sx, sd), iv_appr(Q, sx)), 1.0)]
    if code == 'V9':
        Qa, Qb, lam = kw.get('qa', .20)*rho, kw.get('qb', QA)*rho, kw.get('lam', .5)
        return lambda sx, sd: [(inter(iv_major(sx, sd), iv_appr(Qa, sx)), lam), (inter(iv_major(sx, sd), iv_appr(Qb, sx)), 1-lam)]
    raise ValueError(code)

def payoff_mass(ivw, prior):
    return sum(w*float(prior.mass(max(a, prior.lo), min(b, prior.hi))) if a < b else 0.0 for (a, b), w in ivw)

def _polish(change, t0, tmin, prior, fcal, sx, sd, h, iters=50):
    """[REVISED B1] Continuous best responses started from a grid equilibrium (bounded search within one grid step)."""
    def br(other, side):
        def f(s):
            m = payoff_mass(change(s, other) if side == 'x' else change(other, s), prior)
            return -((m if side == 'x' else 1 - m) - cost(s - tmin, t0, tmin, fcal))
        mid = sx if side == 'x' else sd
        lo, hi = max(tmin, mid - h), min(1.0, mid + h)
        r = optimize.minimize_scalar(f, bounds=(lo, hi), method='bounded', options={'xatol': 1e-10})
        return r.x if -r.fun > -f(mid) + 1e-12 else mid
    for _ in range(iters):
        sx2 = br(sd, 'x'); sd2 = br(sx2, 'd')
        if abs(sx2 - sx) < 1e-9 and abs(sd2 - sd) < 1e-9: break
        sx, sd = sx2, sd2
    return sx, sd

def solve_eq(change, t0, tmin, prior, fcal=None, n=301, start=None, iters=60, polish=True):
    """Pure-strategy equilibrium by iterated best responses on a grid of side rates in [tmin, 1],
    then polished by continuous best responses [REVISED B1]."""
    if fcal is None: fcal = float(prior.pdf(.5))
    g = np.linspace(tmin, 1.0, n)
    c = cost(g - tmin, t0, tmin, fcal)
    sx, sd = (start if start else (min(t0, 1.0), min(t0, 1.0)))
    hist = []
    for _ in range(iters):
        px = np.array([payoff_mass(change(s, sd), prior) for s in g]) - c
        sx_new = g[int(np.argmax(px))]
        pd = np.array([1 - payoff_mass(change(sx_new, s), prior) for s in g]) - c
        sd_new = g[int(np.argmax(pd))]
        if abs(sx_new-sx) < 1e-12 and abs(sd_new-sd) < 1e-12:
            if polish: sx, sd = _polish(change, t0, tmin, prior, fcal, sx, sd, g[1] - g[0])
            return dict(sx=sx, sd=sd, pure=True)
        hist.append((sx_new, sd_new)); sx, sd = sx_new, sd_new
        if hist.count((sx, sd)) > 2: break
    return dict(sx=sx, sd=sd, pure=False)

# ---------------- A11 noise ----------------
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80); GH_W = GH_W/GH_W.sum()   # [REVISED B1] 80 nodes (40 erred by 0.002 in one kinked cell)
def noisy_rates(sx, sd, sig, r):
    """Gauss-Hermite nodes for (sx~, sd~): additive BVN shocks SD sig, corr r, truncated to [0,1]; a zero rate stays zero."""
    Z1, Z2 = np.meshgrid(GH_X, GH_X, indexing='ij'); W = np.outer(GH_W, GH_W)
    ux = sig*Z1; ud = sig*(r*Z1 + np.sqrt(max(1-r*r, 0))*Z2)
    X = np.clip(sx + ux, 0, 1) if sx > 0 else np.zeros_like(ux)
    D = np.clip(sd + ud, 0, 1) if sd > 0 else np.zeros_like(ud)
    return X.ravel(), D.ravel(), W.ravel()

def pass_probs(change, sx, sd, prior, sig, r=.5):
    """Ex-ante Pr[pass], Pr[pass & mu>1/2], Pr[pass & mu<1/2]. Pass = the binding (weight 1-tier) change.
    For V9 pass is the last interval (binding); advisory reported separately."""
    if sig == 0: X, D, W = np.array([sx]), np.array([sd]), np.array([1.0])
    else: X, D, W = noisy_rates(sx, sd, sig, r)
    P = Pmaj = Pmin = 0.0
    for x, d, w in zip(X, D, W):
        (a, b), _ = change(x, d)[-1]
        if a < b:
            P += w*float(prior.mass(max(a, prior.lo), min(b, prior.hi)))
            Pmaj += w*float(prior.mass(max(a, .5), min(b, prior.hi)))
            Pmin += w*float(prior.mass(max(a, prior.lo), min(b, .5)))
    return P, Pmaj, Pmin
