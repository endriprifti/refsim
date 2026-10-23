"""IV.6 and Remark 2: lock crossings rho_lock (Corollary 4(b), Table IV.13) and noise-anticipating campaigns."""
import numpy as np
from scipy import stats
from s4_core import *

def rho_lock(t0, tmin, pi, tbar=1.0):          # closed form, capped branch, uniform prior
    c = (tbar - tmin)**2/(8*t0*(t0 - tmin)); return 3*tbar*(1 - pi/2 - c)

def campaign_gain_noisy(Q, t0, tmin, prior, sig, r=.5, n=401):
    """Challenger anticipates A11 noise on both sides (default at floor). Win iff sx~ mu >= Q (Remark 1 domain)."""
    fcal = float(prior.pdf(.5)); s = np.linspace(tmin + 1e-4, 1.0, n)
    z = stats.norm.ppf((np.arange(200)+.5)/200)
    best = -np.inf
    for si in s:
        x = np.clip(si + sig*z, 1e-9, 1.0)
        win = np.mean(prior.sf(np.minimum(Q/x, 2.0)))
        best = max(best, win - cost(si - tmin, t0, tmin, fcal))
    x0 = np.clip(tmin + sig*z, 1e-9, 1.0) if tmin > 0 else np.full(200, 1e-9)
    return best - np.mean(prior.sf(np.minimum(Q/x0, 2.0)))

def rho_lock_noisy(t0, tmin, pi, sig):
    pr = uniform_prior(pi)
    f = lambda rho: campaign_gain_noisy(rho/3, t0, tmin, pr, sig)
    if f(1.5) <= 0: return float('nan')   # locked already at rho = 1.5 (Remark 1 domain ends)
    return optimize.brentq(f, 1.5, 3.2, xtol=1e-4)

def rho_lock_exact(t0, tmin, pi):
    f = lambda r: campaign_gain(r/3, t0, tmin, uniform_prior(pi))
    if f(1.5) <= 0: return float('nan')
    if f(3.0) > 0: return 3.0
    return optimize.brentq(f, 1.5, 3.0, xtol=1e-5)

if __name__ == '__main__':
    import math
    rho0, g = 3713761/1920000, math.log((1 + 31200/3713761)/(2335930/2363314))
    sig = 1.93*.04
    print('gamma_rho', round(g, 4))
    print('Table IV.13: t0, tmin | exact rho_lock (pi 0/.2/.4) | capped (4.24) | noisy | years')
    for t0 in (.5, .6, .75, .9):
        for tm in (0, .2, .35):
            ex = [rho_lock_exact(t0, tm, p) for p in (0, .2, .4)]
            nz = [rho_lock_noisy(t0, tm, p, sig) for p in (0, .2, .4)]
            yrs = [round(math.log(x/rho0)/g, 1) if x == x else 'nan' for x in ex]
            print(t0, tm, [round(x, 3) for x in ex], [round(rho_lock(t0, tm, p), 3) for p in (0, .2, .4)], [round(x, 3) for x in nz], yrs)
    print('Remark 2 (E_full, sigma_tau .04): (pi* deterministic, pi* noise-anticipating)')
    for tau in (.30, .40, .50):
        out = []
        for tm in (0, .2, .35):
            f = lambda p: campaign_gain_noisy(1.93/3, 1.93*tau, tm, uniform_prior(p), sig)
            ps = float('nan') if f(0) <= 0 else (optimize.brentq(f, 0, .99, xtol=1e-4) if f(.99) <= 0 else 1)
            out.append((round(pistar(1.93, tau, tm), 3), round(ps, 3)))
        print(tau, out)
    for tm in (0, .2, .35):
        f = lambda t: campaign_gain_noisy(1.93/3, 1.93*t, tm, uniform_prior(0), sig)
        print('noise-anticipating lock boundary, tmin', tm, round(optimize.brentq(f, max(.2, tm/1.93 + .03), .5, xtol=1e-5), 4))
