"""IV.6: support needed for Pr[pass | mu] >= p (Table IV.12), by bisection with 400,000 common random normal pairs."""
import numpy as np
from scipy import stats
from s4_core import *
from s4_rules import make_rule, solve_eq
from s4_table1 import v7_rate
RNG = np.random.default_rng(2026); NN = 400_000
Z1, Z2 = RNG.standard_normal(NN), RNG.standard_normal(NN)
def draws(sx, sd, sig, th=.5):
    x = np.clip(sx + sig*Z1, 0, 1) if sx > 0 else np.zeros(NN)
    d = np.clip(sd + sig*(th*Z1 + np.sqrt(1-th*th)*Z2), 0, 1) if sd > 0 else np.zeros(NN)
    return x, d
def passes(code, rho, x, d, mu):
    Y, N = x*mu, d*(1-mu); maj = Y > N
    if code == 'V1': return maj
    if code == 'V3': return maj & (Y >= rho/3)
    if code == 'V6d': return maj & (Y >= 1/3)
    if code == 'V2': return maj & (Y + N >= .4*rho)
    if code == 'V7': return (Y >= .6*(Y+N)) & (x > 0)
    if code == 'V8': return (Y >= .55*(Y+N)) & (Y >= rho/3) & (x > 0)
    if code in ('V4', 'V4c'): return Y > rho/2
def rates(code, rho, t0, tmin, pi=0.0):
    pr = uniform_prior(pi)
    if code == 'V1': return t0, t0
    if code == 'V7': s = v7_rate(t0, tmin); return s, s
    if code == 'V4c': return 1.0, tmin
    if code == 'V4':   # [REVISED B1] optimal rate, interior where the cap is not optimal
        if campaign_gain(rho/2, t0, tmin, pr) <= 0: return tmin, tmin
        return challenger_best(rho/2, t0, tmin, pr)[1], tmin
    e = solve_eq(make_rule(code, rho), t0, tmin, pr, n=601)
    return (e['sx'], e['sd']) if e['pure'] else (None, None)
def mu_needed(code, rho, t0, tmin, sig, p=.9, pi=0.0, th=.5):
    sx, sd = rates(code, rho, t0, tmin, pi)
    if sx is None: return 'Z2'
    x, d = draws(sx, sd, sig, th)
    f = lambda m: passes(code, rho, x, d, m).mean() - p
    if f(1.0) < 0: return 'L' if sx <= tmin + 1e-9 else '>1'
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo+hi)/2
        if f(mid) >= 0: hi = mid
        else: lo = mid
    return hi
if __name__ == '__main__':
    sig = 1.93*.04; z9 = stats.norm.isf(.1)
    RHOS = [1.0, 1.25, 1.5, 1.75, 1.93, 2.2, 2.5, 3.0, 3.5, 4.0]
    for t0 in (.60, .75):
        print('t0', t0)
        for code in ('V1','V2','V3','V4','V4c','V6d','V7','V8'):
            row = [mu_needed(code, r, t0, .2, sig) for r in RHOS]
            print('|', code, '|', ' | '.join(v if isinstance(v,str) else f'{v:.3f}' for v in row), '|')
    print('closed V3 capped rho=1.93, t0=.75:', round((1.93/3)/(1 - z9*sig),4), 'numeric', round(mu_needed('V3',1.93,.75,.2,sig),4))
    # tmin sensitivity at rho=1.93, t0=.75
    for tm in (0,.2,.35):
        print('tmin',tm, [ (c, round(mu_needed(c,1.93,.75,tm,sig),3) if not isinstance(mu_needed(c,1.93,.75,tm,sig),str) else mu_needed(c,1.93,.75,tm,sig)) for c in ('V1','V3','V6d','V7')])
    # sigma sensitivity (V3 capped closed form) at rho 1.93
    for st in (0, .02, .04, .06):
        s = 1.93*st; print('sigma_tau', st, 'V3 capped mu_.9 =', round((1.93/3)/(1 - z9*s),4), 'feasible up to rho', round((1 - z9*1.93*st)*3,3))
