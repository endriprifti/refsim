"""Table IV.1 and Table B.23-1: pass probabilities under V1-V11 at Section III equilibrium play, with and without A11 noise."""
import numpy as np, json
from s4_rules import *
from s4_units import qark_table
QK = qark_table(); WQ = np.array([q['E'] for q in QK]); WQ = WQ/WQ.sum()

def v7_rate(t0, tmin, q=.6):
    return .5*(tmin + np.sqrt(tmin**2 + 16*q*(1-q)*t0*(t0-tmin)))
def t0_s(t0, tmin, s):
    return .5*(tmin + np.sqrt(tmin**2 + 4*s*t0*(t0-tmin)))

def v10_pass(t0, tmin, sig, r=.5, sq=.05, k=7, N=400000, seed=1):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0, 1, N)
    d = rng.normal(0, sq, (N, 12)); d = d - (d*WQ).sum(1, keepdims=True)
    z1, z2 = rng.normal(size=N), rng.normal(size=N)
    sx = np.clip(t0 + sig*z1, 0, 1); sd = np.clip(t0 + sig*(r*z1 + np.sqrt(1-r*r)*z2), 0, 1)
    m = sd/(sx+sd)
    nat = mu > m
    mq = np.clip(mu[:, None] + d, 0, 1)
    k_yes = (mq > m[:, None]).sum(1)
    P = nat & (k_yes >= k)
    maj = mu > .5
    return P.mean(), (P & maj).mean()/maj.mean(), (P & ~maj).mean()  # (ex ante, Pr[pass|mu>1/2], Pr[pass & mu<1/2])

def row_values(code, rho, tau, tmin, sigtau, r=.5):
    t0 = rho*tau; sig = rho*sigtau; pr = uniform_prior(0)
    if code == 'V1':  ch = make_rule('V1', rho); sx = sd = t0
    elif code == 'V11': ch = make_rule('V1', rho); sx = sd = t0_s(t0, tmin, .5)
    elif code == 'V7': ch = make_rule('V7', rho, qS=.6); sx = sd = v7_rate(t0, tmin)
    elif code == 'V4':  ch = make_rule('V4', rho); sx = sd = tmin   # F28: opposition never campaigns
    elif code == 'V10': return v10_pass(t0, tmin, sig, r)
    else:
        ch = make_rule(code, rho); e = solve_eq(ch, t0, tmin, pr, n=601); sx, sd = e['sx'], e['sd']
        if not e['pure']: return (np.nan,)*3
    P, Pm, Pn = pass_probs(ch, sx, sd, pr, sig, r)
    return P, Pm/0.5, Pn

CODES = ['V1','V2','V3','V4','V5a','V6d','V7','V8','V9','V10','V11']
if __name__ == '__main__':
    res = {}
    for rho in (1.93, 1.81):
        for code in CODES:
            for tau in (.2, .3, .4, .5):
                for tm in (0, .2, .35):
                    for st in (0, .04):
                        v = row_values(code, rho, tau, tm, st)
                        res[f'{rho}|{code}|{tau}|{tm}|{st}'] = [float(x) for x in v]
    json.dump(res, open('table1.json', 'w'))
    for rho in (1.93, 1.81):
        print('rho', rho)
        for code in CODES:
            line = []
            for tau in (.2,.3,.4,.5):
                c0 = '/'.join(f"{res[f'{rho}|{code}|{tau}|{tm}|0.04'][1]:.3f}" for tm in (0,.2,.35))
                line.append(c0)
            print(code.ljust(4), ' | '.join(line))
        print('-- deterministic')
        for code in CODES:
            line = []
            for tau in (.2,.3,.4,.5):
                c0 = '/'.join(f"{res[f'{rho}|{code}|{tau}|{tm}|0'][1]:.3f}" for tm in (0,.2,.35))
                line.append(c0)
            print(code.ljust(4), ' | '.join(line))

def turnout_yes_correlation(rho=1.93, tau=.40, sigtau=.04, th=.5, N=200000, seed=2):
    """IV.1(a): correlation across the prior of turnout (share of E) and the Yes share of votes."""
    rng = np.random.default_rng(seed); sig = rho*sigtau
    mu = rng.uniform(0, 1, N); z1 = rng.normal(size=N); z2 = rng.normal(size=N)
    def corr(sx, sd):
        x = np.clip(sx + sig*z1, 0, 1); d = np.clip(sd + sig*(th*z1 + np.sqrt(1-th*th)*z2), 0, 1)
        Y, Nn = x*mu, d*(1-mu); ok = (Y+Nn) > 0
        return np.corrcoef(((Y+Nn)/rho)[ok], (Y/(Y+Nn))[ok])[0, 1]
    return dict(V3_demob_tau40=corr(1.0, .2), V3_demob_tau30=corr(.899, .2), V1=corr(rho*tau, rho*tau))
if __name__ == '__main__':
    print('turnout-Yes correlation:', {k: round(v, 3) for k, v in turnout_yes_correlation().items()})
