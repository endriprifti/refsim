"""Section IV core: Section III baseline model (A1-A2, floor t_min) + Section IV extensions.
All shares: resident units unless stated. tau = turnout share of E; t0 = rho*tau (tau_ab = 0).
"""
import numpy as np
from scipy import stats, optimize

TBAR = 1.0
QA = 1/3

# ---------------- priors ----------------
class Uniform:
    def __init__(self, lo, hi): self.lo, self.hi = lo, hi
    def sf(self, x):  # Pr[mu >= x]
        return np.clip((self.hi - np.asarray(x, float)) / (self.hi - self.lo), 0, 1)
    def cdf(self, x): return 1 - self.sf(x)
    def pdf(self, x):
        x = np.asarray(x, float); return np.where((x >= self.lo) & (x <= self.hi), 1/(self.hi-self.lo), 0.0)
    def mass(self, a, b):  # Pr[a <= mu <= b]
        return np.clip(np.minimum(b, self.hi) - np.maximum(a, self.lo), 0, None)/(self.hi-self.lo)

class BetaP:
    def __init__(self, a, b, lo=0.0, hi=1.0):
        self.d = stats.beta(a, b, loc=lo, scale=hi-lo); self.lo, self.hi = lo, hi
    def sf(self, x): return self.d.sf(x)
    def cdf(self, x): return self.d.cdf(x)
    def pdf(self, x): return self.d.pdf(x)
    def mass(self, a, b): return np.clip(self.d.cdf(b) - self.d.cdf(a), 0, None)

def uniform_prior(pi, mub=0.5):
    lo = pi*mub; return Uniform(lo, lo + 1 - pi)

def beta_matched(pi, mub=0.5, concentrate=1.0):
    """Beta(a,a) on [0,1] with the mean (1/2) and variance (1-pi)^2/12 of A2's uniform;
    concentrate=2 halves the variance."""
    var = (1-pi)**2/12/concentrate
    a = (1/(4*var) - 1)/2
    return BetaP(a, a), a

# ---------------- cost ----------------
def cost(t, t0, tmin, fcal):
    """campaign cost in units of B: fcal*t^2/(8 t0 (t0-tmin)); fcal = prior density at 1/2 (3.4)."""
    return fcal * t**2 / (8*t0*(t0 - tmin))

# ---------------- V3 (approval) with default at floor (Remark 1 domain Q >= tbar/2) ----------------
def v3_threshold(sx, sd, Q):
    return np.maximum(sd/(sx+sd) if (sx+sd) > 0 else 1.0, Q/sx if sx > 0 else np.inf)

def challenger_best(Q, t0, tmin, prior, fcal=None, tbar=TBAR, n=4001):
    """Challenger's best rate s in [tmin, tbar] with default at floor; returns (campaign, s, payoff).
    Payoff in units of B: Pr[mu >= thr(s)] - cost."""
    if fcal is None: fcal = float(prior.pdf(0.5))
    s = np.linspace(tmin if tmin > 0 else 1e-6, tbar, n)
    thr = np.array([v3_threshold(si, tmin, Q) if tmin > 0 else Q/si for si in s])
    thr = np.minimum(thr, 2.0)
    win = prior.sf(thr)
    pay = win - cost(s - tmin, t0, tmin, fcal)
    i = int(np.argmax(pay))
    # refine
    if 0 < i < n-1:
        f = lambda x: -(prior.sf(min(max(Q/x, tmin/(x+tmin) if tmin>0 else 0), 2.0)) - cost(x - tmin, t0, tmin, fcal))
        r = optimize.minimize_scalar(f, bounds=(s[i-1], s[i+1]), method='bounded', options={'xatol':1e-10})
        sbest, pbest = r.x, -r.fun
    else:
        sbest, pbest = s[i], pay[i]
    p0 = pay[0]  # t_x = 0
    return (pbest > p0 + 1e-12), sbest, pbest, p0

def campaign_gain(Q, t0, tmin, prior, fcal=None, tbar=TBAR, n=4001):
    """max over s in (tmin, tbar] of [win(s) - cost] minus win(tmin): >0 iff the challenger campaigns."""
    if fcal is None: fcal = float(prior.pdf(0.5))
    s = np.linspace(tmin + 1e-4, tbar, n)
    thr = np.maximum(Q/s, tmin/(s+tmin)) if tmin > 0 else Q/s
    pay = prior.sf(np.minimum(thr, 2.0)) - cost(s - tmin, t0, tmin, fcal)
    i = int(np.argmax(pay)); best = pay[i]
    if 0 < i < n-1:
        f = lambda x: -(prior.sf(min(max(Q/x, tmin/(x+tmin) if tmin>0 else 0), 2.0)) - cost(x - tmin, t0, tmin, fcal))
        r = optimize.minimize_scalar(f, bounds=(s[i-1], s[i+1]), method='bounded', options={'xatol':1e-10}); best = max(best, -r.fun)
    w0 = float(prior.sf(min(Q/tmin, 2.0))) if tmin > 0 else 0.0
    return best - w0

def v3_equilibrium(rho, tau, tmin, pi=0.0, mub=0.5, prior=None, qA=QA, tbar=TBAR, fcal=None):
    t0 = rho*tau; Q = qA*rho
    if prior is None: prior = uniform_prior(pi, mub)
    camp, s, pay, p0 = challenger_best(Q, t0, tmin, prior, fcal=fcal, tbar=tbar)
    sx = s if camp else tmin
    return dict(t0=t0, Q=Q, campaign=camp, sx=sx, sd=tmin,
                muhat=(Q/sx if sx > 0 else np.inf), prior=prior)

def pistar_capped(rho, tau, tmin, qA=QA, tbar=TBAR):
    t0 = rho*tau; Q = qA*rho
    return 2*(1 - Q/tbar - (tbar - tmin)**2/(8*t0*(t0 - tmin)))

def pistar(rho, tau, tmin, qA=QA, tbar=TBAR, prior_family='uniform', conc=1.0):
    """pi at which the challenger stops campaigning (uniform prior, mu_b = 1/2); bisection."""
    def g(pi):
        pr = uniform_prior(pi) if prior_family=='uniform' else beta_matched(pi, concentrate=conc)[0]
        return campaign_gain(qA*rho, rho*tau, tmin, pr, tbar=tbar)
    lo, hi = 0.0, 0.999
    if g(lo) <= 0: return float('nan')   # locked at every pi (pi* <= 0)
    if g(hi) > 0: return 1.0
    return optimize.brentq(g, lo, hi, xtol=1e-7)

def prchange_v3(rho, tau, tmin, pi=0.0, prior=None):
    e = v3_equilibrium(rho, tau, tmin, pi, prior=prior)
    if not e['campaign']: return 0.0
    return float(e['prior'].sf(e['muhat']))

# ---------------- poll model (A3, Proposition 10) with a general prior ----------------
def poll_model(Q, t0, tmin, prior, h, fcal=None, tbar=TBAR, nm=1201, ns=801):
    """Ex-ante Pr[change] and Pr[campaign]. Poll error U[-h,h];
    posterior = prior truncated to [m-h, m+h]; default at floor (Remark 1); kappa calibrated at fcal."""
    if fcal is None: fcal = float(prior.pdf(0.5))
    lo, hi = prior.lo, prior.hi
    m = np.linspace(lo - h, hi + h, nm); dm = m[1]-m[0]
    s = np.linspace(tmin + 1e-4, tbar, ns)
    thr = np.maximum(Q/s, tmin/(s+tmin)) if tmin > 0 else Q/s
    c = cost(s - tmin, t0, tmin, fcal)
    pc = pcamp = 0.0
    for mt in m:
        a, b = max(lo, mt - h), min(hi, mt + h)
        mass = float(prior.mass(a, b))
        if mass <= 1e-14: continue
        dens = mass/(2*h)                       # density of the poll reading
        win = prior.mass(np.maximum(thr, a), b)/mass  # posterior Pr[mu >= thr]
        pay = win - c
        i = int(np.argmax(pay))
        w0 = float(prior.mass(max(Q/tmin, a), b)/mass) if tmin > 0 else 0.0
        if pay[i] > w0 + 1e-12:
            pc += dens*dm*win[i]; pcamp += dens*dm
        else:
            pc += dens*dm*w0
    return pc, pcamp

if __name__ == '__main__':
    # Section III checkpoints
    for tm in (0, .2, .35):
        print('pi* tau=.40 Efull tmin', tm, round(pistar(1.93, .40, tm), 3), 'capped formula', round(pistar_capped(1.93,.40,tm),3))
    for rho in (1.93, 1.81):
        for tm in (0, .2, .35):
            print('Table III.2 rho', rho, 'tmin', tm, [round(prchange_v3(rho, t, tm), 3) for t in (.2, .3, .4, .5)])
