"""IV.1(c): stress test of A2 -- Beta priors with the same mean/variance as A2's uniform, and a concentrated one."""
import numpy as np
from s4_core import *

def a_of(pi, conc): return (3*conc/(1-pi)**2 - 1)/2
def prior_of(fam, pi, conc=1.0):
    if fam == 'U': return uniform_prior(pi)
    return BetaP(a_of(pi, conc), a_of(pi, conc))

def eq_generic(rho, tau, tmin, prior, fcal=None):
    Q, t0 = rho/3, rho*tau
    g = campaign_gain(Q, t0, tmin, prior, fcal=fcal)
    if g <= 0: return dict(camp=False, s=tmin, muhat=np.inf, pc=0.0, gain=g)
    c, s, p, p0 = challenger_best(Q, t0, tmin, prior, fcal=fcal)
    mh = max(Q/s, tmin/(s+tmin)) if tmin > 0 else Q/s
    return dict(camp=True, s=s, muhat=mh, pc=float(prior.sf(mh)), gain=g)

def pistar_fam(rho, tau, tmin, fam, conc=1.0, fixed_kappa=False):
    def g(pi):
        pr = prior_of(fam, pi, conc)
        fcal = 1/(1-pi) if fixed_kappa else None
        return campaign_gain(rho/3, rho*tau, tmin, pr, fcal=fcal)
    if g(0.0) <= 0: return float('nan')
    if g(0.95) > 0: return 0.95
    return optimize.brentq(g, 0.0, 0.95, xtol=1e-6)

def taulock_fam(rho, tmin, fam, pi=0.0, conc=1.0, fixed_kappa=False):
    pr = prior_of(fam, pi, conc); fcal = 1/(1-pi) if fixed_kappa else None
    g = lambda t: campaign_gain(rho/3, rho*t, tmin, pr, fcal=fcal)
    if g(0.5) <= 0: return float('inf')     # locked throughout D3
    lo = max(0.15, tmin/rho + 0.02)
    return optimize.brentq(g, lo, 0.5, xtol=1e-6)

if __name__ == '__main__':
    rho = 1.93
    print('alpha matched at pi=0,.2,.3,.4:', [round(a_of(p,1),3) for p in (0,.2,.3,.4)], 'conc', [round(a_of(p,2),3) for p in (0,.2,.3,.4)])
    print('f(1/2):', [round(float(prior_of('B',p,c).pdf(.5)),3) for p in (0,.3) for c in (1,2)])
    print('\npi* (E_full): uniform | Beta matched | Beta concentrated | conc fixed-kappa')
    for tau in (.30,.40,.50):
        for tm in (0,.2,.35):
            print(tau, tm, [round(pistar_fam(rho,tau,tm,f,c,fk),3) for (f,c,fk) in [('U',1,False),('B',1,False),('B',2,False),('B',2,True)]])
    print('\nlock boundary tau (E_full, pi=0): uniform | conc Beta(2.5) | conc fixed kappa ; and at pi=.3 matched/conc')
    for r in (1.93,1.81):
        for tm in (0,.2,.35):
            print(r, tm, [round(taulock_fam(r,tm,'U'),4), round(taulock_fam(r,tm,'B',0,2),4), round(taulock_fam(r,tm,'B',0,2,True),4),
                          round(taulock_fam(r,tm,'U',.3),4), round(taulock_fam(r,tm,'B',.3,1),4), round(taulock_fam(r,tm,'B',.3,2),4)])
    print('\nPr[change] and muhat at pi=0 / pi=.3 (E_full)')
    for tau in (.30,.40,.50):
        for tm in (0,.2,.35):
            out=[]
            for (f,pi,c) in [('U',0,1),('B',0,2),('U',.3,1),('B',.3,1),('B',.3,2)]:
                e=eq_generic(rho,tau,tm,prior_of(f,pi,c)); out.append(f"{e['pc']:.3f}({e['muhat'] if e['camp'] else float('nan'):.3f})")
            print(tau,tm,' '.join(out))
