"""IV.4: register error -- propagate rho ranges to Q_A, muhat, pi*, lock boundary, T5 margin; local rho_u error."""
import numpy as np
from s4_core import *
from s4_units import units

def taulock(rho, tm):
    g = lambda t: campaign_gain(rho/3, rho*t, tm, uniform_prior(0.0))
    return optimize.brentq(g, max(.15, tm/rho+.02), .6, xtol=1e-7)

def t5_campaign_gain(rho, tau, tm):
    """Regime B: opposition (challenger) vs Q_R = rho/2; reform side at floor."""
    return campaign_gain(rho/2, rho*tau, tm, uniform_prior(0.0))
def tmin_star_t5(rho, tau=.5):
    f = lambda tm: t5_campaign_gain(rho, tau, tm)
    lo, hi = 0.0, min(rho*tau - 1e-3, .95)
    if f(lo) > 0: return 0.0
    if f(hi) <= 0: return float('nan')
    return optimize.brentq(f, lo, hi, xtol=1e-6)
def tau_star_t5(rho, tm=.35):
    """[REVISED B1] tau above which the T5 opposition campaigns, searched only where t0 = rho*tau <= tbar = 1;
    nan = none (the opposition does not campaign at any feasible tau)."""
    f = lambda t: t5_campaign_gain(rho, t, tm)
    hi = 1/rho - 1e-9
    if f(hi) <= 0: return float('nan')
    return optimize.brentq(f, .45, hi, xtol=1e-6)

if __name__ == '__main__':
    print('rho  Q_A   pi*(.40;0/.2/.35)        lock tau (0/.2/.35)   tmin*(3.11)=4rho-7  tmin*(tau=.5)  tau*(tmin=.35)')
    for rho in (1.78, 1.806, 1.84, 1.90, 1.934, 1.97):
        ps = [round(pistar(rho, .40, tm), 3) for tm in (0, .2, .35)]
        tl = [round(taulock(rho, tm), 4) for tm in (0, .2, .35)]
        print(rho, round(rho/3, 4), ps, tl, round(4*rho-7, 3), round(tmin_star_t5(rho), 3), round(tau_star_t5(rho), 3))
    # national variance component
    for lab, (a, b) in (('full', (1.90, 1.97)), ('dom', (1.78, 1.84))):
        sd = (b-a)/np.sqrt(12); print(lab, 'SD rho (uniform)', round(sd,4), 'SD muhat', round(sd/3,4))
    # derivative of pi* (capped) wrt rho at fixed tau
    for tm in (0,.2,.35):
        d = (pistar_capped(1.94,.4,tm)-pistar_capped(1.92,.4,tm))/.02
        print('d pi*/d rho at tau .4 tmin', tm, round(d,3))
    # local: multiplicative error on rho_u, common + idiosyncratic
    rng = np.random.default_rng(3); R = 20000
    base = np.array([u['rho'] for u in units])
    for s_c, s_i in ((0, .05), (.05, .05), (.05, .10), (.05, .20)):
        rr = base[None, :]*np.exp(rng.normal(0, s_c, (R, 1)) + rng.normal(0, s_i, (R, len(base))))
        inf = (rr > 3).sum(1); low = (rr < 1.5).sum(1)
        print('local s_c', s_c, 's_i', s_i, 'infeasible count mean', round(inf.mean(),2), '90% band', np.quantile(inf,[.05,.95]),
              '| rho<1.5 mean', round(low.mean(),2), np.quantile(low,[.05,.95]))
    # per-unit probability of being misclassified (infeasible) at s_i=.10
    rr = base[None, :]*np.exp(rng.normal(0, .05, (R, 1)) + rng.normal(0, .10, (R, len(base))))
    pinf = (rr > 3).mean(0)
    amb = [(u['name'], round(u['rho'],2), round(p,2)) for u, p in zip(units, pinf) if .1 < p < .9]
    print('ambiguous (10-90%) at s_c=.05, s_i=.10:', amb)

if __name__ == '__main__':
    f = lambda r: (1 - r/2) - (1 - .35)**2/(2*r*(r - .7))      # (4.18) at t_min = .35
    print('rho_dom at which F28 E_dom clause fails at t_min=.35:', round(optimize.brentq(f, 1.6, 1.9), 4))
    print('rho = 1.81 (Section III rounding): t_min*', round(tmin_star_t5(1.81), 4), 'tau*', round(tau_star_t5(1.81), 4))
    print('tau_lock(pi) closed form (4.17):', {p: round((1/3)/(2*((2-p)/3)**1.5), 5) for p in (0, .3)})
    print('F29 crossover in tau = 0.471/rho:', {r: round(.471/r, 3) for r in (1.78, 1.84, 1.90, 1.97)})

if __name__ == '__main__':
    # [REVISED B1] Proposition 18's domain: numerical lock boundary (t_min = 0) across rho
    for p in (0.0, 0.3):
        out = []
        for r in (1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2):
            g = lambda t: campaign_gain(r/3, r*t, 0.0, uniform_prior(p))
            hi = min(.999/r, .6)
            out.append(f'{r}:' + (f'{optimize.brentq(g, .2, hi, xtol=1e-7):.4f}' if g(hi) > 0 else 'none'))
        print('Prop. 18 check, pi =', p, ' '.join(out))
