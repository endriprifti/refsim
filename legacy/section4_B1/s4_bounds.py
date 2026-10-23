"""IV.2: one-sided tail bounds and required margins."""
import numpy as np
from scipy import stats, optimize
def normal(k): return stats.norm.sf(k)
def cantelli(k): return 1/(1+k*k)
def chebyshev(k): return min(1.0, 1/(k*k))
def vp1(k): return 4/(9*(1+k*k)) if k > np.sqrt(5/3) else np.nan   # Mercadier-Strobel 2021, unimodal
def margin(alpha):
    return dict(normal=stats.norm.isf(alpha), vp1=np.sqrt(4/(9*alpha)-1), cantelli=np.sqrt(1/alpha-1), chebyshev=1/np.sqrt(alpha))
def hoeffding_margin(n, alpha): return np.sqrt(np.log(1/alpha)/(2*n))
def bernstein_margin(n, alpha, v=.25):
    L = np.log(1/alpha); f = lambda k: n*k*k/(2*v + 2*k/3) - L
    return optimize.brentq(f, 1e-9, 1.0)
def normal_margin(n, alpha, v=.25): return stats.norm.isf(alpha)*np.sqrt(v/n)
if __name__ == '__main__':
    print('k/sigma  normal  VP1  Cantelli  Chebyshev')
    for k in (1, 1.5, 2, 2.5, 3):
        print(k, round(normal(k),4), round(vp1(k),4) if k>np.sqrt(5/3) else 'n/a', round(cantelli(k),4), round(chebyshev(k),4))
    for a in (.10,.05,.01): print('alpha',a,{kk:round(v,3) for kk,v in margin(a).items()})
    for (lab,n) in [('national R 1.92M',1_920_000),('Korçë R',48_813),('Pustec R',1_481),('station ~700 reg / ~300 votes',300)]:
        for a in (.10,.01):
            print(lab, a, 'Hoeffding %.4f'%hoeffding_margin(n,a), 'Bernstein %.4f'%bernstein_margin(n,a), 'Normal %.4f'%normal_margin(n,a))
    # variance of approval share y = tau*yV with correlation
    tau, yV, st, sV = .35, .55, .04, .05
    for rc in (-.5, 0, .5, .8):
        var = (yV*st)**2 + (tau*sV)**2 + 2*rc*tau*yV*st*sV
        print('corr', rc, 'SD(y)=%.4f'%np.sqrt(var))

if __name__ == '__main__':
    a = np.array([53.6, 48.0, 50.8, 53.5, 46.8, 46.3, 44.8]); l = np.array([50.7, 45.7, 46.3, 50.9, 47.3])  # Table C.3
    print('A11 anchor: SD of first differences, Assembly 2001-2025 %.2f pp; local 2000-2015 %.2f pp' % (np.diff(a).std(ddof=1), np.diff(l).std(ddof=1)))

def conditional_moments(mu, rho=1.93, tau=.40, sx=1.0, sd=.20, sigtau=.04, th=.5, N=400000, seed=1):
    """[REVISED B1] Moments given the support level mu (what (4.12) needs): corr(turnout, Yes share | mu),
    SD of y = Yes/E, and the downside truncation of the challenger's capped rate (E[1 - x~]/sigma_s)."""
    r = np.random.default_rng(seed); sig = rho*sigtau
    z1, z2 = r.normal(size=N), r.normal(size=N)
    x = np.clip(sx + sig*z1, 0, 1); d = np.clip(sd + sig*(th*z1 + np.sqrt(1-th*th)*z2), 0, 1)
    Y, Nn = x*mu, d*(1-mu)
    return np.corrcoef((Y+Nn)/rho, Y/(Y+Nn))[0, 1], (Y/rho).std(), (sx - x.mean())/sig

if __name__ == '__main__':
    for mu in (.60, .65, .70, .75):
        c, sdy, tr = conditional_moments(mu)
        print(f'Regime A demobilisation, tau .40, mu {mu}: corr(turnout, Yes share | mu) {c:.3f}; SD(y | mu) {sdy:.4f}; mean shortfall {tr:.3f} sigma_s')
    agg = .028
    for n in (1_920_000, 50_000, 10_000, 2_500, 1_481, 300):
        ind = np.sqrt(1/(4*n)); print(f'n = {n:>9,}: individual SD {ind:.4f}; inflation of an aggregate SD of 0.028: {100*(np.sqrt(agg**2 + ind**2)/agg - 1):.2f}%')
