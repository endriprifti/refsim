"""IV.3: trajectory (bridge-maximum) test of organic apathy vs orchestrated abstention (Lemma 13, Table IV.7).
Stations ordered by the default-bloc share b_i at the reference Assembly election, strongest first.
Delta_i = log(tau_i / tau_i_ref); H0 (A14): Delta_i = shift + residual, residuals independent, mean 0,
|residual| <= cmax or Var <= sig^2.  Statistic Upsilon = max_k ( -Pi_k ),  Pi_k = sum_{i<=k} (Delta_(i) - mean).
Design (illustrative): residuals N(0, sig^2) clipped (winsorised) at +-3 sig; bloc shares Beta(a, a), mean 1/2,
SD .15; alternative = Regime A demobilisation (s_x = .899, t_min = .20), issue component iota = 1/2.
[REVISED B1] sizes from 20,000-200,000 null draws; permutation test with a data-set-specific critical value."""
import numpy as np
rng = np.random.default_rng(7)

def crit_azuma(n, cmax, alpha):   # Pr[U>=lam] <= 2 exp(-lam^2/(8 n cmax^2))
    return np.sqrt(8*n*cmax**2*np.log(2/alpha))
def crit_doob(n, sig, alpha):     # Kolmogorov/Doob: Pr[U>=lam] <= 4 n sig^2/lam^2
    return 2*sig*np.sqrt(n/alpha)
def crit_bridge(n, sig, alpha):   # asymptotic: Pr[sup bridge >= x] = exp(-2x^2)
    return sig*np.sqrt(n*np.log(1/alpha)/2)
def T_many(D):                     # rows = data sets, columns in the fixed (bloc) order
    D = np.atleast_2d(D); d = D - D.mean(1, keepdims=True); return np.maximum(0, (-np.cumsum(d, 1)).max(1))
def T_stat(D, order): return float(T_many(D[order])[0])

def alternative_D(b, sx, tmin, pi_al, t_ref, iota=0.5):
    mu = pi_al*(1-b) + (1-pi_al)*iota
    return np.log((tmin + (sx - tmin)*mu)/t_ref)
def noise(sig, shape): return np.clip(rng.normal(0, sig, shape), -3*sig, 3*sig)
def beta_a(sd_b=.15): return (0.25/sd_b**2 - 1)/2

def size_bridge(n, sig, alpha=.05, reps=100000):
    lam, hits, done = crit_bridge(n, sig, alpha), 0, 0
    while done < reps:
        m = min(5000 if n < 1000 else 500, reps - done); hits += (T_many(noise(sig, (m, n))) >= lam).sum(); done += m
    p = hits/reps; return p, np.sqrt(p*(1-p)/reps)

def null_quantile(n, sig, alpha=.05, reps=20000):
    out = []; done = 0
    while done < reps:
        m = min(5000 if n < 1000 else 500, reps - done); out.append(T_many(noise(sig, (m, n)))); done += m
    return float(np.quantile(np.concatenate(out), 1 - alpha))

def power(n, sig, p_al, crit, R=300, Bp=199, alpha=.05, sx=.899, tmin=.2):
    """crit = a number (fixed critical value) or 'perm' (data-set-specific permutation test, Bp permutations)."""
    a = beta_a(); rej = 0
    for _ in range(R):
        b = rng.beta(a, a, n); order = np.argsort(-b)
        Do = (alternative_D(b, sx, tmin, p_al, .5) + noise(sig, n))[order]
        T = T_many(Do)[0]
        if crit == 'perm':
            Tp = T_many(Do[np.array([rng.permutation(n) for _ in range(Bp)])])
            rej += (1 + (Tp >= T).sum())/(Bp + 1) <= alpha
        else: rej += T >= crit
    return rej/R

def smallest_pi(n, sig, crit, grid, target=.8, **kw):
    for p in grid:
        if power(n, sig, p, crit, **kw) >= target: return float(p)
    return None

CASES = [(61, .06), (373, .08), (5225, .15)]
if __name__ == '__main__':
    for n, sig in CASES:
        s, se = size_bridge(n, sig, reps=200000 if n < 1000 else 20000)
        print(f'I={n}: Azuma {crit_azuma(n, 3*sig, .05):.2f}  Doob {crit_doob(n, sig, .05):.2f}  asymptotic {crit_bridge(n, sig, .05):.3f} '
              f'(exact size {s:.4f}, SE {se:.4f})  simulated null 95% quantile {null_quantile(n, sig):.3f}', flush=True)
    grids = {61: np.arange(.08, .20, .005), 373: np.arange(.04, .10, .0025), 5225: np.arange(.02, .05, .0025)}
    for n, sig in CASES:
        perm = smallest_pi(n, sig, 'perm', grids[n])
        doob = smallest_pi(n, sig, crit_doob(n, sig, .05), np.arange(0, 1.0001, .01), R=600)
        az = smallest_pi(n, sig, crit_azuma(n, 3*sig, .05), np.arange(0, 1.0001, .01), R=600)
        print(f'I={n}: smallest pi with power .8 -- permutation {perm}  Doob {doob}  Azuma {az}', flush=True)
    b = np.linspace(.2, .8, 7); print('Delta pattern (pi=.3):', np.round(alternative_D(b, .899, .2, .3, .5), 3))
