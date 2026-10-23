"""V.3(c), F23: outcome of one Regime A vote (A1-A2, Remark 1: default at its
floor).  [REVISED B2] In the lock with t_min = 0 nobody votes: the cell is
printed as 'no ballots' (the t_min -> 0+ limit of the Yes-majority share is 1/2)."""
import numpy as np
from game import bloc_prior, campaigns, RHO_FULL, RHO_DOM

def outcomes(rho, tau, tmin, pi):
    Q = rho / 3
    t0 = rho * tau
    pr = bloc_prior(pi)
    lo, hi = pr.lo, pr.hi
    camp, s, u = campaigns(Q, t0, tmin, pr, 1.0 / (1 - pi))
    if not camp:
        s = tmin                      # mutual abstention: only the floor votes
    w = lambda a, b: max(0.0, min(b, hi) - max(a, lo)) / (hi - lo)
    mu_hat = max(Q / s, tmin / (s + tmin)) if camp else np.inf
    mu_V = tmin / (s + tmin) if s > 0 else 0.5   # Yes majority among votes
    if not camp:
        mu_V = 0.5                    # floor votes sincerely: y_V = mu
    p_change = w(mu_hat, hi) if camp else 0.0
    p_void_yesmaj = w(mu_V, min(mu_hat, hi))
    p_void_nomaj = w(lo, mu_V)
    return camp, s, p_change, p_void_yesmaj, p_void_nomaj

print('Table V.16: one Regime A vote; void = no valid change (default stands)')
print('rho    tau  tmin  pi   campaign  s_x    Pr[change] Pr[void, Yes majority of votes] Pr[void, No majority]')
for rho in (RHO_FULL, RHO_DOM):
    for tmin in (0.0, 0.20, 0.35):
        for pi in (0.0, 0.3):
            for tau in (0.25, 0.30, 0.40, 0.50):
                if rho * tau >= 1: continue
                c, s, a, b, d = outcomes(rho, tau, tmin, pi)
                if (not c) and tmin == 0.0:
                    print('%.3f %.2f %.2f %.1f %-5s %.3f  %.3f  -- (no ballots; limit t_min->0+: %.3f)' % (rho, tau, tmin, pi, c, s, a, b))
                    continue
                print('%.3f %.2f %.2f %.1f %-5s %.3f  %.3f  %.3f  %.3f' % (rho, tau, tmin, pi, c, s, a, b, d))
