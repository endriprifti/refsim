"""Worked illustration (ILLUSTRATIVE inputs; Handoff II figures for E, R)."""
import numpy as np
from scipy import stats
from s4_core import *
from s4_rules import make_rule, solve_eq, pass_probs
from s4_margins import rho_lock_exact
from s4_margins2 import mu_needed
from s4_beta import prior_of, eq_generic, pistar_fam
from s4_verify import rule_exact, seq_hoeffding, setup
import math
E, R = 3_713_761, 1_920_000
rho = E/R; tau, tmin, pi, st, r = 0.35, 0.20, 0.20, 0.04, 0.5
sig = rho*st; t0 = rho*tau; Q = rho/3
print('rho', round(rho,4), 't0', round(t0,4), 'Q_A', round(Q,4), 'ceil(E/3)', math.ceil(E/3), 'sigma_s', round(sig,4))
print('pi* (tau .35, tmin .2)', round(pistar(rho, tau, tmin),4), 'capped formula', round(pistar_capped(rho,tau,tmin),4))
pr = uniform_prior(pi)
e = v3_equilibrium(rho, tau, tmin, pi)
print('campaign', e['campaign'], 'sx', round(e['sx'],4), 'muhat', round(e['muhat'],4))
ch = make_rule('V3', rho)
P0 = pass_probs(ch, e['sx'], e['sd'], pr, 0); P1 = pass_probs(ch, e['sx'], e['sd'], pr, sig, r)
pmaj = float(pr.mass(.5, pr.hi))
print('Pr[change] det', round(P0[0],4), 'noisy', round(P1[0],4), '| Pr[pass|mu>1/2] det', round(P0[1]/pmaj,4), 'noisy', round(P1[1]/pmaj,4))
print('Pr[pass|mu] (Lemma 12):', {m: round(float(stats.norm.cdf((e['sx'] - Q/m)/sig)), 3) for m in (.65, .70, .75)})
mu9 = Q/(e['sx'] - stats.norm.ppf(.9)*sig); print('mu_.9 (4.3)', round(mu9, 4), 'bisection', round(mu_needed('V3', rho, t0, tmin, sig, pi=pi), 4), 'expected Yes votes', round(e['sx']*mu9*R))
# Beta-matched prior at the same pi
pb = prior_of('B', pi, 1.0); eb = eq_generic(rho, tau, tmin, pb)
print('Beta matched: campaign', eb['camp'], 'Pr[change]', round(eb['pc'],4), 'pi*_B', round(pistar_fam(rho,tau,tmin,'B'),4))
# register range
for rr in (1.90, 1.97):
    eq = v3_equilibrium(rr, tau, tmin, pi)   # [B1] at rho = 1.90 the optimum is interior (s_x < 1), so muhat > Q_A
    print('register rho', rr, 'Q_A', round(rr/3,4), 's_x', round(eq['sx'],4), 'muhat', round(eq['muhat'],4), 'pi*', round(pistar(rr,tau,tmin),4))
# V6d and V1 contrast
for code in ('V6d','V1'):
    ch2 = make_rule(code, rho) if code != 'V1' else make_rule('V1', rho)
    if code == 'V1': sx = sd = t0
    else:
        e2 = solve_eq(ch2, t0, tmin, pr, n=601); sx, sd = e2['sx'], e2['sd']
    Pd = pass_probs(ch2, sx, sd, pr, sig, r)
    print(code, 'rates', round(sx,3), round(sd,3), 'Pr[pass|maj]', round(Pd[1]/pmaj,4), 'Pr[pass|minority]', round(Pd[2]/(1-pmaj),4), 'mu_.9', mu_needed(code, rho, t0, tmin, sig, pi=pi))
# poll
h = 0.5*(1-pi)/2; print('poll H0=.5: Pr[change], Pr[campaign]', [round(x,3) for x in poll_model(Q, t0, tmin, pr, h)])
# signatures: J = 58,000, v = .88
S = 50000; J = 58000; v = .88; V = round(J*v); D = V/S - 1
print('signatures V', V, 'Delta', round(D,4))
print(' R0 error', round(rule_exact('R0', S, D, v)[0],4), '| R3 (err, lines, Pr[full])', [round(x,4) for x in rule_exact('R3', S, D, v)], '| R4', [round(float(x),4) for x in seq_hoeffding(S, D, v, reps=4000)])
# trajectory test illustration: expected max drift with n=5,225, pi_al=.2
from s4_trajectory import alternative_D, crit_bridge, crit_doob, crit_azuma, T_stat
rng = np.random.default_rng(5); n = 5225; sd_b = .15; a = (0.25/sd_b**2-1)/2
Ts = []
for _ in range(500):
    b = rng.beta(a, a, n); Dv = alternative_D(b, e['sx'], tmin, 0.2, .5) + np.clip(rng.normal(0,.15,n),-.45,.45); Ts.append(T_stat(Dv, np.argsort(-b)))
print('trajectory: E[T]', round(np.mean(Ts),2), 'crit bridge/doob/azuma', round(crit_bridge(n,.15,.05),2), round(crit_doob(n,.15,.05),2), round(crit_azuma(n,.45,.05),2))
# hitting time illustration
gE = 31200/E; dR = 1 - 2335930/2363314
g = math.log((1+gE)/(1-dR)); print('g_rho per year', round(g,4), 'dR', round(dR,4))
for t0_ in (.60, .75):
    rl = rho_lock_exact(t0_, .2, 0.0); print('resident turnout', t0_, 'rho_lock', round(rl, 3), 'years', round(math.log(rl/rho)/g, 1))
