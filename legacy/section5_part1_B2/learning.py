# Section V.6 -- campaign learning model (B.41). All parameters illustrative unless stated.
import json, math
import numpy as np

c_star = 0.75                                  # target accuracy c_I*: 3:1 odds of voting the full-information preference
x_star = math.tan(math.pi * (c_star - 0.5)) ** 2   # = 1
L0, A0 = 30.0, 0.95                            # calibration: 30-day campaign informs 95% on a one-issue question
r_bar = x_star / (L0 * math.log(1 / A0))       # mean individual learning rate (per day), exponential heterogeneity
L_blk = 5.0                                    # Code 78(4)
QA = {'E_full': 1.93 / 3, 'E_dom': 1.81 / 3}


def accuracy(r, t, k, beta):
    # (5.x) c_I = 1/2 + arctan(sqrt(r t / k^beta)) / pi  (Sheppard 1899 via normal learning).
    return 0.5 + math.atan(math.sqrt(r * t / k ** beta)) / math.pi


def A_I(L_eff, k, beta):
    # share informed (c_I >= c_I*) with r ~ Exponential(mean r_bar).
    return math.exp(-k ** beta * x_star / (r_bar * L_eff))


def L_star(k, beta, A_target):
    return k ** beta * x_star / (r_bar * math.log(1 / A_target))


def omega_blk(L, k, beta, Lb=L_blk):
    x = r_bar * L / k ** beta; xb = r_bar * (L - Lb) / k ** beta
    return 1 - (xb / (1 + xb)) / (x / (1 + x))


out = dict(x_star=x_star, r_bar=r_bar)
# check accuracy formula against Monte Carlo of the Gaussian model
rng = np.random.default_rng(3)
k, beta, t, r = 4, 1.0, 20.0, r_bar
nv = 200000
sV2 = 1.0
v = rng.normal(0, math.sqrt(sV2 / k), size=(nv, k))
V = v.sum(1)
# simulate the beta_I = 2 structure exactly: each provision observed for t/k days, noise variance sigma_e2 per day
sigma_e2 = sV2 / (r)        # so that r = sV2 / sigma_e2
beta_sim = 2.0
tt = t / k
y = v + rng.normal(0, 1, size=(nv, k)) * np.sqrt(sigma_e2 / tt)
prior_prec = k / sV2
post_mean = (y * (tt / sigma_e2)) / (prior_prec + tt / sigma_e2)
m = post_mean.sum(1)
mc_acc = float((np.sign(m) == np.sign(V)).mean())
out['check_beta2'] = dict(k=k, t=t, mc=mc_acc, formula=accuracy(r, t, k, beta_sim))

# Table: L* by k, beta, target share
Ltab = []
for kk in (1, 3, 5, 10, 30, 124):
    row = dict(k=kk)
    for b in (0.5, 1.0, 2.0):
        for At in (0.8, 0.95):
            row[f'L_{b}_{At}'] = L_star(kk, b, At)
    Ltab.append(row)
out['Lstar'] = Ltab

# Table: reach, threshold and blackout at L = 30, beta = 1
reach = []
for kk in (1, 3, 5, 10, 30):
    A = A_I(30, kk, 1.0)
    row = dict(k=kk, A=A, omega=omega_blk(30, kk, 1.0))
    row['H_blk'] = math.sqrt(row['omega'])
    for tm in (0.0, 0.20, 0.35):
        tb = tm + (1 - tm) * A
        row[f'tbar_{tm}'] = tb
        row[f'muhat_{tm}'] = QA['E_full'] / tb if tb > 0 else float('inf')
        row[f'prch_{tm}'] = max(0.0, 1 - QA['E_full'] / tb)
    reach.append(row)
out['reach'] = reach
out['omega_bound'] = L_blk / 30

# levers [REVISED B2]: multiplicative form (5.13), L_eff = m_C L + m_C (m_I - 1) d_I with d_I <= L;
# the blackout's own effective learning time is m_C L_blk + m_C (m_I - 1) min(d_I, L_blk)
def rho_blk(Leff, Lb_eff, k, beta):
    x = r_bar * Leff / k ** beta; xb = r_bar * (Leff - Lb_eff) / k ** beta
    return 1 - (xb / (1 + xb)) / (x / (1 + x))

lev = []
for name, L, mC, mI, dI in [('statutory 30 days', 30, 1, 1, 0), ('booklet, two weeks (m_I = 2, d_I = 14)', 30, 1, 2, 14),
                            ('recognised campaigners and airtime (m_C = 1.5)', 30, 1.5, 1, 0), ('both', 30, 1.5, 2, 14),
                            ('60-day campaign, no booklet', 60, 1, 1, 0)]:
    assert dI <= L
    Leff = mC * L + mC * (mI - 1) * dI
    Lbe = mC * L_blk + mC * (mI - 1) * min(dI, L_blk)
    row = dict(lever=name, L_eff=Leff, Lblk_eff=Lbe)
    for kk in (1, 5, 10):
        row[f'A_k{kk}'] = A_I(Leff, kk, 1.0)
    row['rho_k10'] = rho_blk(Leff, Lbe, 10, 1.0)
    row['bound'] = Lbe / Leff; row['H_bound'] = math.sqrt(Lbe / Leff)
    lev.append(row)
out['levers'] = lev

# infeasibility thresholds of Corollary 7 [REVISED B2]: change infeasible iff tbar(L, k) * mu_hi < Q_A
def k_min(Q, pi, tmin, beta, L=30.0, mu_b=0.5):
    mu_hi = pi * mu_b + 1 - pi
    for kk in range(1, 100000):
        tb = tmin + (1 - tmin) * A_I(L, kk, beta)
        if tb * mu_hi < Q:
            return kk
    return None

thr = []
for lab, Q in QA.items():
    for pi in (0.0, 0.3):
        for tm in (0.0, 0.20, 0.35):
            thr.append(dict(denominator=lab, pi=pi, t_min=tm, **{f'beta_{b}': k_min(Q, pi, tm, b) for b in (0.5, 1.0, 2.0)}))
out['k_min'] = thr

# cue-taking sensitivity [REVISED B2]: a share gamma_c of uninformed supporters mobilisable on a bloc cue;
# feasibility returns iff gamma_c >= (A* - A_I)/(1 - A_I), A* = (Q/mu_hi - t_min)/(1 - t_min)
def gamma_c(k, Q=QA['E_full'], pi=0.0, tmin=0.0, beta=1.0, L=30.0):
    mu_hi = pi * 0.5 + 1 - pi
    Astar = (Q / mu_hi - tmin) / (1 - tmin); A = A_I(L, k, beta)
    return max(0.0, (Astar - A) / (1 - A))

out['gamma_c'] = {k: gamma_c(k) for k in (9, 10, 15, 20, 30)}

# calibration sensitivity of the k = 1 threshold [REVISED B2]
out['k1_threshold'] = {A0_: [QA['E_full'] / (tm + (1 - tm) * A0_) for tm in (0.0, 0.20, 0.35)] for A0_ in (0.95, 0.99)}

# 1994: draft published 6 Oct 1994, date decreed 11 Oct 1994, vote 6 Nov 1994 = 31 and 26 days
c94 = []
for kk, lab in [(5, 'five contested issues'), (10, 'ten contested issues'), (124, '124 articles')]:
    for b in (0.5, 1.0):
        A = A_I(31, kk, b)
        c94.append(dict(k=kk, label=lab, beta=b, A=A, cue_share=1 - A, A26=A_I(26, kk, b), Lstar80=L_star(kk, b, 0.8)))
out['y1994'] = c94
# T4: maximum campaign inside the 60 days
out['T4_Lmax'] = {rd: 60 - rd for rd in (14, 24, 30)}
json.dump(out, open('learning.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
