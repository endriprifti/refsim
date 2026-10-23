# Runs the computations of Section V, Part 1 for Tables V.2-V.4, V.12 and B.40-1 (lambda_ee = 0).
# Tables V.5-V.7 and the B2 checks are in run_b2.py [REVISED B2]. Output: results.json.
import json, math, datetime as dt
import numpy as np
from scipy import stats
from pipeline import *

R = {}
S_NAT = 50000

# ------------------------------------------------ A. verification rules (feeds the KQZ stage)
ver = []
for (unit, S) in [('national', S_NAT), ('Korce', 11030), ('Pustec', 444)]:
    for vL in (0.70, 0.85):
        for dS in (0.02, 0.05, 0.10, -0.02, -0.05, -0.10):
            a, b = r0(S, vL, dS), r3(S, vL, dS)
            err0 = (1 - a['acc']) if dS > 0 else a['acc']
            err3 = (1 - b['acc']) if dS > 0 else b['acc']
            ver.append(dict(unit=unit, S=S, vL=vL, dS=dS, J=a['J'], R0_err=err0, R0_work=a['work'],
                            R3_err=err3, R3_work=b['work'], R3_pband=b.get('pband')))
R['verification'] = ver
FR_R0_base = [v for v in ver if v['unit'] == 'national' and v['vL'] == 0.85 and v['dS'] == 0.05][0]['R0_err']
FR_R3_base = [v for v in ver if v['unit'] == 'national' and v['vL'] == 0.85 and v['dS'] == 0.05][0]['R3_err']
R['FR_base'] = dict(R0=FR_R0_base, R3=FR_R3_base)

# ------------------------------------------------ B. stage-level canonical chain (Prop. 21), one cycle
pmf, fail = col_pmf(BASE)
p_col = pmf.sum()
Etau_succ = (np.arange(len(pmf)) * pmf).sum() / p_col
hold_col = (np.arange(len(pmf)) * pmf).sum() + fail * BASE['W']
D_cap = 1 - 1.93 / 3          # Pr[change] on the capped branch, pi = 0, E_full (Section III: 0.357)
stage = dict(names=['Request/forms', 'Collection', 'KQZ verification', 'Court review', 'Decree and date'],
             p=[1.0, p_col, 1 - FR_R0_base, 0.5, None],
             h=[BASE['FORM'], hold_col, BASE['KQZ'] + BASE['p_cure'] * BASE['CURE'], BASE['CC'], BASE['DEC']])
# absorbing: ballot (this cycle), rolled over, mooted, abandoned, rej KQZ, rej CC
m = 1.0 * D_cap                 # omega_A = 1, no mooting rule: mooted iff the vote would change the law
share_this = (W_CUT + 1) / WY    # uniform clearance week: cleared by 15 March
Q = np.zeros((5, 5)); Rm = np.zeros((5, 6))
Q[0, 1] = 1.0
Q[1, 2] = p_col; Rm[1, 3] = 1 - p_col
Q[2, 3] = 1 - FR_R0_base; Rm[2, 4] = FR_R0_base
Q[3, 4] = 0.5; Rm[3, 5] = 0.5
Rm[4, 0] = (1 - m) * share_this; Rm[4, 1] = (1 - m) * (1 - share_this); Rm[4, 2] = m
N = np.linalg.inv(np.eye(5) - Q)
B = N @ Rm
tN = N @ np.array(stage['h'])
R['stage_chain'] = dict(N=N.round(4).tolist(), B_row1=B[0].tolist(), t_weeks=tN.round(2).tolist(),
                        p_col=p_col, Etau_succ=Etau_succ, hold_col=hold_col, m=m, share_this=share_this,
                        absorbing=['ballot (this cycle)', 'rolled over', 'mooted', 'abandoned', 'rejected KQZ', 'rejected Court'])

# ------------------------------------------------ C. lever table, exact lifted chain (filed request, p_CC = 1)
def P(**kw):
    q = dict(BASE); q['FR'] = FR_R0_base; q.update(kw); return q

levers = [
    ('Baseline (current law, R0, illustrative collection)', P()),
    ('Fee abolished (phi_f = 0)', P(phi_f=0.0)),
    ('Fee doubled (phi_f = 0.5)', P(phi_f=0.5)),
    ('Collection period W = 52 weeks', P(W=52)),
    ('Collection period W = 13 weeks', P(W=13)),
    ('E-signatures, half the lines (e_s = 0.5)', P(e_s=0.5)),
    ('Verification R3 (band rule)', P(FR=FR_R3_base)),
    ('Thin list (margin +2%, v_L = 0.70), R0', P(FR=[v for v in ver if v['unit']=='national' and v['vL']==0.70 and v['dS']==0.02][0]['R0_err'])),
    ('Thin list (margin +2%, v_L = 0.70), R3', P(FR=[v for v in ver if v['unit']=='national' and v['vL']==0.70 and v['dS']==0.02][0]['R3_err'])),
    ('KQZ stage as in 2012-13 (42 weeks)', P(KQZ=42, p_cure=0.0)),
    ('Court review 30 days', P(CC=5)),
    ('15 March cut-off repealed (one day a year kept)', P(cutoff=False)),
    ('President sets the latest feasible date', P(president='latest')),
]
lev = []
for name, q in levers:
    r = chain_results(q)
    lev.append(dict(lever=name, p_col=float(col_pmf(q)[0].sum()), Pr_ballot=float(r['B']['POLLED']),
                    le24=float(r['cum'][104]), le48=float(r['cum'][208]),
                    E_months=float(r['ET_polled'] * 12 / WY)))
R['levers'] = lev

# conditional-valid chain (for C4 and the calendar table check)
R['chain_valid_O0'] = {k: (float(v) if not isinstance(v, dict) else {kk: float(vv) for kk, vv in v.items()})
                      for k, v in chain_results(dict(BASE), conditional_valid=True).items() if k in ('B', 'cum', 'ET_polled')}
R['chain_valid_O0_latest'] = {k: (float(v) if not isinstance(v, dict) else {kk: float(vv) for kk, vv in v.items()})
                             for k, v in chain_results(dict(BASE, president='latest'), conditional_valid=True).items() if k in ('B', 'cum', 'ET_polled')}

# ------------------------------------------------ D. timing options: moved to run_b2.py (Table V.6) [REVISED B2]

# ------------------------------------------------ E. Assembly-initiated chains (T3, T5) and local (T7/T8)
def sim_assembly(review_weeks, pres, n=20000, seed=7, request_weeks=0, option='O0'):
    rng = np.random.default_rng(seed)
    cyc = gen_timeline(0.0, rng); cal = Calendar(cyc, option, -WY * 60, WY * 80)
    res = []
    for k in range(n):
        D = int(rng.integers(0, 4 * WY * 3))
        a = (D - W_FS) % CYC
        if a >= CYC - 9:        # Const. 65(4): no laws in the 60 days before the end of the mandate
            continue
        c = D + request_weeks + review_weeks
        tp = schedule(cal, c, option, BASE['DEC'] + BASE['GMIN'], pres)
        res.append(tp - D)
    return np.array(res)

asm = {}
for lab, rw, req in [('T3', 9, 0), ('T5', 9, 2)]:
    for pres in ('earliest', 'latest'):
        d = sim_assembly(rw, pres, request_weeks=req)
        asm[f'{lab}_{pres}'] = dict(mean_months=float(d.mean() * 12 / WY), median=float(np.median(d) * 12 / WY),
                                    p10=float(np.percentile(d, 10) * 12 / WY), p90=float(np.percentile(d, 90) * 12 / WY),
                                    max=float(d.max() * 12 / WY), le12=float((d <= WY).mean()), le24=float((d <= 2 * WY).mean()))
R['assembly'] = asm


def sim_local(variant, n=20000, seed=9, pres='earliest'):
    # T7: valid local initiative; local window Code 119(3) approximated as weeks 13-39 of local polling years.
    rng = np.random.default_rng(seed)
    t0 = -WY * 60
    blockL = np.zeros(WY * 140, bool)
    y = LOC_ABS
    while y - 4 * WY > t0:
        y -= 4 * WY
    while y < WY * 80:
        a = (y // WY) * WY
        blockL[a + 13 - t0:a + 40 - t0] = True
        y += 4 * WY
    p = dict(BASE)
    pmf, fail = col_pmf(p); pmf = pmf / pmf.sum()
    out = []
    for k in range(n):
        f = int(rng.integers(0, 4 * WY * 3))
        tau = rng.choice(len(pmf), p=pmf)
        t = f + p['FORM'] + tau
        if t % WY > W_DEP_LAST:
            t = (t // WY + 1) * WY
        t += p['KQZ'] + (p['CURE'] if rng.random() < p['p_cure'] else 0)
        if variant in ('LT2',):
            t += p['CC']
        c = t; gap = p['DEC'] + p['GMIN']
        if variant in ('LT1', 'LT2'):
            Y = c // WY if c % WY <= W_CUT else c // WY + 1
            while True:
                cand = [u for u in range(max(Y * WY, c + gap), Y * WY + WY) if not blockL[u - t0]]
                if cand:
                    tp = cand[0] if pres == 'earliest' else cand[-1]; break
                Y += 1
        else:  # LT3: annual-slot rule does not reach T7-T8; no cut-off
            u = c + gap
            while blockL[u - t0]:
                u += 1
            tp = u
        out.append(tp - f)
    d = np.array(out)
    return dict(le12=float((d <= WY).mean()), le24=float((d <= 2 * WY).mean()), E_months=float(d.mean() * 12 / WY))

R['local'] = {v: sim_local(v) for v in ('LT1', 'LT2', 'LT3')}     # local timing readings LT1-LT3 [REVISED B2]
R['local']['LT1_latest'] = sim_local('LT1', pres='latest')

# ------------------------------------------------ F. T4 table: moved to run_b2.py (Table V.7) [REVISED B2]

# ------------------------------------------------ G. C4 and C4' (V.1(e))
D_vals = {'tau0.25': 0.0, 'tau0.30_tmin0': 0.0, 'tau0.30_tmin0.20': 0.284, 'tau0.30_tmin0.35': 0.231,
          'tau0.40_Efull': 1 - 1.93 / 3, 'tau0.40_Edom': 1 - 1.81 / 3, 'V1': 1.0}
R['C4'] = dict(D=D_vals)   # Pr_vote values moved to run_b2.py (Table V.5) [REVISED B2]
fa = {}
for dS in (-0.02, -0.05, -0.10):
    v = [x for x in ver if x['unit'] == 'national' and x['vL'] == 0.85 and x['dS'] == dS][0]
    fa[str(dS)] = dict(R0=v['R0_err'], R3=v['R3_err'])
R['C4prime'] = fa

# ------------------------------------------------ H. worked illustration (filing 20 Aug 2026)
fil = dt.date(2026, 8, 20)
wk = WY * 1 + (fil.timetuple().tm_yday - 1) // 7
ill = {}
for lab, q in [('current_earliest', dict(BASE, FR=FR_R0_base, p_CC=0.5)),
               ('current_latest', dict(BASE, FR=FR_R0_base, p_CC=0.5, president='latest')),
               ('reform_R3_earliest', dict(BASE, FR=FR_R3_base, p_CC=0.5))]:
    r = chain_results(q, start_week=wk % CYC)
    ill[lab] = dict(B=float(r['B']['POLLED']), le24=float(r['cum'][104]), le12=float(r['cum'][52]),
                    E_months=float(r['ET_polled'] * 12 / WY), abandon=float(r['B']['ABANDON']),
                    rejK=float(r['B']['REJ_KQZ']), rejC=float(r['B']['REJ_CC']))
ill['filing_week_abs'] = wk
# deterministic median trace (dates)
tau_med = math.ceil(BASE['W0'] / BASE['K0'])
d_forms = fil + dt.timedelta(days=20)
d_dep = d_forms + dt.timedelta(weeks=tau_med)
d_kqz = d_dep + dt.timedelta(days=90)
d_cure = d_kqz + dt.timedelta(days=45)
d_cc = d_cure + dt.timedelta(days=60)
d_dec = d_cc + dt.timedelta(days=32)
first_sun_2028 = next(dt.date(2028, 1, 1) + dt.timedelta(days=i) for i in range(7) if (dt.date(2028, 1, 1) + dt.timedelta(days=i)).weekday() == 6)
last_sun_2028 = next(dt.date(2028, 12, 31) - dt.timedelta(days=i) for i in range(7) if (dt.date(2028, 12, 31) - dt.timedelta(days=i)).weekday() == 6)
ill['trace'] = dict(tau_med_weeks=tau_med, forms=str(d_forms), deposit=str(d_dep), kqz=str(d_kqz), cure=str(d_cure),
                    court=str(d_cc), decree_by=str(d_dec), earliest_poll=str(first_sun_2028), latest_poll=str(last_sun_2028),
                    months_earliest=(first_sun_2028 - fil).days / 30.44, months_latest=(last_sun_2028 - fil).days / 30.44)
R['illustration'] = ill

# ------------------------------------------------ I. misc anchors
R['anchors'] = dict(phi_B=39 / 208, lam_1991=2 / 35.529, lam_rule3=-math.log(0.05) / 27.866,
                    g2013=(dt.date(2013, 12, 22) - dt.date(2013, 4, 9)).days,
                    win2013_end=str(dt.date(2013, 12, 9)), sundays_after=[str(dt.date(2013, 12, 15)), str(dt.date(2013, 12, 22))])

json.dump(R, open('results.json', 'w'), indent=1, default=float)
print('done')
