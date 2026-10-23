# Revision B2 [REVISED B2]: recomputes Tables V.5-V.7, the Proposition 22 variants, Corollary 6(ii),
# the indirect initiative, the whole-year loss rate and the worked-illustration dates.
# lambda_ee = 0: exact (exact_option); lambda_ee > 0: simulation without foresight of dissolutions (mc_nf).
import json, math, datetime as dt
import numpy as np
from pipeline import *

R = {}
LAM = 0.0563
N_MC = 100000

# ---------------------------------------------------------------- 1. Table V.6
rows = []
for o in OPTIONS:
    row = dict(option=o)
    for lam in (0.0, 0.0563, 0.1075):
        sl, ml = usable_slots(o, lam, np.random.default_rng(11), years=2000)
        row[f'slots_{lam}'] = sl; row[f'mandate_{lam}'] = ml
    for pres in ('earliest', 'latest'):
        row[f'exact_{pres}'] = exact_option(dict(BASE, president=pres), o)
        for cal_ in ('current', 'longrun'):
            row[f'{cal_}_{pres}'] = summarise(mc_nf(dict(BASE, president=pres), o, LAM, n=N_MC, seed=5, calendar=cal_))
    rows.append(row)
R['options'] = rows

# ---------------------------------------------------------------- 1b. Monte Carlo standard errors, clustered by calendar (20 runs share one)
se = {}
for o, pres in (('O0', 'earliest'), ('O0', 'latest'), ('O6', 'earliest'), ('O6', 'latest'), ('O9', 'latest')):
    out = mc_nf(dict(BASE, president=pres), o, LAM, n=N_MC, seed=5, calendar='current')
    x = np.array([1.0 if (r[0] == 'POLLED' and r[1] <= 2 * WY) else 0.0 for r in out])
    cm = x.reshape(-1, 20).mean(1)
    se[f'{o}_{pres}'] = dict(p=float(x.mean()), se_cluster=float(cm.std(ddof=1) / np.sqrt(len(cm))))
R['mc_se_current'] = se
# checker's current-mandate values (Pr_vote(24)), for the comparison line in V.2(c)
R['checker_current'] = dict(O0=(0.974, 0.269), O5=(0.992, 0.954), O6=(0.773, 0.713), O9=(0.992, 0.966))

# ---------------------------------------------------------------- 2. whole-year losses (TS0, constant hazard)
R['whole_year_losses_per_10000'] = {str(l): whole_year_losses(l) for l in (0.0563, 0.1075)}

# ---------------------------------------------------------------- 3. Proposition 22: prompt President, legal floor, decree-inclusive reading
def delay_profile(gap, year=1, president='earliest', decinc=False, dec=0):
    cal = Calendar(gen_timeline(0.0, np.random.default_rng(0)), 'O0', -WY * 60, WY * 80)
    out = []
    for wk in range(WY):
        c = WY * year + wk
        tp = schedule_decinc(cal, c, gap, president, dec) if decinc else schedule(cal, c, 'O0', gap, president)
        out.append(tp - c)
    return out

prompt = delay_profile(11); floor_ = delay_profile(5); decinc = delay_profile(11, decinc=True, dec=5)
R['prop22'] = dict(prompt_mean=float(np.mean(prompt)), prompt_jump=prompt[W_CUT + 1] - prompt[W_CUT],
                   floor_mean=float(np.mean(floor_)), floor_jump=floor_[W_CUT + 1] - floor_[W_CUT],
                   decinc_mean=float(np.mean(decinc)),
                   floor_valid=exact_option(dict(BASE, DEC=0, GMIN=5), 'O0'),
                   floor_valid_latest=exact_option(dict(BASE, DEC=0, GMIN=5, president='latest'), 'O0'),
                   decinc_valid=exact_option(dict(BASE), 'O0', decinc=True))

# ---------------------------------------------------------------- 4. Corollary 6(ii): saving S(wk, d) for fixed filing
def S_formula(wk, d, cut=W_CUT):
    if wk <= cut or wk - d >= 42:
        return d
    if cut < wk - d and wk <= 41:
        return 0
    if wk - d <= 41 < wk and wk - d > cut:
        return wk - 41
    return d + max(41 - wk, 0)          # wk - d <= cut < wk

mism = 0; checked = 0
for wk in range(WY):
    for d in range(1, 21):
        if wk - d < 0:
            continue
        S_true = d + prompt[wk] - prompt[wk - d]
        checked += 1; mism += (S_true != S_formula(wk, d))
R['cor6'] = dict(checked=checked, mismatches=mism, max_saving=max(d + prompt[w] - prompt[w - d] for w in range(WY) for d in range(1, 21) if w - d >= 0))

# ---------------------------------------------------------------- 5. indirect initiative (Remark 4)
ind = {}
for TA in (13, 26):
    ind[f'post_TS0_{TA}'] = exact_option(dict(BASE), 'O0', TA_post=TA)
    ind[f'pre_TS0_{TA}'] = exact_option(dict(BASE), 'O0', TA_pre=TA)
    ind[f'post_TS9_{TA}'] = exact_option(dict(BASE), 'O9', TA_post=TA)
R['indirect'] = ind

# ---------------------------------------------------------------- 6. Table V.5 (current law and reform package)
ROW = {r['option']: r for r in rows}      # Table V.5 reuses the Table V.6 runs (TS0 = O0, TS9 = O9)
R['V5'] = {f'{lab}_{k}': ROW[o][k] for lab, o in (('cur', 'O0'), ('ref', 'O9'))
           for k in ('exact_earliest', 'exact_latest', 'current_earliest', 'current_latest', 'longrun_earliest', 'longrun_latest')}
# Remark 4 under V1 with an opposing majority and no rule: C4 = Pr_vote (1 - Pr_ch), Pr_ch = 1/2 at pi = 0
R['V1_noRule'] = dict(Pr_ch=0.5, C4_24=R['V5']['cur_exact_earliest']['le104'] * 0.5)

# ---------------------------------------------------------------- 7. Table V.7 (T4), leads 30, 42 and 28 days
def t4(lead_list=(30, 42, 28), drevs=(7, 10, 12, 13, 15, 18, 19, 21, 24, 25, 28, 30, 31, 35)):
    fs = dt.date(2025, 9, 12); end = dt.date(2029, 9, 12)
    win = (dt.date(2029, 3, 12), dt.date(2029, 12, 12)); prevwin_end = dt.date(2025, 12, 12)
    elect = {dt.date(2027, 5, 16), dt.date(2029, 5, 13)}       # assumed polling Sundays [VERIFY]
    freeze = end - dt.timedelta(days=60)                         # Const. 65(4)
    days = [fs + dt.timedelta(days=i) for i in range((freeze - fs).days)]
    out = []
    for rd in drevs:
        row = dict(d_rev=rd)
        for lead in lead_list:
            ok = ok_nowin = 0
            for D in days:
                lo = D + dt.timedelta(days=rd + lead); hi = D + dt.timedelta(days=60)
                sund = [lo + dt.timedelta(days=i) for i in range((hi - lo).days + 1) if (lo + dt.timedelta(days=i)).weekday() == 6]
                ok_nowin += bool(sund)
                ok += any((not (win[0] <= s < win[1])) and (not (fs <= s < prevwin_end)) and s not in elect for s in sund)
            row[f'nowin_{lead}'] = ok_nowin / len(days); row[f'win_{lead}'] = ok / len(days)
            row[f'formula_{lead}'] = min(1.0, max(0.0, (61 - rd - lead) / 7))
        out.append(row)
    return out, len(days)

R['T4'], R['T4_days'] = t4()

# ---------------------------------------------------------------- 8. worked illustration dates (TS9)
fil = dt.date(2026, 8, 20); clear = dt.date(2027, 9, 21)
dec = clear + dt.timedelta(days=32)
e = dec + dt.timedelta(weeks=6); e_sun = e + dt.timedelta(days=(6 - e.weekday()) % 7)
l = clear + dt.timedelta(weeks=26); l_sun = l - dt.timedelta(days=(l.weekday() - 6) % 7)
R['illustration_TS9'] = dict(decree=str(dec), earliest=str(e_sun), months_earliest=(e_sun - fil).days / 30.44,
                             latest=str(l_sun), months_latest=(l_sun - fil).days / 30.44)

# exact timing of the illustrative initiative (filing week 85 = 20 Aug 2026), current law and TS9
FR5 = r0(50000, 0.85, 0.05); FR5 = 1 - FR5['acc']
FR5_R3 = 1 - r3(50000, 0.85, 0.05)['acc']
for lab, o, q in (('TS0_prompt', 'O0', dict(BASE, FR=FR5, p_CC=0.5)), ('TS0_latest', 'O0', dict(BASE, FR=FR5, p_CC=0.5, president='latest')),
                  ('TS9_prompt', 'O9', dict(BASE, FR=FR5_R3, p_CC=0.5)), ('TS9_latest', 'O9', dict(BASE, FR=FR5_R3, p_CC=0.5, president='latest'))):
    R['illustration_TS9'][lab] = exact_option(q, o, conditional_valid=False, start_weeks=[85])

# savings for the illustrative initiative (Corollary 6(ii)): 30-day Court review and cut-off repeal, prompt President
base_ill = R['illustration_TS9']['TS0_prompt']['E_months']
R['illustration_TS9']['saving_court30_months'] = base_ill - exact_option(dict(BASE, FR=FR5, p_CC=0.5, CC=5), 'O0', conditional_valid=False, start_weeks=[85])['E_months']
R['illustration_TS9']['saving_nocut_months'] = base_ill - exact_option(dict(BASE, FR=FR5, p_CC=0.5, cutoff=False), 'O0', conditional_valid=False, start_weeks=[85])['E_months']

# ---------------------------------------------------------------- 9. hazard anchors (A20)
from scipy import stats
T91 = (dt.date(2026, 10, 10) - dt.date(1991, 3, 31)).days / 365.25
R['hazard'] = dict(T1991=T91, rate=2 / T91, lo=stats.chi2.ppf(0.025, 4) / 2 / T91, hi=stats.chi2.ppf(0.975, 6) / 2 / T91,
                   T1998=(dt.date(2026, 10, 10) - dt.date(1998, 11, 28)).days / 365.25)
R['y1994'] = dict(decree_to_vote=(dt.date(1994, 11, 6) - dt.date(1994, 10, 11)).days,
                  publication_to_vote=(dt.date(1994, 11, 6) - dt.date(1994, 10, 6)).days)

json.dump(R, open('results_b2.json', 'w'), indent=1, default=float)
print('done')
