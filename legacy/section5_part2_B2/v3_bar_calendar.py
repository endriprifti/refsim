"""V.3(d): effective length of the three-year bar for a repeat initiative
(Table V.17) [REVISED B2].  Readings: BP the bar constrains the repeat polling
day (pipeline may run during the bar); BF the bar constrains the filing of the
repeat request.  Prompt and latest President as in Part 1 (A18).
BF rules: a collection completed in December deposits on 1 January (Code 2003,
127(2)); clearance = deposit + 200 days; clearance after 15 March assigns the
next year (119(6))."""
import numpy as np, datetime as dt
from dateutil.relativedelta import relativedelta as rd
from calendar_al import FEAS, first_feasible_on_or_after, latest_in_year_on_or_after, \
    poll_after_clearance, D

rng = np.random.default_rng(5252)
MONTH = 30.4375
first_polls = [s for y in range(2026, 2042) for s in FEAS[y]]

def months(a, b):
    return (b - a).days / MONTH

# collection time (Part 1, A19): lognormal median 31/1.2 weeks, sigma 0.5,
# success iff <= W = 31 weeks; deposit window 1 Jan - 30 Nov (Code 127(2));
# statutory stages: 200 days (Part 1 mean 6.58 months, deterministic here).
M_COL, SIG, W = 31 / 1.2, 0.5, 31.0
STAT_DAYS = 200

def draw_hcol():
    while True:
        h = np.ceil(np.exp(np.log(M_COL) + SIG * rng.standard_normal()))
        if h <= W:
            return int(h)

def deposit_date(d):
    if d.month == 12:
        return D(d.year + 1, 1, 1)
    return d

res = {k: [] for k in ('BPp', 'BPl', 'BFp', 'BFl')}
by_phase = {}
for t_pol in first_polls:
    anniv = t_pol + rd(years=3)
    # BP: polling day >= anniversary; prepared challenger
    p1 = first_feasible_on_or_after(anniv)
    l1 = latest_in_year_on_or_after(anniv)
    res['BPp'].append(months(t_pol, p1))
    res['BPl'].append(months(t_pol, l1))
    ph = (t_pol.year - 2025) % 4
    by_phase.setdefault(ph, []).append(months(t_pol, p1))
    # BF: request filed on the anniversary
    for _ in range(40):
        h = draw_hcol()
        dep = deposit_date(anniv + dt.timedelta(weeks=h))
        clear = dep + dt.timedelta(days=STAT_DAYS)
        p2 = poll_after_clearance(clear, 77, latest=False)
        l2 = poll_after_clearance(clear, 77, latest=True)
        res['BFp'].append(months(t_pol, p2))
        res['BFl'].append(months(t_pol, l2))

print('Table V.17: effective bar c_eff (months from first polling day to the earliest')
print('feasible repeat polling day); first polls uniform over feasible Sundays 2026-2041')
for k, lab in (('BPp', 'BP poll-bar, prompt'), ('BPl', 'BP poll-bar, latest'),
               ('BFp', 'BF filing-bar, prompt'), ('BFl', 'BF filing-bar, latest')):
    x = np.array(res[k])
    print('%-24s mean %5.1f  median %5.1f  P10 %5.1f  P90 %5.1f  min %5.1f  max %5.1f  Pr[>40m] %.3f' %
          (lab, x.mean(), np.median(x), np.percentile(x, 10), np.percentile(x, 90), x.min(), x.max(),
           (x > 40).mean()))
print('\nBP prompt by year of first vote in the Assembly cycle (0 = election year 2025/29/...)')
for ph in sorted(by_phase):
    x = np.array(by_phase[ph])
    print(' phase %d: n=%d mean %.1f  Pr[delayed > 36.5 m] %.3f  max %.1f' %
          (ph, len(x), x.mean(), (x > 36.5).mean(), x.max()))
x = np.array(res['BPp'])
print('Share of first votes whose anniversary falls in a national window: %.3f' % (x > 36.5).mean())

# conditional median of the collection time (A19), used in the illustration
from scipy import stats
p_ok = stats.norm.cdf((np.log(W) - np.log(M_COL)) / SIG)
med = np.exp(np.log(M_COL) + SIG * stats.norm.ppf(p_ok / 2))
print('Pr[qualify within %.0f weeks] = %.4f; median collection time given qualification = %.2f weeks' % (W, p_ok, med))
