# Section V, Part 1 -- pipeline chain, calendar and Monte Carlo (B.38-B.40).
#
# Time unit: one week. Idealised calendar: 52-week year, week 0 starts 1 January.
# Absolute week 0 = 1 January 2025. Assembly first sitting 12 Sep 2025 = week 36.
# Period of the lifted chain: 208 weeks (four calendar years 2025-2028 mod 4).
import math
import numpy as np
from scipy import sparse, stats
from scipy.sparse.linalg import spsolve

WY = 52            # weeks per idealised year
CYC = 208          # weeks per Assembly mandate (Const. 65(1))
W_FS = 36          # week of first sitting (12 Sep 2025 -> week 36)
W_CUT = 10         # week containing 15 March (day 73): Code 119(6) cut-off
W_DEP_LAST = 47    # week containing 30 November: Code 127(2) deposit window
EL_PH = 190        # Assembly polling day: week 18 (11 May 2025) = 18 weeks before mandate end
LOC_ABS = 2 * WY + 19   # local polling day 2027 (week 19, cf. 14 May 2023), every 208 weeks

# ---------------------------------------------------------------- parameters
BASE = dict(
    FORM=3,          # forms within 20 days (Code 127(1))
    W=31,            # collection period set by KQZ (2026: 27 Aug 2026-31 Mar 2027 = 216 days)
    K0=1.2,          # median capacity ratio at W0, phi_f0 (illustrative)
    sig_c=0.5,       # dispersion of log capacity (illustrative)
    W0=31, phi_f0=0.25,
    phi_f=0.25,      # fee per line / canvassing cost per line (illustrative)
    e_s=0.0, ce_ratio=0.1,   # share of e-signatures; their cost relative to paper
    KQZ=13,          # 90 days from deposit (Code 128(2); purposive reading, as in 2026 practice)
    CURE=7, p_cure=0.5,      # 5 + 30 + 10 days (Code 128), illustrative probability
    FR=0.0,          # false-rejection probability at KQZ (set from verification rule)
    CC=9,            # 60 days (Code 129(2), Const. 152(1))
    p_CC=1.0,
    DEC=5,           # decree: 32 days in 2013 (within 45 days, Const. 152(3))
    GMIN=6,          # prompt President's decree-to-polling lead (A18; the legal floor is the 30-day campaign)
    president='earliest',   # 'earliest' = prompt President; 'latest' = latest feasible Sunday
    cutoff=True,
)


def line_cost(p):
    # cost per line relative to canvassing cost c_J (A19).
    return (1 - p['e_s']) * (1 + p['phi_f']) + p['e_s'] * p['ce_ratio']


def col_pmf(p):
    # P(tau_c = i), i = 1..W, and P(fail). tau_c = ceil(W0 / Ktilde),
    # Ktilde ~ LN(ln(K0 * cost0/cost), sig_c): capacity ratio at W0 (A19).
    cost0 = 1 + p['phi_f0']
    mu = math.log(p['K0'] * cost0 / line_cost(p))
    W, W0 = p['W'], p['W0']
    pmf = np.zeros(W + 1)
    for i in range(1, W + 1):
        # tau = i  <=>  W0/i <= K < W0/(i-1)
        lo = W0 / i
        hi = np.inf if i == 1 else W0 / (i - 1)
        Fh = 1.0 if hi == np.inf else stats.norm.cdf((math.log(hi) - mu) / p['sig_c'])
        pmf[i] = Fh - stats.norm.cdf((math.log(lo) - mu) / p['sig_c'])
    fail = 1 - pmf.sum()
    return pmf, fail


# ---------------------------------------------------------------- calendar (O0) for the lifted chain
def phase_info(option='O0'):
    # feasible[p], barred[p] for p in 0..207 (abs week mod 208), current law O0.
    feas = np.ones(CYC, bool)
    for p in range(CYC):
        a = (p - W_FS) % CYC
        if a >= CYC - 26 or a < 13:      # Code 119(2): 6 months before end .. 3 months after first sitting
            feas[p] = False
    el = (W_FS + EL_PH) % CYC           # Assembly polling day (p = 18)
    loc = LOC_ABS % CYC                 # local polling day (p = 123)
    feas[el] = False; feas[loc] = False  # Code 119(1): no referendum on an election date
    last = np.zeros(CYC, bool)
    for y in range(4):
        idx = [y * WY + w for w in range(WY) if feas[y * WY + w]]
        if idx:
            last[idx[-1]] = True
    return feas, last


# ---------------------------------------------------------------- exact lifted chain (Lemma 15)
def build_chain(p, start='filing', conditional_valid=False):
    # Homogeneous chain on (substate, phase). Returns P (sparse), index helpers.
    feas, last = phase_info()
    pmf, fail = col_pmf(p)
    if conditional_valid:
        pmf = pmf / pmf.sum(); fail = 0.0
    W = p['W']
    surv = np.array([pmf[i:].sum() + fail for i in range(W + 2)])
    haz = np.zeros(W + 1)
    for i in range(1, W + 1):
        haz[i] = pmf[i] / surv[i] if surv[i] > 0 else 0.0
    GAP = p['DEC'] + p['GMIN']
    subs = []
    subs += [('FORM', k) for k in range(p['FORM'], 0, -1)]
    subs += [('COL', i) for i in range(1, W + 1)]
    subs += [('WAITDEP', 0)]
    subs += [('KQZ', k) for k in range(p['KQZ'], 0, -1)]
    subs += [('CURE', k) for k in range(p['CURE'], 0, -1)]
    subs += [('CC', k) for k in range(p['CC'], 0, -1)]
    for fl in ('this', 'next'):
        subs += [('DG', k, fl) for k in range(GAP, 0, -1)]
        subs += [('READY', 0, fl)]
    absorbing = ['POLLED', 'ABANDON', 'REJ_KQZ', 'REJ_CC']
    sidx = {s: i for i, s in enumerate(subs)}
    nS = len(subs)
    nT = nS * CYC
    nA = len(absorbing)
    aidx = {a: nT + j for j, a in enumerate(absorbing)}

    def st(s, ph):
        return sidx[s] * CYC + (ph % CYC)

    rows, cols, vals = [], [], []

    def add(i, j, v):
        if v > 0:
            rows.append(i); cols.append(j); vals.append(v)

    def flag_at_clear(ph):
        if not p['cutoff']:
            return 'this'
        return 'this' if (ph % WY) <= W_CUT else 'next'

    def flag_update(fl, ph_next):
        # crossing into a new calendar year: 'next' becomes 'this'; a missed 'this' rolls (stays 'this')
        if ph_next % WY == 0:
            return 'this'
        return fl

    for s in subs:
        for ph in range(CYC):
            i = st(s, ph); nx = ph + 1
            kind = s[0]
            if kind == 'FORM':
                k = s[1]
                add(i, st(('FORM', k - 1), nx) if k > 1 else st(('COL', 1), nx), 1.0)
            elif kind == 'COL':
                c = s[1]
                h = haz[c]
                dep_target = st(('KQZ', p['KQZ']), nx) if (nx % WY) <= W_DEP_LAST else st(('WAITDEP', 0), nx)
                add(i, dep_target, h)
                if c < W:
                    add(i, st(('COL', c + 1), nx), 1 - h)
                else:
                    add(i, aidx['ABANDON'], 1 - h)
            elif kind == 'WAITDEP':
                add(i, st(('KQZ', p['KQZ']), nx) if (nx % WY) == 0 else st(('WAITDEP', 0), nx), 1.0)
            elif kind in ('KQZ', 'CURE'):
                k = s[1]
                if k > 1:
                    add(i, st((kind, k - 1), nx), 1.0)
                else:
                    pc = p['p_cure'] if kind == 'KQZ' else 0.0
                    add(i, st(('CURE', p['CURE']), nx), pc)
                    add(i, st(('CC', p['CC']), nx), (1 - pc) * (1 - p['FR']))
                    add(i, aidx['REJ_KQZ'], (1 - pc) * p['FR'])
            elif kind == 'CC':
                k = s[1]
                if k > 1:
                    add(i, st(('CC', k - 1), nx), 1.0)
                else:
                    fl = flag_at_clear(nx)   # clearance occurs at the start of week nx
                    add(i, st(('DG', GAP, fl), nx), p['p_CC'])
                    add(i, aidx['REJ_CC'], 1 - p['p_CC'])
            elif kind == 'DG':
                k, fl = s[1], s[2]
                fl2 = flag_update(fl, nx)
                add(i, st(('DG', k - 1, fl2), nx) if k > 1 else st(('READY', 0, fl2), nx), 1.0)
            elif kind == 'READY':
                fl = s[2]
                pp = ph % CYC
                can = feas[pp] and fl == 'this' and (p['president'] == 'earliest' or last[pp])
                if can:
                    add(i, aidx['POLLED'], 1.0)
                else:
                    add(i, st(('READY', 0, flag_update(fl, nx)), nx), 1.0)
    for a in absorbing:
        add(aidx[a], aidx[a], 1.0)
    P = sparse.csr_matrix((vals, (rows, cols)), shape=(nT + nA, nT + nA))
    return P, dict(nT=nT, nA=nA, aidx=aidx, absorbing=absorbing, st=st, subs=subs)


def chain_results(p, conditional_valid=False, horizons=(52, 104, 208), start_week=None):
    P, info = build_chain(p, conditional_valid=conditional_valid)
    nT = info['nT']
    Q = P[:nT, :nT].tocsc(); Rm = P[:nT, nT:].tocsc()
    I = sparse.identity(nT, format='csc')
    A = (I - Q)
    tvec = spsolve(A, np.ones(nT))                 # expected steps to absorption, N 1
    B = spsolve(A, Rm.toarray())                   # absorption probabilities, N R
    # starting distribution: FORM at start phase
    s0 = info['st'](('FORM', p['FORM']), 0) - 0
    starts = [info['st'](('FORM', p['FORM']), ph) for ph in (range(CYC) if start_week is None else [start_week])]
    pi0 = np.zeros(nT + info['nA']); pi0[starts] = 1.0 / len(starts)
    Babs = pi0[:nT] @ B
    texp = pi0[:nT] @ tvec
    # time to POLLED: expected conditional steps via N (R_polled weighted)
    jP = info['absorbing'].index('POLLED')
    # E[T * 1{polled}] = pi0 N (N R_polled)  (standard: N applied to absorption-probability vector)
    bP = B[:, jP]
    TP = spsolve(A, bP)
    ET_polled = (pi0[:nT] @ TP) / Babs[jP]
    # distribution by forward iteration
    v = pi0.copy(); PT = P.T.tocsr(); cum = {}
    maxh = max(horizons) + 1
    hist = []
    for n in range(1, maxh + 200):
        v = PT @ v
        hist.append(v[nT + jP])
    hist = np.array(hist)
    for h in horizons:
        cum[h] = hist[h]   # absorbed by step h+1 <=> polling week - filing week <= h
    return dict(B=dict(zip(info['absorbing'], Babs)), E_steps=texp, ET_polled=ET_polled - 1,
                cum=cum, hist=hist, nT=nT)


# ---------------------------------------------------------------- verification rules (Lemma 10, Prop. 20)
def r0(S, vL, dS):
    J = round(S * (1 + dS) / vL); VL = round(vL * J)
    n = math.ceil(0.05 * J); thr = math.ceil(n * S / J)
    acc = stats.hypergeom.sf(thr - 1, J, VL, n)
    return dict(J=J, VL=VL, n=n, acc=acc, work=n)


def r3(S, vL, dS, nfix=1500):
    J = round(S * (1 + dS) / vL); VL = round(vL * J)
    if J < 3000:   # every line counted (F44)
        return dict(J=J, VL=VL, n=J, acc=1.0 if VL >= S else 0.0, work=J, pband=1.0)
    n = nfix
    x = np.arange(0, n + 1)
    pmf = stats.hypergeom.pmf(x, J, VL, n)
    a = np.where(x / n >= 0.92, 0.02, 0.05)
    proj = J * x / n
    acc_s = proj >= (1 + a) * S
    rej_s = proj < (1 - a) * S
    band = ~(acc_s | rej_s)
    exact_acc = 1.0 if VL >= S else 0.0
    acc = pmf[acc_s].sum() + pmf[band].sum() * exact_acc
    pb = pmf[band].sum()
    return dict(J=J, VL=VL, n=n, acc=acc, work=n + J * pb, pband=pb)


# ---------------------------------------------------------------- Monte Carlo (options, early elections)
OPTIONS = ['O0', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9']
O9_MAXW = 26   # O9: polling within 26 weeks (6 months) of clearance where a feasible week exists


def gen_timeline(lam_year, rng, horizon_cycles=40):
    # Assembly cycles from far past; early dissolution hazard lam_year (A20).
    lam_w = lam_year / WY
    s = W_FS - CYC * 12
    cycles = []
    while s < W_FS + CYC * horizon_cycles:
        u = rng.exponential(1 / lam_w) if lam_w > 0 else np.inf
        if u < EL_PH:
            d = s + int(u); el = d + 6; s_next = el + 4      # 30-45 days to polling; first sitting <=10 days after results
            cycles.append(dict(s=s, early=True, d=d, el=el, s_next=s_next))
        else:
            cycles.append(dict(s=s, early=False, el=s + EL_PH, s_next=s + CYC))
            s_next = s + CYC
        s = s_next
    return cycles


class Calendar:
    def __init__(self, cycles, option, t0, t1):
        self.t0, self.t1 = t0, t1
        n = t1 - t0
        self.susp = np.zeros(n, bool); self.block = np.zeros(n, bool); self.elect = np.zeros(n, bool)
        pre, post = 26, 13
        if option == 'O2':
            pre, post = 13, 4
        for c in cycles:
            if c['early']:
                end = (c['el'] + 4) if option == 'O1' else (c['s_next'] + post)
                self._set(self.susp, c['d'], end); self._set(self.block, c['d'], end)
            else:
                if option == 'O1':
                    self._set(self.block, c['el'] - 8, c['el'] + 5)
                else:
                    self._set(self.block, c['s_next'] - pre, c['s_next'] + post)
            self._set(self.elect, c['el'], c['el'] + 1)
        y = LOC_ABS
        while y - 4 * WY > t0:
            y -= 4 * WY
        while y < t1:
            self._set(self.elect, y, y + 1); y += 4 * WY
        self.option = option

    def _set(self, arr, a, b):
        a = max(a, self.t0); b = min(b, self.t1)
        if b > a:
            arr[a - self.t0:b - self.t0] = True

    def is_susp(self, t):
        return self.susp[t - self.t0]

    def feasible(self, t):
        o = self.option
        i = t - self.t0
        if o == 'O4':
            return self.elect[i]
        if o == 'O3':
            return self.elect[i] or (not self.block[i])
        return (not self.block[i]) and (not self.elect[i])

    def susp_end(self, t):
        while self.is_susp(t):
            t += 1
        return t


def advance(cal, t, d):
    # d active weeks from t, the clock stopped during suspension (Code 119(4)-(5), A20).
    while d > 0:
        if not cal.is_susp(t):
            d -= 1
        t += 1
    return t


def slots_for(option, t):
    # candidate slot windows (list of (start, end, cutoff_week_abs)) from year of t onward.
    y = t // WY
    out = []
    for yy in range(y, y + 6):
        b = yy * WY
        if option in ('O0', 'O1', 'O2', 'O3', 'O8', 'NOCUT', 'O9'):
            out.append((b, b + WY - 1, b + W_CUT))
        elif option == 'O5':
            out.append((b, b + 25, b + W_CUT)); out.append((b + 26, b + 51, b + 36))
        elif option == 'O6':
            out.append((b + 46, b + 50, b + 34))
        elif option == 'O7':
            out.append((b + 20, b + 24, b + 8)); out.append((b + 46, b + 50, b + 34))
    return out


def schedule(cal, c, option, gap, president):
    if option == 'O9':
        cand = [t for t in range(c + gap, c + O9_MAXW + 1) if cal.feasible(t)]
        if cand:
            return cand[0] if president == 'earliest' else cand[-1]
        t = c + O9_MAXW + 1
        while not cal.feasible(t):
            t += 1
        return t
    if option == 'O4':
        t = c + gap
        while not cal.feasible(t):
            t += 1
        return t
    for (a, b, cut) in slots_for(option, c):
        if option == 'NOCUT':
            pass
        elif option in ('O6', 'O7'):
            if c > cut:
                continue
        else:
            # slot of the half-year/year containing c is open only if c is by its cut-off
            if c > b:
                continue
            if a <= c <= b and c > cut:
                continue
        cand = [t for t in range(max(a, c + gap), b + 1) if cal.feasible(t)]
        if cand:
            return cand[0] if president == 'earliest' else cand[-1]
    raise RuntimeError('no slot')


def simulate_one(p, cal, filing, option, rng, conditional_valid=False, record=False):
    gap = p['DEC'] + p['GMIN'] + (7 if option == 'O8' else 0)
    t = advance(cal, filing, p['FORM'])
    # collection
    cost0 = 1 + p['phi_f0']
    mu = math.log(p['K0'] * cost0 / line_cost(p))
    while True:
        K = math.exp(mu + p['sig_c'] * rng.standard_normal())
        tau = math.ceil(p['W0'] / K)
        if tau <= p['W'] or not conditional_valid:
            break
    if tau > p['W']:
        return ('ABANDON', None, None)
    t = advance(cal, t, tau)
    if t % WY > W_DEP_LAST:
        t = (t // WY + 1) * WY
    dep = t
    cure = rng.random() < p['p_cure']
    t = advance(cal, t, p['KQZ'] + (p['CURE'] if cure else 0))
    if rng.random() < p['FR']:
        return ('REJ_KQZ', None, None)
    t = advance(cal, t, p['CC'])
    if rng.random() >= p['p_CC']:
        return ('REJ_CC', None, None)
    c = t
    opt = option if (p['cutoff'] or option == 'O9') else 'NOCUT'
    tp = schedule(cal, c, opt, gap, p['president'])
    guard = 0
    while cal.is_susp(tp) and guard < 10 and not (option == 'O4' and cal.elect[tp - cal.t0]):  # re-decree
        c2 = cal.susp_end(tp)
        tp = schedule(cal, c2, opt, gap, p['president']); guard += 1
    return ('POLLED', tp - filing, tp - dep)


def mc(p, option='O0', lam=0.0, n=20000, seed=1, conditional_valid=False, fixed_filing=None):
    rng = np.random.default_rng(seed)
    out = []
    t0, t1 = -WY * 60, WY * 80
    for k in range(n):
        if k % 200 == 0:
            cyc = gen_timeline(lam, rng)
            cal = Calendar(cyc, option, t0, t1)
        filing = fixed_filing if fixed_filing is not None else int(rng.integers(0, 4 * WY * 3))
        out.append(simulate_one(p, cal, filing, option, rng, conditional_valid))
    return out


def summarise(out, horizons=(52, 104, 208)):
    n = len(out)
    pol = [o for o in out if o[0] == 'POLLED']
    d = np.array([o[1] for o in pol]) if pol else np.array([])
    dd = np.array([o[2] for o in pol]) if pol else np.array([])
    res = dict(n=n, polled=len(pol) / n)
    for h in horizons:
        res[f'le{h}'] = float((d <= h).sum() / n) if len(d) else 0.0
    if len(d):
        res['E_months'] = float(d.mean() * 12 / WY)
        res['p99_dep_months'] = float(np.percentile(dd, 99) * 12 / WY)
        res['max_dep_months'] = float(dd.max() * 12 / WY)
    return res


def usable_slots(option, lam, rng, years=4000):
    # long-run usable polling slots per four years and mean mandate length (years).
    cyc = gen_timeline(lam, rng, horizon_cycles=int(years / 4) + 20)
    t0, t1 = -WY * 60, WY * (years + 40)
    cal = Calendar(cyc, option, t0, t1)
    cnt = 0; Y = 0
    for y in range(0, years):
        b = y * WY
        if option == 'O4':
            cnt += sum(cal.feasible(t) for t in range(b, b + WY)); Y += 1; continue
        for (a, e, cut) in [s for s in slots_for(option, b) if s[0] // WY == y]:
            if any(cal.feasible(t) for t in range(a, e + 1)):
                cnt += 1
        Y += 1
    lens = [(c['s_next'] - c['s']) / WY for c in cyc if c['s'] > 0 and c['s_next'] < t1]
    return 4 * cnt / Y, float(np.mean(lens))


# ================================================================ Revision B2 additions [REVISED B2]
# (1) exact evaluation at lambda_ee = 0 for every timing scenario: the pre-clearance stages are
#     deterministic given (collection time, cure), so the clearance week is enumerated exactly and the
#     calendar schedule is applied to it (no simulation error).
# (2) Monte Carlo without foresight of dissolutions for lambda_ee > 0, on the current-mandate calendar
#     (Assembly first sitting 12 Sep 2025, filings 2025-28) or the long-run (stationary) calendar.

def clearance_cases(p, f, conditional_valid=True, TA_pre=0):
    # list of (probability, clearance week c, deposit week) for filing week f, lambda_ee = 0
    pmf, fail = col_pmf(p)
    if conditional_valid:
        pmf = pmf / pmf.sum()
    out = []
    for tau in range(1, p['W'] + 1):
        if pmf[tau] <= 0:
            continue
        t = f + p['FORM'] + tau
        if t % WY > W_DEP_LAST:
            t = (t // WY + 1) * WY
        dep = t
        for cure, pc in ((1, p['p_cure']), (0, 1 - p['p_cure'])):
            if pc <= 0:
                continue
            c = dep + p['KQZ'] + (p['CURE'] if cure else 0) + p['CC'] + TA_pre
            out.append((pmf[tau] * pc * (1 - p['FR']) * p['p_CC'], c, dep))
    return out


def schedule_decinc(cal, c, gap, president, dec):
    # decree-inclusive reading of Code 2003 Art. 119(6): the cut-off is tested on the decree date c + dec
    cd = c + dec
    for (a, b, cut) in slots_for('O0', cd):
        if cd > b:
            continue
        if a <= cd <= b and cd > cut:
            continue
        cand = [t for t in range(max(a, c + gap), b + 1) if cal.feasible(t)]
        if cand:
            return cand[0] if president == 'earliest' else cand[-1]
    raise RuntimeError('no slot')


def exact_option(p, option='O0', horizons=(52, 104, 208), conditional_valid=True, TA_pre=0, TA_post=0,
                 decinc=False, start_weeks=None):
    # exact Pr[vote <= h], mean months, 99th percentile deposit-to-vote, for filing uniform over one period
    rng = np.random.default_rng(0)
    cal = Calendar(gen_timeline(0.0, rng), option, -WY * 60, WY * 80)
    gap = p['DEC'] + p['GMIN'] + (7 if option == 'O8' else 0) + TA_post
    opt = option if (p['cutoff'] or option == 'O9') else 'NOCUT'
    fs = list(range(CYC, 2 * CYC)) if start_weeks is None else start_weeks
    tot = 0.0; dist = {}; depd = {}
    for f in fs:
        for (w, c, dep) in clearance_cases(p, f, conditional_valid, TA_pre):
            if decinc:
                tp = schedule_decinc(cal, c, gap, p['president'], p['DEC'])
            else:
                tp = schedule(cal, c, opt, gap, p['president'])
            dist[tp - f] = dist.get(tp - f, 0.0) + w / len(fs)
            depd[tp - dep] = depd.get(tp - dep, 0.0) + w / len(fs)
            tot += w / len(fs)
    ks = np.array(sorted(dist)); ps = np.array([dist[k] for k in ks])
    res = dict(polled=float(tot))
    for h in horizons:
        res[f'le{h}'] = float(ps[ks <= h].sum())
    res['E_months'] = float((ks * ps).sum() / ps.sum() * 12 / WY)
    kd = np.array(sorted(depd)); pd_ = np.array([depd[k] for k in kd]); cdf = np.cumsum(pd_) / pd_.sum()
    res['p99_dep_months'] = float(kd[np.searchsorted(cdf, 0.99)] * 12 / WY)
    return res


def gen_timeline_current(lam_year, rng, horizon_cycles=10):
    # current-mandate calendar: regular past up to the first sitting of 12 Sep 2025 (week 36),
    # constant dissolution hazard from then on (A20)
    lam_w = lam_year / WY
    cycles = []
    s = W_FS - CYC * 12
    while s < W_FS:
        cycles.append(dict(s=s, early=False, el=s + EL_PH, s_next=s + CYC)); s += CYC
    while s < W_FS + CYC * horizon_cycles:
        u = rng.exponential(1 / lam_w) if lam_w > 0 else np.inf
        if u < EL_PH:
            d = s + int(u); el = d + 6; s_next = el + 4
            cycles.append(dict(s=s, early=True, d=d, el=el, s_next=s_next))
        else:
            cycles.append(dict(s=s, early=False, el=s + EL_PH, s_next=s + CYC)); s_next = s + CYC
        s = s_next
    return cycles


def projected_cycles(cycles, td):
    # cycles as known at time td: realised history; the current cycle's dissolution is known only if
    # d <= td; afterwards regular four-year terms are projected (no foresight, A20)
    out = []
    for c in cycles:
        if c['s_next'] <= td or (c['early'] and c['d'] <= td):
            out.append(c)
            if c['s'] <= td < c['s_next']:
                s = c['s_next']; break
            continue
        if c['s'] <= td:
            out.append(dict(s=c['s'], early=False, el=c['s'] + EL_PH, s_next=c['s'] + CYC))
            s = c['s'] + CYC; break
    else:
        s = cycles[-1]['s_next']
    for _ in range(6):
        out.append(dict(s=s, early=False, el=s + EL_PH, s_next=s + CYC)); s += CYC
    return out


def simulate_one_nf(p, cal, cycles, filing, option, rng, cache, pmf, TA_post=0):
    # one valid initiative, actual calendar `cal`, President without foresight of dissolutions;
    # pmf = collection-time distribution conditional on success
    gap = p['DEC'] + p['GMIN'] + (7 if option == 'O8' else 0) + TA_post
    t = advance(cal, filing, p['FORM'])
    tau = int(np.searchsorted(np.cumsum(pmf), rng.random()))
    t = advance(cal, t, tau)
    if t % WY > W_DEP_LAST:
        t = (t // WY + 1) * WY
    dep = t
    cure = rng.random() < p['p_cure']
    t = advance(cal, t, p['KQZ'] + (p['CURE'] if cure else 0) + p['CC'])
    c = t
    opt = option if (p['cutoff'] or option == 'O9') else 'NOCUT'
    for _ in range(10):
        if option == 'O4':      # required co-scheduling: the first actual polling day applies by law
            tp = schedule(cal, c, opt, gap, p['president'])
            return ('POLLED', tp - filing, tp - dep)
        pc = projected_cycles(cycles, c)
        key = tuple((x['s'], x.get('d')) for x in pc)
        if key not in cache:
            cache[key] = Calendar(pc, option, cal.t0, cal.t1)
        tp = schedule(cache[key], c, opt, gap, p['president'])
        if cal.feasible(tp) and not cal.is_susp(tp):
            return ('POLLED', tp - filing, tp - dep)
        # an unforeseen dissolution makes the decreed day unusable: re-decree when the suspension ends
        c = cal.susp_end(tp) if cal.is_susp(tp) else tp + 1
    return ('POLLED', tp - filing, tp - dep)


def mc_nf(p, option='O0', lam=0.0, n=20000, seed=1, calendar='current', TA_post=0):
    rng = np.random.default_rng(seed)
    out = []
    t0, t1 = -WY * 60, WY * 80
    pmf, fail = col_pmf(p); pmf = pmf / pmf.sum()
    for k in range(n):
        if k % 20 == 0:          # a fresh electoral calendar every 20 initiatives
            cyc = gen_timeline_current(lam, rng) if calendar == 'current' else gen_timeline(lam, rng)
            cal = Calendar(cyc, option, t0, t1); cache = {}
        filing = int(rng.integers(0, CYC)) if calendar == 'current' else int(rng.integers(0, 4 * WY * 3))
        out.append(simulate_one_nf(p, cal, cyc, filing, option, rng, cache, pmf, TA_post))
    return out


def whole_year_losses(lam, years=500000, seed=21):
    # calendar years (TS0) with no usable polling week, per 10,000 years, constant hazard (A20)
    rng = np.random.default_rng(seed)
    lam_w = lam / WY
    n = years * WY
    feas = np.ones(n + 4 * CYC, bool)
    s = W_FS
    while s < n:
        u = rng.exponential(1 / lam_w) if lam_w > 0 else np.inf
        if u < EL_PH:
            d = s + int(u); el = d + 6; s_next = el + 4
            feas[max(d, 0):s_next + 13] = False; feas[el] = False
        else:
            el = s + EL_PH; s_next = s + CYC
            feas[max(s_next - 26, 0):s_next + 13] = False; feas[el] = False
        s = s_next
    y = LOC_ABS
    while y < n:
        feas[y] = False; y += 4 * WY
    yrs = feas[:n].reshape(years, WY)
    lost = (~yrs.any(axis=1))[1:]
    return float(lost.sum() / (years - 1) * 10000), int(lost.sum())
