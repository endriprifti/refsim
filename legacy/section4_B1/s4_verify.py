"""IV.5: signature verification rules -- error (FA/FR), expected workload (lines examined).
List with V_L valid lines of J; valid share v_L = V_L/J held fixed as the margin varies:
J = round(V_L/v_L).  Fixed-sample rules draw n_smp lines without replacement (exact hypergeometric);
R2 (SPRT) is simulated without replacement, R4 (Lemma 14) with replacement, checked after every draw.
[REVISED B1] worst cases are maxima over every integer V_L within a window of the threshold S
(the supremum sits at V_L = S - 1 for FA and V_L = S for FR), computed with tail functions."""
import math
import numpy as np
from scipy import stats
rng = np.random.default_rng(11)

def setup(S, Delta, v):
    V = int(round(S*(1+Delta))); J = int(round(S*(1+Delta)/v)); return V, J

def n_R0(J): return int(np.ceil(.05*J))
def n_R1(J): return max(500, int(np.ceil(.03*J)))

def band_error(S, V, J, n, lo, hi):
    """Exact error of a band rule (accept if Shat >= hi*S, reject if Shat < lo*S, else count every line).
    lo = hi = 1 gives a sample-only rule (R0). Returns (error, Pr[full count])."""
    n = min(n, J)
    if n >= J: return 0.0, 1.0
    hg = stats.hypergeom(J, V, n)
    k_acc = math.ceil(hi*S*n/J - 1e-9)            # accept iff k >= k_acc
    k_rej = math.ceil(lo*S*n/J - 1e-9) - 1        # reject iff k <= k_rej
    p_acc, p_rej = float(hg.sf(k_acc - 1)), float(hg.cdf(k_rej))
    return (p_rej if V >= S else p_acc), max(0.0, 1 - p_acc - p_rej)

RULES = {'R0': (n_R0, 1.0, 1.0), 'R1': (n_R1, .95, 1.10)}
def rule_exact(rule, S, Delta, v, n_smp=1500, a=.05, full_if=3000):
    """Error and expected lines for R0, R1 or R3 (band +-a, n_smp lines, every line if J <= full_if)."""
    V, J = setup(S, Delta, v)
    if rule == 'R3':
        if J <= full_if: return 0.0, J, 1.0
        nf, lo, hi = (lambda J: n_smp), 1 - a, 1 + a
    else: nf, lo, hi = RULES[rule]
    n = min(J, nf(J)); err, pfull = band_error(S, V, J, n, lo, hi)
    return err, n + pfull*(J - n), pfull

def worst_case(rule, S, v, W=600, **kw):
    """Max error over every integer V_L in [S-W, S+W] (v_L fixed)."""
    best = (0.0, 0)
    for V in range(S - W, S + W + 1):
        e = rule_exact(rule, S, V/S - 1, v, **kw)[0]
        if e > best[0]: best = (e, V - S)
    return best

def switch_error(S, V, v, n=1500, narrow=.02, wide=.05, thr=.92):
    """Statutory option, point 3: band +-narrow if the sampled authentic share >= thr, else +-wide."""
    J = int(round(V/v)); n = min(n, J)
    if n >= J: return 0.0
    hg = stats.hypergeom(J, V, n)
    kt = math.ceil(thr*n - 1e-9)
    ka_n, ka_w = math.ceil((1+narrow)*S*n/J - 1e-9), math.ceil((1+wide)*S*n/J - 1e-9)
    kr_n, kr_w = math.ceil((1-narrow)*S*n/J - 1e-9) - 1, math.ceil((1-wide)*S*n/J - 1e-9) - 1
    P = lambda a, b: 0.0 if b < a else float(hg.cdf(b) - hg.cdf(a - 1))
    if V < S: return P(max(kt, ka_n), n) + P(ka_w, kt - 1)
    return P(kt, kr_n) + P(0, min(kr_w, kt - 1))

def sprt(S, Delta, v, delta=.02, alpha=.01, beta=.01, nmax_frac=.5, reps=4000):
    """R2: Wald SPRT on per-line validity, without replacement; p0, p1 = S(1 -+ delta)/J;
    full count if undecided at nmax = nmax_frac*J (or p1 >= 1, or J <= 2000)."""
    V, J = setup(S, Delta, v)
    p0, p1 = S*(1-delta)/J, S*(1+delta)/J
    if p1 >= 1 or J <= 2000: return 0.0, J, 1.0
    A, B = np.log((1-beta)/alpha), np.log(beta/(1-alpha))
    a, b = np.log(p1/p0), np.log((1-p1)/(1-p0))
    nmax = int(nmax_frac*J)
    errs = works = fulls = 0
    pop = np.zeros(J, bool); pop[:V] = True
    for _ in range(reps):
        x = pop[rng.permutation(J)[:nmax]]
        llr = np.cumsum(np.where(x, a, b))
        hitA = np.argmax(llr >= A) if (llr >= A).any() else None
        hitB = np.argmax(llr <= B) if (llr <= B).any() else None
        if hitA is None and hitB is None:
            fulls += 1; works += J; continue
        if hitB is None or (hitA is not None and hitA < hitB):
            works += hitA+1; errs += (V < S)
        else:
            works += hitB+1; errs += (V >= S)
    return errs/reps, works/reps, fulls/reps

def seq_hoeffding(S, Delta, v, n_seq=6000, alpha=.01, reps=2000, full_if=3000):
    """R4, Lemma 14: draws with replacement, boundary checked after every draw [REVISED B1: previously every 100];
    accept if sum(1_i - S/J) >= Y, reject if <= -Y, Y = sqrt(n_seq ln(1/alpha)/2); undecided by n_seq -> full count.
    Lists of J <= full_if lines (in particular J <= n_seq) are counted in full."""
    V, J = setup(S, Delta, v)
    if J <= max(full_if, n_seq): return 0.0, J, 1.0
    p, c = V/J, S/J; Y = np.sqrt(n_seq*np.log(1/alpha)/2)
    errs = works = fulls = 0
    for _ in range(reps):
        dev = np.cumsum((rng.random(n_seq) < p) - c)
        up = np.flatnonzero(dev >= Y); dn = np.flatnonzero(dev <= -Y)
        iu = up[0] if len(up) else None; idn = dn[0] if len(dn) else None
        if iu is None and idn is None: fulls += 1; works += J; continue
        if idn is None or (iu is not None and iu < idn): works += iu + 1; errs += (V < S)
        else: works += idn + 1; errs += (V >= S)
    return errs/reps, works/reps, fulls/reps

def duplicates(J, dfrac, f):
    d = dfrac*J; n = f*J
    p_in = (n/J)*((n-1)/(J-1)); p_vs_full = 1 - (1-n/J)*(1-(n-1)/(J-1))
    lam1, lam2 = d*p_in, d*p_vs_full
    return dict(exp_found_within=lam1, P_any_within=1-np.exp(-lam1), CV_est_within=1/np.sqrt(lam1) if lam1 > 0 else np.inf,
                exp_found_vs_full=lam2, CV_est_vs_full=1/np.sqrt(lam2))

UNITS = [('National', 50000), ('Korce', 11030), ('Pustec', 444)]
if __name__ == '__main__' and 'base' in __import__('sys').argv:
    print('Table IV.10 (S = 50,000, v_L = .85): margin | R0 err/lines | R1 | R2 (sim) | R3 | R4 (sim, full-count prob) | R5')
    for D in (-.10, -.05, -.02, -.01, .01, .02, .05, .10):
        r0, r1, r3 = (rule_exact(r, 50000, D, .85) for r in ('R0', 'R1', 'R3'))
        sp, sq = sprt(50000, D, .85), seq_hoeffding(50000, D, .85)
        print(f'{D:+.2f}: {r0[0]:.3f}/{r0[1]:,.0f} | {r1[0]:.3f}/{r1[1]:,.0f} | {sp[0]:.3f}/{sp[1]:,.0f} | '
              f'{r3[0]:.3f}/{r3[1]:,.0f} | {sq[0]:.3f}/{sq[1]:,.0f} ({sq[2]:.2f}) | {setup(50000, D, .85)[1]:,}', flush=True)
    print('Korce (S = 11,030, v_L = .85), R4 at -5% / +5%:', [tuple(round(x, 3) for x in seq_hoeffding(11030, D, .85)) for D in (-.05, .05)])
    for v in (.70, .72, .75, .78, .80, .82, .85, .90):
        print('Pustec R0 false admission of a list 10% short, v_L', v, round(rule_exact('R0', 444, 400/444 - 1, v)[0], 3))

if __name__ == '__main__' and 'ext' in __import__('sys').argv:
    print('Table IV.11: worst case over integer margins (exact); entries: error @ V_L - S')
    for nm, S in UNITS:
        for rule, kw in (('R0', {}), ('R1', {}), ('R3', {}), ('R3', {'n_smp': 1000})):
            cells = []
            for v in (.6, .7, .85, .95):
                e, at = worst_case(rule, S, v, W=min(600, S//3), **kw)
                cells.append(f'{e:.4f}@{at:+d}')
            print(nm, rule, kw.get('n_smp', ''), ' '.join(cells), flush=True)
    print('Statutory option, point 3: sup over V_L near S of the error, by true authentic share (S = 50,000, n_smp = 1,500)')
    shares = [.60, .70, .80, .85, .87, .875, .88, .885, .89, .895, .90, .905, .91, .92, .93, .95, .99]
    for (nar, thr, n) in ((.02, .90, 1500), (.02, .92, 1500), (.025, .90, 1500), (.02, .90, 2000)):
        res = [(v, max(switch_error(50000, V, v, n, nar, .05, thr) for V in range(50000-300, 50301))) for v in shares]
        m = max(res, key=lambda t: t[1])
        print(f'+-{nar} if sampled share >= {thr}, n={n}: sup {m[1]:.4f} at {m[0]} |', ' '.join(f'{v}:{e:.4f}' for v, e in res), flush=True)
    for S in (11030, 3000):
        res = [(v, max(switch_error(S, V, v) for V in range(S - 150, S + 151))) for v in shares]
        m = max(res, key=lambda t: t[1]); print(f'switch at .92, S = {S}: sup {m[1]:.4f} at {m[0]}')
    for va in (.90, .92, .95):
        print('pure +-2% band at authentic share', va, round(max(switch_error(50000, V, va, 1500, .02, .02, 0) for V in range(49700, 50301)), 4))
    print('Hoeffding bound (4.19) at n_smp = 1,500, a_v = .05:', {v: round(float(np.exp(-2*1500*.05**2*v**2)), 4) for v in (.6, .7, .85)})
    print('n_min Hoeffding (4.20):', {v: int(np.ceil(np.log(100)/(2*.05**2*v**2))) for v in (.6, .7, .85)})
    print('Ybar_seq (4.21):', round(np.sqrt(6000*np.log(100)/2), 1))
    print('duplicates, J = 58,800, f_smp = .05:', {df: {k: round(float(x), 3) for k, x in duplicates(58800, df, .05).items()} for df in (.01, .03)})
    from s4_units import units
    Smin = [int(np.ceil(.1*u['E'])) if u['E'] < 200000 else 20000 for u in units]
    for v in (.7, .85): print('municipal lists with J <= 3000 at the minimum count, v', v, ':', sum(1 for s_ in Smin if s_/v <= 3000))
