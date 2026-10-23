"""Consistency check 2 (Sections I-V): recomputation of every closed-form number.
Run: python3 cc2_recompute.py > cc2_recompute_output.txt
Blocks R1-R9 match the Appendix of the report. Requires numpy and scipy."""
import math
import numpy as np
from datetime import date, timedelta
from scipy import stats, optimize, integrate, special

def h(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)

# ---------------------------------------------------------------- R1 register
h("R1  Register, denominators, F7, zeta (Sections I-II)")
E_full, E_ocv = 3_713_761, 245_935
E_dom = E_full - E_ocv
R0 = 1.92e6
T_dom, T_all = 1_458_251, 1_664_873
print("E_dom = E_full - E_ocv =", E_dom)
rho_full, rho_dom = E_full / R0, E_dom / R0
print(f"rho_full = {rho_full:.5f}, rho_dom = {rho_dom:.5f}")
for R in (1.89e6, 1.92e6, 1.95e6):
    rf, rd = E_full / R, E_dom / R
    print(f"R={R/1e6:.2f}M  rho_full={rf:.4f} rho_dom={rd:.4f}  qA*rho={rd/3:.4f}..{rf/3:.4f}"
          f"  qR*rho={rd/2:.4f}..{rf/2:.4f}  A=E-R: {(E_dom-R)/1e6:.3f}..{(E_full-R)/1e6:.3f}M")
print(f"F7 factor E/(2 T_dom): full {E_full/(2*T_dom):.4f}, dom {E_dom/(2*T_dom):.4f}")
print(f"tau_R = T_dom/R = {T_dom/R0:.4f} (R 1.89-1.95M: {T_dom/1.95e6:.4f}-{T_dom/1.89e6:.4f}); T_all/R = {T_all/R0:.4f}")
print("ceil(E_full/3) =", math.ceil(E_full / 3), " E_dom/3 =", E_dom / 3, "-> ceil", math.ceil(E_dom / 3))
print("ceil(0.4 E_full) =", math.ceil(0.4 * E_full))
print(f"s = S/E_full = {50000/E_full:.5f};  S/R = {50000/R0:.4f}")
print(f"OCV share of A (E_full - R) = {E_ocv/(E_full-R0):.4f}")
print(f"T_dom/E_dom = {T_dom/E_dom:.5f};  T_all/E_full = {T_all/E_full:.5f}")
print("Out-of-country ballots T_all - T_dom =", T_all - T_dom,
      "; postal returned 191,000-194,889 -> unexplained", T_all - T_dom - 194_889, "to", T_all - T_dom - 191_000)
print(f"Valid votes at 96.55% of T_all = {T_all*0.9655:,.0f} (IDEA lists 1,607,347)")
print(f"OCV return rate {191000/E_ocv:.4f} - {194889/E_ocv:.4f}")
print("zeta = (E/2)/T = 1/(2*turnout):")
for yr, t in [(2001, .536), (2005, 1_397_519 / 2_848_144), (2009, .508), (2013, .535), (2017, .468), (2021, .463), (2025, T_all / E_full)]:
    print(f"  {yr}: turnout {t:.4f} zeta {1/(2*t):.3f}")
tauab_term = (T_all - T_dom) / R0
print(f"III-2 Table III.8: tau_ab(rho-1) = (T_all-T_dom)/R = {tauab_term:.4f}; Q_A with OCV (rho=1.93) = {1.93/3 - 0.5*tauab_term:.4f}")

h("R1b  Resident adults: R_2023 -> R_2025 versus d_R")
P_cen, R2023 = 2_402_113, 1.93e6
res25, res26 = 2_363_314, 2_335_930
lamR = R2023 / P_cen
dR_total = 1 - res26 / res25
print(f"lambda_R = {lamR:.5f}; d_R (total residents 2025->2026) = {dR_total:.5f}")
print(f"R_2025 if adult share constant = lambda_R * residents(1 Jan 2025) = {lamR*res25/1e6:.4f}M (Section II uses 1.92M)")
for yrs in (1.25, 1.5):
    print(f"  implied adult decline 1.93M -> 1.92M over {yrs} yr = {math.log(1.93/1.92)/yrs:.4f}/yr")
print(f"Adult share needed for 1.92M on 1 Jan 2025: {1.92e6/res25:.4f} (census {lamR:.4f})")
for dR in (0.0116, 0.004):
    print(f"  gamma_rho = gamma_E/E + d_R = {31200/E_full + dR:.4f} (d_R = {dR})")

# ---------------------------------------------------------------- R2 Table C.4
h("R2  Table C.4 municipal proxies")
rows = [
("Tiranë","Tiranë",598176,757361,647195,480610,20000),("Durrës","Durrës",153614,299989,256353,123423,20000),
("Elbasan","Elbasan",115101,205892,175943,92479,17595),("Shkodër","Shkodër",102434,200889,171668,82302,17167),
("Fier","Fier",101963,196324,167767,81923,16777),("Vlorë","Vlorë",83683,194147,165906,67236,16591),
("Korçë","Korçë",60754,129065,110291,48813,11030),("Lushnjë","Fier",63135,127438,108901,50726,10891),
("Kamëz","Tiranë",96137,125632,107358,77242,10736),("Lezhë","Lezhë",51354,106245,90791,41261,9080),
("Berat","Berat",62232,98875,84493,50001,8450),("Pogradec","Korçë",46070,90976,77743,37015,7775),
("Krujë","Durrës",51191,79608,68028,41130,6803),("Kavajë","Tiranë",30012,79556,67984,24113,6799),
("Dibër","Dibër",50775,78940,67457,40796,6746),("Kurbin","Lezhë",34405,72987,62370,27643,6237),
("Maliq","Korçë",31008,64664,55258,24914,5526),("Kukës","Kukës",36125,59393,50754,29025,5076),
("Kuçovë","Berat",31077,55293,47250,24969,4725),("Malësi e Madhe","Shkodër",21684,53918,46075,17422,4608),
("Divjakë","Fier",24882,53372,45609,19992,4561),("Gjirokastër","Gjirokastër",23270,52054,44482,18696,4449),
("Sarandë","Vlorë",22613,50680,43308,18169,4331),("Vau i Dejës","Shkodër",19261,48966,41843,15475,4185),
("Cërrik","Elbasan",25163,46652,39866,20217,3987),("Librazhd","Elbasan",23312,44181,37754,18730,3776),
("Shijak","Durrës",22058,44103,37688,17723,3769),("Mallakastër","Fier",15838,43033,36773,12725,3678),
("Patos","Fier",18227,42738,36521,14645,3653),("Devoll","Korçë",25897,42388,36222,20807,3623),
("Rrogozhinë","Tiranë",12567,40684,34766,10097,3477),("Bulqizë","Dibër",26826,40223,34372,21554,3438),
("Dimal","Berat",28135,39551,33798,22605,3380),("Finiq","Vlorë",11413,39055,33374,9170,3338),
("Peqin","Elbasan",16580,38900,33242,13321,3325),("Mat","Dibër",17405,38615,32998,13984,3300),
("Mirditë","Lezhë",13625,37384,31946,10947,3195),("Gramsh","Elbasan",16533,36305,31024,13284,3103),
("Vorë","Tiranë",21621,36230,30960,17372,3096),("Selenicë","Vlorë",9580,35793,30587,7697,3059),
("Prrenjas","Elbasan",18768,33031,28226,15079,2823),("Belsh","Elbasan",17123,32389,27678,13758,2768),
("Roskovec","Fier",16332,31548,26959,13122,2696),("Tropojë","Kukës",14189,28216,24112,11400,2412),
("Himarë","Vlorë",8328,27049,23114,6691,2312),("Dropull","Gjirokastër",8259,23131,19766,6636,1977),
("Memaliaj","Gjirokastër",6578,21467,18344,5285,1835),("Has","Kukës",11684,21247,18156,9388,1816),
("Përmet","Gjirokastër",7980,20301,17348,6412,1735),("Kolonjë","Korçë",7519,19919,17022,6041,1703),
("Skrapar","Berat",10750,19072,16298,8637,1630),("Poliçan","Berat",8762,18341,15673,7040,1568),
("Delvinë","Vlorë",6166,18078,15448,4954,1545),("Konispol","Vlorë",4898,17737,15157,3935,1516),
("Pukë","Shkodër",6222,16836,14387,4999,1439),("Klos","Dibër",12172,16618,14201,9780,1421),
("Tepelenë","Gjirokastër",6761,16371,13990,5432,1399),("Këlcyrë","Gjirokastër",4400,12468,10654,3535,1066),
("Fushë-Arrëz","Shkodër",4878,11651,9956,3919,996),("Libohovë","Gjirokastër",2765,7158,6117,2222,612),
("Pustec","Korçë",1843,5191,4436,1481,444)]
names = [r[0] for r in rows]; qark = [r[1] for r in rows]
C = np.array([r[2] for r in rows], float); CR = np.array([r[3] for r in rows], float)
Eh = np.array([r[4] for r in rows], float); Rh = np.array([r[5] for r in rows], float)
minsig = np.array([r[6] for r in rows])
print("n =", len(rows), " sum C_u =", int(C.sum()), " (P_cen =", P_cen, ")  sum CR_u =", int(CR.sum()))
lamE = E_full / CR.sum()
print(f"lambda_E = E_full/sum CR = {lamE:.5f}")
print("max |E_hat - lamE*CR| =", round(np.max(np.abs(Eh - lamE * CR)), 1),
      " max |R_hat - lamR*C| =", round(np.max(np.abs(Rh - lamR * C)), 1))
print("sum E_hat =", int(Eh.sum()), " sum R_hat =", int(Rh.sum()), f" national proxy rho = {Eh.sum()/Rh.sum():.4f}")
rho_u = Eh / Rh
branch20k = Eh >= 200000
ms = np.where(branch20k, 20000, np.ceil(0.1 * Eh)).astype(int)
print("min-signature rule reproduced in all rows:", bool(np.all(ms == minsig)),
      "; rows of the 10% branch one higher than ordinary rounding:", int(np.sum((np.ceil(0.1*Eh) != np.round(0.1*Eh)) & ~branch20k)))
print("rows in 20,000 branch:", [names[i] for i in np.where(branch20k)[0]], f"; Elbasan / 200,000 = {Eh[2]/2e5:.3f}")
print(f"Tiranë share of E_full: {Eh[0]/E_full:.4f}")
print(f"median rho_u = {np.median(rho_u):.4f}; counts rho_u>2: {np.sum(rho_u>2)}, >2.5: {np.sum(rho_u>2.5)}, >3: {np.sum(rho_u>3)}, >=2.55: {np.sum(rho_u>=2.55)}, <1.5: {np.sum(rho_u<1.5)}")
print("rho_u > 3:", [(names[i], round(float(rho_u[i]), 3)) for i in np.argsort(-rho_u) if rho_u[i] > 3])
print("rho_u < 1.5:", [(names[i], round(float(rho_u[i]), 4)) for i in np.argsort(rho_u) if rho_u[i] < 1.5])
loc = minsig / Rh
print(f"local S_u/R_hat range {loc.min():.4f}-{loc.max():.4f}, median {np.median(loc):.4f}")
print(f"feasibility lock at median: 2(1-rho/3) = {2*(1-np.median(rho_u)/3):.4f}")
qk = sorted(set(qark)); print("qarks:", len(qk))
shares = []
for q in qk:
    idx = [i for i in range(len(rows)) if qark[i] == q]
    sh = C[idx].max() / C[idx].sum(); shares.append(sh)
    rq = Eh[idx].sum() / Rh[idx].sum()
    print(f"  {q:12s} largest-municipality census share {sh:.3f}  rho_q {rq:.4f}  Q_q {rq/3:.4f}")
print(f"largest-municipality share range {min(shares):.3f}-{max(shares):.3f}")
print("Drift at 2%/yr (clock starts at the table): count q_A*rho_u > 1")
for t in (0, 2, 5, 10, 15, 20):
    print(f"  t={t:2d}: {int(np.sum(rho_u*math.exp(0.02*t) > 3))}")
print("joiners (years to rho_u=3):", [(names[i], round(math.log(3/rho_u[i])/0.02, 2)) for i in np.argsort(-rho_u) if 2.85 < rho_u[i] <= 3])
print("Remark 1 exits (years to rho_u=1.5):", [(names[i], round(math.log(1.5/rho_u[i])/0.02, 2)) for i in np.argsort(rho_u) if rho_u[i] < 1.5])
reb = 1.93 / 1.92
print(f"Rebased to R_2025=1.92M (x{reb:.5f}): count >3 = {int(np.sum(rho_u*reb>3))}, count <1.5 = {int(np.sum(rho_u*reb<1.5))}; Pustec {rho_u[-1]*reb:.4f}, Dimal {rho_u[names.index('Dimal')]*reb:.4f}")

# ---------------------------------------------------------------- R3 Section III
h("R3  Section III closed forms (Regime A, V6d, Regime B)")
def best_challenger(Q, t0, tmin, tbar=1.0):
    """max over s in (tmin, tbar] of U(s) = 1 - muhat(s) - (s-tmin)^2/(8 t0 (t0-tmin)) (pi=0 units)."""
    def muhat(s):
        return max(tmin / (s + tmin) if s + tmin > 0 else 0.0, Q / s)
    def U(s):
        m = muhat(s)
        return (max(0.0, 1 - m)) - (s - tmin) ** 2 / (8 * t0 * (t0 - tmin))
    grid = np.linspace(max(tmin, 1e-6) + 1e-9, tbar, 4001)
    vals = np.array([U(s) for s in grid]); i = int(np.argmax(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = optimize.minimize_scalar(lambda s: -U(s), bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
    s = r.x if -r.fun >= vals[i] else grid[i]
    return s, U(s), muhat(s)

for rho in (1.93, rho_full, 1.81, rho_dom):
    Q = rho / 3
    out = []
    for tmin in (0, 0.2, 0.35):
        t0 = rho * 0.40; s, U, m = best_challenger(Q, t0, tmin)
        out.append(f"tmin={tmin}: s*={s:.3f} pi*={2*U:.4f}")
    print(f"rho={rho:.4f} tau=0.40: " + "; ".join(out))
print("Table III.2 Pr[change] at pi=0 (campaign iff pi*>=0):")
for rho in (1.93, 1.81):
    for tmin in (0, 0.2, 0.35):
        vals = []
        for tau in (0.2, 0.3, 0.4, 0.5):
            t0 = rho * tau; s, U, m = best_challenger(rho / 3, t0, tmin)
            vals.append(round(float(1 - m), 3) if U >= 0 else 0.0)
        print(f"  rho={rho} tmin={tmin}: {vals}")
print("Lock boundary in tau (pi*=0):")
for rho in (1.93, rho_full, 1.81, rho_dom, 2.14):
    res = []
    for tmin in (0, 0.2, 0.35):
        f = lambda tau: best_challenger(rho / 3, rho * tau, tmin)[1]
        tl = optimize.brentq(f, 0.2, 0.45, xtol=1e-7)
        s, U, m = best_challenger(rho / 3, rho * tl, tmin)
        res.append(f"tmin={tmin}: {tl:.4f}")  # at the boundary the best response collapses to the floor; see R7 for mu_hat just above
    print(f"  rho={rho:.4f}: " + "; ".join(res))
for pi in (0.0, 0.3):
    print(f"Prop 18 tau_lock({pi}) = {(1/3)/(2*((2-pi)/3)**1.5):.4f}; domain 1.5 <= rho <= {2-pi}")
for rho in (2.1, 2.14, 2.2):
    print(f"  capped branch t_min=0, pi=0: tau = 1/(rho*sqrt(8(1-rho/3))) at rho={rho}: {1/(rho*math.sqrt(8*(1-rho/3))):.4f}")
print("(3.11) rho > (7+tmin)/4 -> max t_min = 4 rho - 7:")
for rho in (1.81, rho_dom, 1.93, rho_full):
    print(f"  rho={rho:.4f}: t_min < {4*rho-7:.4f}")
for tmin in (0, 0.2, 0.35): print(f"  rho threshold at t_min={tmin}: {(7+tmin)/4:.4f}")
def p19_gap(rho, tmin): return (1 - rho / 2) - (1 - tmin) ** 2 / (2 * rho * (rho - 2 * tmin))
print("Prop 19 t_min* (equality at tau=0.50):")
for rho in (1.78, rho_dom, 1.81, 1.84, 1.90, rho_full, 1.97):
    grid = np.linspace(0, rho / 2 - 1e-3, 20001); g = np.array([p19_gap(rho, t) for t in grid])
    k = int(np.argmax(g >= 0))  # first sign change: the opposition campaigns iff t_min > t*
    ts = optimize.brentq(lambda t: p19_gap(rho, t), grid[k - 1], grid[k])
    print(f"  rho={rho:.4f}: t_min* = {ts:.4f}")
print("rho root at t_min=0.35:", round(optimize.brentq(lambda r: p19_gap(r, 0.35), 1.7, 1.9), 5))
print("(3.11') left minus right at tau=0.50:")
for rho in (1.81, rho_dom, 1.93, rho_full):
    print(f"  rho={rho:.4f}: " + ", ".join(f"{p19_gap(rho,t):+.4f}" for t in (0, 0.2, 0.35)))
print("tau threshold at t_min=0.35 (capped, opposition campaigns iff tau >):")
for rho in (1.78, rho_dom, 1.81, 1.84):
    c = 0.65 ** 2 / (8 * (1 - rho / 2)); t0 = (0.35 + math.sqrt(0.35 ** 2 + 4 * c)) / 2
    print(f"  rho={rho:.4f}: t0={t0:.4f} tau={t0/rho:.4f} {'(infeasible: t0>1)' if t0>1 else ''}")
print("Capped mu_hat_R = rho/2:", [round(r/2, 4) for r in (1.81, rho_dom, 1.93, rho_full)])
print("Feasibility locks 2(1-Q_R):", [round(2*(1-r/2), 4) for r in (1.81, rho_dom, 1.93, rho_full)])
print("Prop 3(ii) cap tau (t_min=0): tau >= 1/(2 rho sqrt(rho/2)):", [round(1/(2*r*math.sqrt(r/2)), 4) for r in (1.81, 1.93)])
print("mu_hat_R uncapped at tau=0.20, E_dom (rho=1.81): ", round((1.81/2/(0.362)/2)**(2/3), 4))
lam_star = lambda e: (e + math.sqrt(e * e + 8 * (1 - e))) / 4
print("Lambda*(eps):", [round(lam_star(e), 4) for e in (0, .2, .4)])
print("F29 V6d bias point t0: ", [round((tm + math.sqrt(tm*tm + 4*(1/3)*(2/3 - tm)))/2, 4) for tm in (0, .2, .35)])
def Vfun(a):
    r = optimize.minimize_scalar(lambda d: -(d / (a + d) - d * d / 8), bounds=(0, 10), method="bounded")
    return -r.fun
L2 = optimize.brentq(lambda L: L ** (2/3) * 4 ** (-1/3) - Vfun((4 * L) ** (1/3)), 0.3, 0.5)
print(f"Lambda_2(0) = {L2:.6f}; cap threshold t0 <= 1/(4 Lambda_2)^(1/3) = {1/(4*L2)**(1/3):.4f}")
print("pi_7 = (1-qS)(2-qS):", [round((1-q)*(2-q), 4) for q in (0.55, 0.6, 2/3)])
for Om in (2, 0.5):
    r = math.sqrt(Om); print(f"V1 existence bound at Omega={Om}: {min(r*(1+2*r), 2+r)/(1+r)**2:.4f}")
print("Corollary 1 slack rho bound 1.125 + 0.375 t_min:", [1.125 + 0.375*t for t in (0, .2, .35)])
print("Q_A/t_min min (rho=1.81, t_min=0.35):", round(1.81/3/0.35, 3))

# ---------------------------------------------------------------- R4 Section IV
h("R4  Section IV: noise, bounds, verification, drift")
z9 = stats.norm.ppf(0.9)
for rho in (1.93, 1.81, rho_full):
    Q = rho / 3; ss = rho * 0.04
    pas = integrate.quad(lambda m: stats.norm.cdf((1 - Q / m) / ss), Q, 1)[0] / 0.5
    print(f"rho={rho:.4f}: sigma_s={ss:.4f} mu_0.9={Q/(1 - z9*ss):.4f} Pr[pass|mu>1/2]={pas:.4f} noiseless={(1-Q)/0.5:.4f}")
Q = rho_full / 3; ss = rho_full * 0.04
print("Lemma 12 illustration:", [round(float(stats.norm.cdf((1 - Q / m) / ss)), 3) for m in (0.65, 0.70, 0.75)])
print(f"cap truncation mean shortfall E[Z-] = {stats.norm.pdf(0):.4f}")
for sig in (0.077, 0.039, 0.116):
    print(f"rho_feas (sigma_s fixed {sig}) = {3*(1 - z9*sig):.4f}")
print(f"rho_feas (sigma_s = 0.04 rho) = {3/(1 + 3*z9*0.04):.4f}")
print(f"margins for 0.90: normal {z9:.4f}, VP {math.sqrt(4/(9*0.1)-1):.4f}, Cantelli {math.sqrt(1/0.1-1):.4f}, Chebyshev {math.sqrt(10):.4f}")
for v in (0.6, 0.7, 0.85):
    print(f"n_min(v_min={v}) = {math.ceil(math.log(100)/(2*0.05**2*v**2))}; bound (4.19) at n=1500: {math.exp(-2*1500*0.0025*v*v):.4f}")
print(f"Upsilon_seq = {math.sqrt(6000*math.log(100)/2):.2f}")
S = 50000
def r3_worst(v, n=1500, a=0.05):
    J = round(S / v)
    FR = stats.hypergeom.cdf(math.ceil((1 - a) * S * n / J) - 1, J, S, n)
    FA = stats.hypergeom.sf(math.ceil((1 + a) * S * n / J) - 1, J, S - 1, n)
    return FR, FA
print("R3 (band ±5%, n=1500) worst cases at the threshold:")
for v in (0.6, 0.7, 0.8, 0.9, 0.95):
    FR, FA = r3_worst(v); print(f"  v_L={v}: FR(V_L=S)={FR:.4f} FA(V_L=S-1)={FA:.2e}")
def r0_accept(J, V, Sx):
    n = math.ceil(0.05 * J); thr = math.ceil(n * Sx / J)
    return stats.hypergeom.sf(thr - 1, J, V, n), n
print("R0 (5% projection) national at the threshold:")
for v in (0.6, 0.7, 0.85, 0.95):
    J = round(S / v); a, n = r0_accept(J, S, S); print(f"  v_L={v}: n={n} FR(V_L=S)={1-a:.3f}")
print("R0 Pustec (S_u=444): FR at V_L=S_u; FA at -10%; FR at +10%; FR at +2%")
for v in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
    Su = 444
    J0 = round(Su / v); a0, n0 = r0_accept(J0, Su, Su)
    Vm = round(Su * 0.9); Jm = round(Su * 0.9 / v); am, _ = r0_accept(Jm, Vm, Su)
    Vp = round(Su * 1.1); Jp = round(Su * 1.1 / v); ap, _ = r0_accept(Jp, Vp, Su)
    V2 = round(Su * 1.02); J2 = round(Su * 1.02 / v); a2, _ = r0_accept(J2, V2, Su)
    print(f"  v_L={v}: J={J0} n={n0}  FR@S={1-a0:.3f}  FA@-10%={am:.3f}  FR@+10%={1-ap:.3f}  FR@+2%={1-a2:.3f}")
def rho_lock_capped(t0, tmin=0.2, pi=0.0): return 3 * (1 - pi / 2 - (1 - tmin) ** 2 / (8 * t0 * (t0 - tmin)))
def branch_ok(t0, tmin=0.2): return rho_lock_capped(t0, tmin) / 3 >= (1 - tmin) / (4 * t0 * (t0 - tmin))
def rho_lock_num(t0, tmin=0.2):
    return optimize.brentq(lambda r: best_challenger(r / 3, t0, tmin)[1], 1.5, 3.0, xtol=1e-8)
print("rho_lock(t0), pi=0, t_min=0.2:")
locks = {}
for t0 in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90):
    rc = rho_lock_capped(t0); ok = branch_ok(t0); rn = rho_lock_num(t0); locks[t0] = rn
    print(f"  t0={t0}: (4.24)={rc:.4f} branch_ok={ok} numeric={rn:.4f}  t_hit(const 0.020 from 1.934)={math.log(rn/1.934)/0.02:6.2f} yr")

# ---------------------------------------------------------------- R5 Section V Part 1
h("R5  Section V Part 1: cut-off, T4, collection, learning, dates")
def dcut(wk, g0, latest=False):
    if latest: return 51 - wk if wk <= 10 else 103 - wk
    return g0 if wk <= 10 else max(52 - wk, g0)
for g0 in (11, 5):
    v = [dcut(w, g0) for w in range(52)]
    print(f"g0={g0}: mean {np.mean(v):.2f} wk, jump {dcut(11,g0)-dcut(10,g0)}")
print(f"latest: mean {np.mean([dcut(w,0,True) for w in range(52)]):.2f} wk")
def t4_share(d_rev, g_min):
    tot = d_rev + g_min
    return 1.0 if tot <= 54 else (0.0 if tot > 60 else (61 - tot) / 7)
for g in (30, 42):
    print(f"T4 g_min={g}: " + ", ".join(f"d_rev {d}:{t4_share(d,g):.2f}" for d in (12, 13, 18, 19, 24, 25, 28, 30, 31)))
for dd in (0, 5, 10):
    print(f"phi_B at delta={dd} days: {(9 + dd/30.4375)/48:.4f}")
print(f"lambda_ee: 2/35.53={2/35.53:.4f}; -ln0.05/27.87={-math.log(0.05)/27.87:.4f}")
mcol = 31 / 1.2; sc = 0.5
base = lambda W=31, phif=0.25: stats.norm.cdf((math.log(W) - math.log(mcol) - math.log((1+phif)/1.25)) / sc)
z = math.log(31/mcol)/sc
print(f"collection: Pr={base():.4f}; phi_f=0 {base(phif=0):.3f}; 0.5 {base(phif=0.5):.3f}; W=52 {base(52):.3f}; W=13 {base(13):.3f};"
      f" dPr/dlnW={stats.norm.pdf(z)/sc:.3f}; dPr/dphi={-stats.norm.pdf(z)/(sc*1.25):.3f}")
print(f"conditional median given success: {mcol*math.exp(sc*stats.norm.ppf(0.5*base())):.2f} wk")
rbar = 1 / (30 * math.log(1 / 0.95))
print(f"r_bar = {rbar:.4f}/day; 80% at L=30 iff k^beta <= {30*rbar*math.log(1/0.8):.3f}")
for beta in (0.5, 1.0):
    print(f"1994 L*(k=124, beta={beta}) = {124**beta/(rbar*math.log(1/0.8)):.1f} days; A_I(26)={math.exp(-124**beta/(rbar*26)):.4f}")
for (lab, Q_, muhi) in (("E_full", 1.93/3, 1.0), ("E_dom", 1.81/3, 1.0), ("E_full pi=0.3", 1.93/3, 0.85)):
    for beta in (0.5, 1, 2):
        k = 1
        while math.exp(-k ** beta / (rbar * 30)) * muhi >= Q_: k += 1
        print(f"  infeasible from k={k} ({lab}, beta={beta})")
Qf = 1.93 / 3
print("gamma_c (beta=1, E_full):", [round((Qf - math.exp(-k/(30*rbar)))/(1 - math.exp(-k/(30*rbar))), 3) for k in (9, 10, 15, 20, 30)])
print("capped threshold at L=30, k=1:", [round(Qf/(tm + (1-tm)*0.95), 4) for tm in (0, .2, .35)])
print(f"H_blk bound sqrt(5/30) = {math.sqrt(5/30):.3f}")
d = lambda a, b: (b - a).days
print("2013 decree->polling g:", d(date(2013,4,9), date(2013,12,22)), "days; ruling->polling:", d(date(2013,3,8), date(2013,12,22)))
f0 = date(2012, 3, 13)
for lab, dt in [("KQZ session", date(2012,6,28)), ("forward", date(2013,1,2)), ("Court", date(2013,3,8)), ("decree", date(2013,4,9)), ("Law 156", date(2013,10,10)), ("polling", date(2013,12,22))]:
    print(f"  C.6 {lab}: day {d(f0, dt)}")
print("1998 repeal -> vote:", d(date(1998,10,12), date(1998,11,22)), "days")
print("1994: 6 Nov 1994 is", date(1994,11,6).strftime("%A"), "; decree 11 Oct -> vote", d(date(1994,10,11), date(1994,11,6)), "; publication 6 Oct ->", d(date(1994,10,6), date(1994,11,6)))
print("2013-12-15 and 2013-12-22:", date(2013,12,15).strftime("%A"), date(2013,12,22).strftime("%A"))

# ---------------------------------------------------------------- R6 Section V Part 2
h("R6  Section V Part 2: bar, window, voter model, drift, void votes")
for q in (0.5, 1/math.e, 0.25, 0.10):
    print(f"Prop 25: c=3 exceeds fidelity horizon iff T_rel < {3/math.log(1/q):.3f} (q_chi={q:.3f})")
for L, t25 in [(0.38, 0.4027), (0.38, 0.403), (0.30, 0.4027), (0.35, 0.4027), (0.40, 0.4027)]:
    print(f"(5.34) L={L}, t25={t25}: T_post <= {5.98/math.log((0.76-L)/(t25-L)):.3f}")
print(f"t_25 = 0.1845 * (2.133..2.183) = {0.1845*2.133:.4f}..{0.1845*2.183:.4f}")
for Tp in (2.12, 6):
    r = [(0.76 - L) * math.exp(-5.98 / Tp) for L in (0.41, 0.38)]
    print(f"residual at 5.98 months, T_post={Tp}: {r[0]:.4f}-{r[1]:.4f}; net floor {0.393-r[1]:.3f}-{0.403-r[0]:.3f} (T_post exactly {Tp})")
print(f"Prop 28 thresholds: post {5.49/math.log(10):.3f}-{7.52/math.log(10):.3f}; pre {1.54/math.log(10):.3f}-{3.55/math.log(10):.3f}; anchor 4/ln10={4/math.log(10):.3f}")
print(f"blocked share of anchored window (4 + d_post)/48: {(4+5.49)/48:.4f}-{(4+7.52)/48:.4f}")
print(f"local premium left at T_post=2.12: 1 month {math.exp(-1/2.12):.3f}, 3 months {math.exp(-3/2.12):.3f}; bar 2.12 ln10 = {2.12*math.log(10):.2f} months")
for T in (1, 5): print(f"Lemma 16 r_v (A=2) for T_rel={T}: {R0/(2*T)/1e6:.3f} million")
fa = lambda a: 2 * special.gamma(a + 0.5) / (math.sqrt(math.pi) * special.gamma(a))
print("f_a(1/2):", [round(float(fa(a)), 3) for a in (0.5, 0.75, 1, 1.5, 2.5, 4, 8, 16)])
for b in (1.93/3, rho_full/3):
    print(f"S_a(b) at b={b:.4f}:", [round(float(1 - special.betainc(a, a, b)), 3) for a in (0.5, 0.75, 1, 1.5, 2.5, 4, 8, 16)])
print(f"a_B(0.3) matched to A2 = {((1/(4*(0.7**2/12)))-1)/2:.3f}; half variance at pi=0: {((1/(4*(1/24)))-1)/2:.3f}")
def ETb(a, b, x=0.5):
    A = 2 * a; B = special.beta(a, a)
    return integrate.quad(lambda y: A * y**(-a) * (1-y)**(-a) * B * special.betainc(a, a, y), x, b)[0]
def dETb(a, b):
    A = 2 * a; B = special.beta(a, a)
    return (1/3) * A * b**(-a) * (1-b)**(-a) * B * special.betainc(a, a, b)
b = rho_full / 3
for a in (1, 2.5):
    print(f"Prop 27 a_B={a}: E[T_b|1/2] = {ETb(a,b):.4f} T_rel; dE/drho = {dETb(a,b):.3f} T_rel per unit rho (b={b:.4f})")
gam = 0.020
for sg in (0.01, 0.02):
    print(f"Prop 29 Pr[rho_dom ever falls to 1.7803] (sigma={sg}, gamma=0.020): {math.exp(-2*gam*math.log(rho_dom/1.7803)/sg**2):.4f}")
def path_rho(t, E0, dR, gE=31200, R_0=R0): return (E0 + gE * t) / (R_0 * math.exp(-dR * t))
def t_cross(target, E0, dR):
    if path_rho(0, E0, dR) >= target: return 0.0
    return optimize.brentq(lambda t: path_rho(t, E0, dR) - target, 0, 200)
print("A29 deterministic path, E_full from April 2025 (t=0):")
for t0, rl in locks.items():
    print(f"  t0={t0}: rho_lock={rl:.4f}  d_R=0.0116 -> t={t_cross(rl, E_full, 0.0116):5.2f} yr ;  d_R=0.004 -> t={t_cross(rl, E_full, 0.004):5.2f} yr")
for tgt in (1.8375, 2.0):
    print(f"  rho_dom reaches {tgt}: d_R=0.0116 -> {t_cross(tgt, E_dom, 0.0116):.2f} yr; d_R=0.004 -> {t_cross(tgt, E_dom, 0.004):.2f} yr")
print("Remark 5: Pr[void with a Yes majority] (Regime A, mu_b=1/2, tau_ab=0)")
cells = []
for rho in (1.93, 1.81):
    for tmin in (0, 0.2, 0.35):
        for pi in (0, 0.3):
            for tau in (0.25, 0.30, 0.40, 0.50):
                t0 = rho * tau
                if t0 >= 1: continue
                s, U, m = best_challenger(rho / 3, t0, tmin)
                lo, hi = pi / 2, 1 - pi / 2
                if 2 * U >= pi:  # campaign
                    muV = tmin / (s + tmin)
                    pr = (min(m, hi) - max(muV, lo)) / (1 - pi)
                elif tmin > 0:
                    pr = (hi - max(0.5, lo)) / (1 - pi)
                else:
                    pr = None
                cells.append((rho, tmin, pi, tau, pr))
vals = [c[4] for c in cells if c[4] is not None]
print(f"  cells {len(cells)}, with ballots {len(vals)}, without {sum(c[4] is None for c in cells)}; range {min(vals):.3f}-{max(vals):.3f}")

# ---------------------------------------------------------------- R7 supplements
h("R7  Supplements: mu_hat just above the lock (Corollary 2), R0 worst case in small lists")
for rho in (1.93, 1.81):
    for tmin in (0.0, 0.2, 0.35):
        tl = optimize.brentq(lambda tau: best_challenger(rho / 3, rho * tau, tmin)[1], 0.2, 0.45, xtol=1e-9)
        s, U, m = best_challenger(rho / 3, rho * (tl + 1e-4), tmin)
        print(f"rho={rho} tmin={tmin}: lock tau={tl:.4f}; just above: s*={s:.3f} mu_hat={m:.3f}")
for Su, nm in ((444, "Pustec"), (612, "Libohovë"), (996, "Fushë-Arrëz")):
    worst = (0, None)
    for J in range(math.ceil(Su / 0.95), math.floor(Su / 0.6) + 1):
        a, n = r0_accept(J, Su, Su)
        if 1 - a > worst[0]: worst = (1 - a, (J, n, round(Su / J, 3)))
    print(f"R0 worst FR at V_L=S over v_L in [0.6,0.95], {nm} (S={Su}): {worst[0]:.3f} at (J, n, v_L) = {worst[1]}")
print("Remark 5 by denominator:")
for rho in (1.93, 1.81):
    v = [c[4] for c in cells if c[0] == rho and c[4] is not None]
    print(f"  rho={rho}: {min(v):.3f}-{max(v):.3f} over {len(v)} cells")
print("R0 worst FA at V_L=S-1 (list one line short), same scan:")
for Su, nm in ((444, "Pustec"), (612, "Libohovë"), (996, "Fushë-Arrëz")):
    worst = (0, None)
    for J in range(math.ceil(Su / 0.95), math.floor(Su / 0.6) + 1):
        a, n = r0_accept(J, Su - 1, Su)
        if a > worst[0]: worst = (a, (J, n, round(Su / J, 3)))
    print(f"  {nm}: {worst[0]:.3f} at (J, n, v_L) = {worst[1]}")

# ---------------------------------------------------------------- R8 adults (INSTAT 1 Jan 2026 release)
h("R8  Adult decline from INSTAT dependency ratios (Population on 1 January 2026)")
# youth and old dependency ratios as printed (per cent of the 15-64 population)
dep = {2025: (23.8, 33.3), 2026: (23.7, 35.7)}
pop = {2025: res25, 2026: res26}
p15 = {}
for yr, (yd, od) in dep.items():
    tot = 1 + (yd + od) / 100
    sh0_14 = (yd / 100) / tot
    p15[yr] = pop[yr] * (1 - sh0_14)
    print(f"{yr}: share 0-14 {sh0_14:.4f}; population 15+ {p15[yr]:,.0f}")
dec15 = 1 - p15[2026] / p15[2025]
print(f"decline of the 15+ population 2025->2026: {dec15:.4f} (rounding of the ratios to 0.1 pp moves it by about +/-0.001)")
for ded in (90_000, 110_000):
    r25, r26 = p15[2025] - ded, p15[2026] - ded
    print(f"  18+ with a 15-17 deduction of {ded:,}: R(1 Jan 2025) {r25/1e6:.4f}M, R(1 Jan 2026) {r26/1e6:.4f}M, decline {1-r26/r25:.4f}")
dR_mid = 0.0085
print(f"gamma_rho at d_R={dR_mid}: {31200/E_full + dR_mid:.4f}")
print("A29 path with d_R = 0.0085:")
for t0, rl in locks.items():
    print(f"  t0={t0}: rho_lock={rl:.4f} -> t={t_cross(rl, E_full, dR_mid):5.2f} yr")
for tgt in (1.8375, 2.0):
    print(f"  rho_dom reaches {tgt}: {t_cross(tgt, E_dom, dR_mid):.2f} yr")
print(f"municipal joiners/exits at gamma=0.017: Pustec {math.log(3/rho_u[-1])/0.017:.2f}, Tiranë exit {math.log(1.5/rho_u[0])/0.017:.2f} yr")
for t in (2, 5): print(f"  count q_A*rho_u>1 at t={t}, gamma 0.017: {int(np.sum(rho_u*math.exp(0.017*t) > 3))}")

# ---------------------------------------------------------------- R9 Regime B (T5) under Beta(a, a) priors
h("R9  F28 under non-uniform priors: T5 opposition's campaign condition at the cap, (4.4) applied to V4")
# opposition (challenger to the adopted amendment) campaigns at the cap iff S(Q_R/tbar)/f(1/2) >= (1-tmin)^2/(8 t0 (t0-tmin)),
# kappa recalibrated at f(1/2) (Prop. 17 / Cor. 5 logic, with the reform side at its floor by Lemma 4(a)); pi = 0, tau_ab = 0
for lab, rho in (("E_dom", rho_dom), ("E_full", rho_full)):
    QR = rho / 2
    for a in (0.5, 0.75, 1.0, 1.5, 2.5):
        S_ = 1 - special.betainc(a, a, QR); f_ = fa(a); lhs = S_ / f_
        res = []
        for tmin in (0.0, 0.2, 0.35):
            rhs = lambda tau: (1 - tmin) ** 2 / (8 * rho * tau * (rho * tau - tmin))
            camp50 = lhs >= rhs(0.5)
            if camp50:
                tt = optimize.brentq(lambda tau: lhs - rhs(tau), tmin / rho + 1e-6, 0.5)
                res.append(f"tmin={tmin}: campaigns for tau > {tt:.3f}")
            else:
                res.append(f"tmin={tmin}: never within D3")
        print(f"{lab} a_B={a}: S(Q_R)={S_:.4f} f(1/2)={f_:.3f} S/f={lhs:.4f} -> " + "; ".join(res))
