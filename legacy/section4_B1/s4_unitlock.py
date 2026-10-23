"""IV.1(d): unit baselines and the unit equilibrium lock (T7, T8) on illustrative unit turnouts."""
import numpy as np
from s4_core import *
from s4_rules import make_rule, solve_eq
from s4_units import units, qark_table
EN = sum(u['E'] for u in units); RN = sum(u['R'] for u in units); RHO_N = EN/RN

def unit_regime(rho_u, t0u, tmin):
    Q = rho_u/3
    if t0u <= tmin + 1e-6: return 'floor>=baseline'
    if Q > 1.0: return 'infeasible'
    if Q >= 0.5:
        g0 = campaign_gain(Q, t0u, tmin, uniform_prior(0.0))
        if g0 <= 0: return 'locked-all-pi'
        ps = optimize.brentq(lambda p: campaign_gain(Q, t0u, tmin, uniform_prior(p)), 0.0, 0.999, xtol=1e-6) if campaign_gain(Q,t0u,tmin,uniform_prior(.999))<=0 else 1.0
        return ('contested', ps)
    # Remark 1 fails: compute the equilibrium numerically (pi = 0)
    ch = make_rule('V3', rho_u)
    e = solve_eq(ch, t0u, tmin, uniform_prior(0.0), n=601)
    return ('Q<1/2', round(e['sx'],3), round(e['sd'],3), e['pure'])

if __name__ == '__main__':
    print('national proxy rho', round(RHO_N, 4))
    for scen in ('S1 equal resident turnout', 'S2 equal registered turnout'):
        for tau in (0.30, 0.40):
            t0n = RHO_N*tau
            for tm in (0.0, 0.2, 0.35):
                counts = {}; pis = []; low = []
                for u in units:
                    t0u = t0n if scen.startswith('S1') else u['rho']*tau
                    t0u = min(t0u, 0.999)
                    r = unit_regime(u['rho'], t0u, tm)
                    key = r if isinstance(r, str) else r[0]
                    counts[key] = counts.get(key, 0) + 1
                    if key == 'contested': pis.append(r[1])
                    if key == 'Q<1/2': low.append((u['name'], r[1:]))
                print(scen, 'tau', tau, 'tmin', tm, counts, 'pi* range', (round(min(pis),3), round(max(pis),3)) if pis else None)
                if low: print('   Q<1/2 units:', low)
    print('\nqarks (S1, tau .40, tmin .2):')
    for q in qark_table():
        r = unit_regime(q['rho'], RHO_N*0.40, 0.2); print(q['qark'], round(q['rho'],3), r)
