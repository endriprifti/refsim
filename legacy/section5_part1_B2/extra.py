# B.40 extras: collection elasticities (5.6) and indirect-initiative delay (Remark 4).
import math, json
from scipy import stats
from pipeline import *
p=dict(BASE)
m_col = p['W0']/p['K0']; z = math.log(p['K0'])/p['sig_c']
out = dict(m_col=m_col, z=z, Pr=stats.norm.cdf(z),
           dPr_dlnW=stats.norm.pdf(z)/p['sig_c'],
           dPr_dphi=-stats.norm.pdf(z)/(p['sig_c']*(1+p['phi_f0'])))
# indirect initiative: Assembly period T_A = 13 weeks before the decree
for TA in (0, 13, 26):
    r = chain_results(dict(BASE, DEC=BASE['DEC']+TA), conditional_valid=True)
    out[f'indirect_TA{TA}'] = dict(E_months=r['ET_polled']*12/WY, le24=float(r['cum'][104]), le12=float(r['cum'][52]))
    r2 = chain_results(dict(BASE, DEC=BASE['DEC']+TA, cutoff=False), conditional_valid=True)
    out[f'indirect_TA{TA}_nocut'] = dict(E_months=r2['ET_polled']*12/WY, le24=float(r2['cum'][104]))
json.dump(out, open('extra.json','w'), indent=1); print(json.dumps(out, indent=1))
# decomposition of the mean time to vote of a valid initiative (TS0, earliest), V.1 lead
pmf, fail = col_pmf(BASE); Et = (np.arange(len(pmf)) * pmf).sum() / pmf.sum()
ET = chain_results(dict(BASE), conditional_valid=True)['ET_polled']
stat = BASE['FORM'] + BASE['KQZ'] + BASE['p_cure'] * BASE['CURE'] + BASE['CC']; gap = BASE['DEC'] + BASE['GMIN']
mo = lambda w: w * 12 / WY
out['decomposition_months'] = dict(collection=mo(Et), stages=mo(stat), decree_and_lead=mo(gap), calendar_wait=mo(ET - Et - stat - gap), total=mo(ET))
json.dump(out, open('extra.json','w'), indent=1); print(out['decomposition_months'])
