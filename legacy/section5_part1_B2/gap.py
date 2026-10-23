# B.40: decree-to-polling gap g by clearance week (uniform over the cycle), Prop. 22 check.
import numpy as np, json
from pipeline import *
rng=np.random.default_rng(1)
cal=Calendar(gen_timeline(0.0,rng),'O0',-WY*60,WY*80)
calO9=Calendar(gen_timeline(0.0,rng),'O9',-WY*60,WY*80)
res={}
for lab,cl,opt,pres in [('O0_earliest',cal,'O0','earliest'),('O0_latest',cal,'O0','latest'),('O9_earliest',calO9,'O9','earliest'),('O9_latest',calO9,'O9','latest')]:
    g=[]; D=[]
    for c in range(0,CYC*3):
        tp=schedule(cl,c,opt,BASE['DEC']+BASE['GMIN'],pres)
        g.append(tp-c-BASE['DEC']); D.append(tp-c)
    g=np.array(g); D=np.array(D)
    res[lab]=dict(g_min=int(g.min()),g_med=float(np.median(g)),g_max=int(g.max()),D_mean_weeks=float(D.mean()),D_max=int(D.max()))
# Prop 22 closed form, non-election year, earliest: D(w)=GAP if w<=W_CUT else max(52-w, GAP)
GAP=BASE['DEC']+BASE['GMIN']
Dw=[GAP if w<=W_CUT else max(WY-w,GAP) for w in range(WY)]
res['prop22_nonelection_mean']=float(np.mean(Dw)); res['prop22_jump']=int(Dw[W_CUT+1]-Dw[W_CUT])
# numerical check on a non-election year of the calendar (year 1 = 2026)
Dn=[schedule(cal,WY*1+w,'O0',GAP,'earliest')-(WY*1+w) for w in range(WY)]
res['check_year2026']=dict(mean=float(np.mean(Dn)), equal=bool(Dn==Dw))
Dn3=[schedule(cal,WY*3+w,'O0',GAP,'earliest')-(WY*3+w) for w in range(WY)]   # 2028: next year 2029 is an election year
res['year2028_mean']=float(np.mean(Dn3))
Dn4=[schedule(cal,WY*4+w,'O0',GAP,'earliest')-(WY*4+w) for w in range(WY)]   # 2029 election year
res['year2029_mean']=float(np.mean(Dn4)); res['year2029_max']=int(max(Dn4))
Dlat=[schedule(cal,WY*1+w,'O0',GAP,'latest')-(WY*1+w) for w in range(WY)]
res['latest_nonelection_mean']=float(np.mean(Dlat))
json.dump(res,open('gap.json','w'),indent=1); print(json.dumps(res,indent=1))
