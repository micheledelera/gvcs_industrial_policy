"""§5i addendum 3. Where does the §5h U come from?

§5h's event study (HIGH vs the 1,484 weighted donors, unit + sector x year FE) showed the
HIGH-LOW differential falling 0.24 log points over 2007-2017. §5i's raw bin means show the
opposite: Q4 grew MORE than zero-policy units pre-2018 (+0.64 vs +0.56). Both are computed
on the same 5,166 units, so the difference must come from one of the ingredients §5h added:

  the comparison group    all LOW (3,658) vs the drawn placebos (500) vs the SC's weighted
                          donors (1,484) -- the last is SELECTED ON PRE-PERIOD FIT, which
                          could induce a pre-trend all by itself
  the fixed effects       unit + year vs unit + sector x year

This crosses the two. Nothing else changes.
"""
import pandas as pd, numpy as np, pyfixest as pf, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])
Lfull=np.hstack([Lp,Lq])

def pool_of(a, lopool):
    m=(np.abs(Dv[lopool]-Dv[a])<=DECB)&(np.abs(Sv[lopool]-Sv[a])<=SZB)&(CT[lopool]!=CT[a])
    cand=lopool[m]
    if len(cand)<8: return None,None
    pool=cand[np.argsort(np.sqrt(((Z[cand]-Z[a])**2).sum(axis=1)))[:K]]
    return pool, sc(np.hstack([Lp[pool],Z[pool]]),np.concatenate([Lp[a],Z[a]]))

ROWS=[]
for meas,lab in [('tgt','TARGETING'),('vol','VOLUME')]:
    v=M[meas].values; pos=v[v>0]
    hi=np.where(v>=np.quantile(pos,.75))[0]; lo=np.where(v<=np.quantile(pos,.25))[0]
    drawn=rng.choice(lo,size=min(NPLAC,len(lo)),replace=False)
    swt=np.zeros(len(M)); used=set()
    for a in hi:
        p,w=pool_of(a,lo)
        if p is None: continue
        swt[p]+=w; used.update(p[w>1e-6].tolist())
    eff=np.array(sorted(used))
    # untrimmed band donors: eligible by the band rule but NOT selected on fit
    band=set()
    for a in hi:
        m=(np.abs(Dv[lo]-Dv[a])<=DECB)&(np.abs(Sv[lo]-Sv[a])<=SZB)&(CT[lo]!=CT[a])
        band.update(lo[m].tolist())
    bnd=np.array(sorted(band))
    HI=np.zeros(len(M),bool); HI[hi]=True
    print(f"\n{'='*104}\n{lab}: HIGH {len(hi)} | all LOW {len(lo)} | band-eligible {len(bnd)} "
          f"| drawn {len(drawn)} | weighted donors {len(eff)}\n{'='*104}")
    for cname,ctrl in [("all LOW",lo),("band-eligible",bnd),("drawn placebos",drawn),
                       ("SC weighted donors",eff)]:
        units=np.concatenate([hi,ctrl])
        d=pd.DataFrame({'u':np.repeat(units,len(YRS)),'t':np.tile(YRS,len(units)),
                        'ly':Lfull[units].ravel()})
        d['i']=CT[d['u'].values]; d['k']=SE[d['u'].values]
        d['high']=HI[d['u'].values].astype(float)
        for y in YRS:
            if y!=2017: d[f"y{y}"]=d['high']*(d['t']==y)
        rhs=" + ".join(f"y{y}" for y in YRS if y!=2017)
        for fname,fe in [("unit + year",'u + t'),("unit + sector x year",'u + t^k')]:
            m=pf.feols(f"ly ~ {rhs} | {fe}",data=d,vcov={'CRV1':'i'})
            co,se=m.coef(),m.se()
            b=np.array([co[f"y{y}"] for y in YRS if y!=2017])
            s=np.array([se[f"y{y}"] for y in YRS if y!=2017])
            ys=[y for y in YRS if y!=2017]
            prem=np.array([b[ys.index(y)] for y in PRE if y!=2017])
            pos_=np.array([b[ys.index(y)] for y in POST])
            mt=np.abs(prem/np.array([s[ys.index(y)] for y in PRE if y!=2017])).max()
            print(f"  {cname:20s} {fname:22s}  2007 {b[0]:+.3f}  2012 {b[ys.index(2012)]:+.3f} "
                  f" 2015 {b[ys.index(2015)]:+.3f}  2018 {b[ys.index(2018)]:+.3f} "
                  f" 2024 {b[-1]:+.3f}   pre mean {prem.mean():+.3f} (max|t| {mt:.2f})"
                  f"  post mean {pos_.mean():+.3f}")
            for y,bb,ss in zip(ys,b,s):
                ROWS.append(dict(meas=meas,ctrl=cname,fe=fname,t=y,b=bb,s=ss))
            ROWS.append(dict(meas=meas,ctrl=cname,fe=fname,t=2017,b=0.0,s=0.0))
pd.DataFrame(ROWS).sort_values(["meas","ctrl","fe","t"]).to_csv("sc19d_locate.csv",index=False)
log("DONE")
