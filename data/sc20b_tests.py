"""§5j addendum. Joint pre-trend test and the extensive/intensive split.

Two things §5j leaves open.

1  The continuous event study's pre-2018 coefficients are individually small but all of one
   sign (targeting: mean -0.017 to -0.021, every year negative relative to 2017). §3m
   established that year-by-year insignificance is NOT a pre-trend test -- the joint Wald
   test rejected in all 24 cases there. So: joint Wald test of all ten pre-2018
   coefficients, plus a dose-specific linear trend in the DiD.

2  §5j's quintile table shows the response DECREASING in dose: the jump is from zero policy
   to the lowest positive quintile, and higher quintiles give less. That says the variable
   is behaving as an extensive-margin indicator, not a dose. This decomposes it explicitly:

       ln X = a_ik + g_t + b1 (Any_ik x Post) + b2 (dose_ik x Post) + e

   with dose standardised WITHIN the positive units, so b1 is the zero-to-positive step and
   b2 the return to more policy among users. If b1 carries it and b2 is nil, the result is
   about which country-sectors appear in GTA at all.
"""
import pandas as pd, numpy as np, pyfixest as pf, time
from scipy import stats
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])
Lfull=np.hstack([Lp,Lq])
PREY=[y for y in PRE if y!=2017]

for meas,lab in [('tgt','TARGETING'),('vol','VOLUME')]:
    v=M[meas].values
    d=pd.DataFrame({'u':np.repeat(np.arange(len(M)),len(YRS)),'t':np.tile(YRS,len(M)),
                    'ly':Lfull.ravel()})
    d['i']=CT[d['u'].values]; d['k']=SE[d['u'].values]
    d['post']=(d['t']>=2018).astype(float)
    z=(v-v.mean())/v.std(); d['z']=z[d['u'].values]
    print(f"\n{'='*104}\n{lab}\n{'='*104}")

    # ---- 1. joint pre-trend test ----
    for y in YRS:
        if y!=2017: d[f"z{y}"]=d['z']*(d['t']==y)
    rhs=" + ".join(f"z{y}" for y in YRS if y!=2017)
    ys=[y for y in YRS if y!=2017]
    for fname,fe in [("sector x year","u + t^k"),("+ country x year","u + t^k + t^i")]:
        m=pf.feols(f"ly ~ {rhs} | {fe}",data=d,vcov={'CRV1':'i'})
        names=list(m.coef().index)
        R=np.zeros((len(PREY),len(names)))
        for r,y in enumerate(PREY): R[r,names.index(f"z{y}")]=1.0
        w=m.wald_test(R=R)
        p=float(np.asarray(w['pvalue']).ravel()[0]); st=float(np.asarray(w['statistic']).ravel()[0])
        print(f"  joint test, 10 pre-2018 coefs = 0 ({fname:16s}): "
              f"stat {st:7.2f}, p = {p:.4f}   -> "
              f"{'REJECTS parallel pre-trends' if p<0.10 else 'does not reject'}")
    # dose-specific linear trend
    d['zt']=d['z']*(d['t']-2017)
    for fname,fe in [("sector x year","u + t^k"),("+ country x year","u + t^k + t^i")]:
        m=pf.feols(f"ly ~ z:post + zt | {fe}",data=d,vcov={'CRV1':'i'})
        co,se=m.coef(),m.se()
        b,s=co['z:post'],se['z:post']; bt,st_=co['zt'],se['zt']
        print(f"  + dose linear trend ({fname:16s}): dose x Post {b:>+7.4f} ({s:.4f}) "
              f"t={b/s:>+5.2f}   trend {bt:>+7.4f} ({st_:.4f}) t={bt/st_:>+5.2f}")
    # trend fitted on the PRE period only, then imposed
    dp=d[d['t']<2018]
    for fname,fe in [("sector x year","u + t^k")]:
        m=pf.feols(f"ly ~ zt | {fe}",data=dp,vcov={'CRV1':'i'})
        g=m.coef()['zt']
        d['ly_adj']=d['ly']-g*d['zt']
        m2=pf.feols("ly_adj ~ z:post | u + t^k",data=d,vcov={'CRV1':'i'})
        b,s=m2.coef()['z:post'],m2.se()['z:post']
        print(f"  pre-period trend {g:+.4f}/yr removed, then dose x Post: "
              f"{b:>+7.4f} ({s:.4f}) t={b/s:>+5.2f}")

    # ---- 2. extensive vs intensive ----
    any_=(v>0).astype(float)
    pos=v>0
    zi=np.zeros(len(v)); zi[pos]=(v[pos]-v[pos].mean())/v[pos].std()
    d['anyp']=any_[d['u'].values]; d['zin']=zi[d['u'].values]
    print(f"\n  extensive vs intensive  ({int(pos.sum())} policy users of {len(v)} units; "
          f"dose standardised within users)")
    for fname,fe in [("unit + year","u + t"),("+ sector x year","u + t^k"),
                     ("+ country x year","u + t^i"),("both","u + t^k + t^i")]:
        m=pf.feols("ly ~ anyp:post + zin:post | "+fe,data=d,vcov={'CRV1':'i'})
        co,se=m.coef(),m.se()
        b1,s1=co['anyp:post'],se['anyp:post']; b2,s2=co['zin:post'],se['zin:post']
        print(f"    {fname:18s} any policy {b1:>+7.4f} ({s1:.4f}) t={b1/s1:>+5.2f}   "
              f"| dose among users {b2:>+7.4f} ({s2:.4f}) t={b2/s2:>+5.2f}")
    # pre-period placebo on both margins (dp rebuilt so the new columns are present)
    dp2=d[d['t']<2018]
    m=pf.feols("ly ~ anyp:zt + zin:zt | u + t^k",data=dp2,vcov={'CRV1':'i'})
    co,se=m.coef(),m.se()
    print(f"    PRE-2018 placebo trends: any policy {co['anyp:zt']:+.4f} "
          f"({se['anyp:zt']:.4f}) t={co['anyp:zt']/se['anyp:zt']:+.2f}   "
          f"| dose among users {co['zin:zt']:+.4f} ({se['zin:zt']:.4f}) "
          f"t={co['zin:zt']/se['zin:zt']:+.2f}")

    # quintile shape under the saturated FE, which §5j did not run
    pq=np.quantile(v[v>0],[.2,.4,.6,.8])
    qb=np.digitize(v,np.concatenate([[1e-12],pq]))
    for b in range(1,6): d[f'Q{b}']=(qb==b).astype(float)[d['u'].values]
    rhsq=" + ".join(f"Q{b}:post" for b in range(1,6))
    for fname,fe in [("+ country x year","u + t^i"),("both","u + t^k + t^i")]:
        m=pf.feols(f"ly ~ {rhsq} | {fe}",data=d,vcov={'CRV1':'i'})
        co,se=m.coef(),m.se()
        print(f"    quintiles, {fname:16s} "+"  ".join(
            f"Q{b} {co[f'Q{b}:post']:+.3f}({se[f'Q{b}:post']:.3f})" for b in range(1,6)))
log("DONE")
