"""§6b. Holding the dollars fixed: the HIGH-vs-LOW contrast within size strata.

§6 showed the whole gravity-vs-SC disagreement is size weighting. Unweighted rows are null
(+0.003, −0.021, −0.084); size-weighted rows are significant (+0.193, +0.382). Those answer
"did the typical targeted SECTOR benefit" and "did the typical targeted DOLLAR benefit".

Rather than choosing, stratify. Split country-sectors into quintiles of pre-period US export
value and run the identical regression within each. If the effect appears only in the top
quintile, targeting pays off among large flows specifically -- a real and reportable fact.
If it is uniform across quintiles, the weighting was pure composition and either aggregate
is defensible.

Specification, identical within each stratum and identical to §6 row 6:

    log X_ist = beta (HIGH_is x post_t) + alpha_is + alpha_it + alpha_st + e

on HIGH + LOW units with a complete positive 2007-2024 US series. Reported unweighted and
weighted by pre-period exports, plus PPML on levels, so the three estimators of §6 can be
compared WITHIN a size class rather than across them.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc15_bridge.py").read().split('FE="fe_is + fe_it + fe_st"')[0])
FE="fe_is + fe_it + fe_st"

HL=X[((X['HIGH']==1)|(X['LOW']==1)) & X['complete']].copy()
sz=HL.groupby('pair')['wpre'].first()
HL['q']=HL['pair'].map(pd.qcut(sz.rank(method='first'),5,labels=[1,2,3,4,5]))
log(f"HIGH+LOW complete-series: {HL['pair'].nunique():,} units, {len(HL):,} rows")

def fit(d,kind,wt=None):
    d=d.copy()
    if kind!='ppml':
        d=d[d['X']>0].copy(); d['ly']=np.log(d['X'])
    try:
        if kind=='ppml':
            m=pf.fepois(f"X ~ HIGHxP | {FE}",data=d,vcov={"CRV1":"cl"},iwls_maxiter=400,
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=20000),
                        lean=True,store_data=False,copy_data=False)
        else:
            m=pf.feols(f"ly ~ HIGHxP | {FE}",data=d,weights=wt,vcov={"CRV1":"cl"},
                       lean=True,store_data=False,copy_data=False)
        t=m.tidy().loc['HIGHxP']
        r=(t['Estimate'],t['Std. Error'],t['Pr(>|t|)'],int(m._N))
        del m; gc.collect(); return r
    except Exception as e:
        return (np.nan,np.nan,np.nan,0)

def cell(r):
    if not np.isfinite(r[0]): return f"{'--':>21s}"
    st='***' if r[2]<.01 else '**' if r[2]<.05 else '*' if r[2]<.10 else ''
    return f"{r[0]:>+8.3f} ({r[1]:.3f}){st:3s}"

tot=HL.groupby('pair')['wpre'].first().sum()
print(f"\n{'='*112}")
print(f"HIGH vs LOW, WITHIN size quintiles of pre-period US exports")
print(f"{'='*112}")
print(f"{'quintile':>9s}{'units':>7s}{'HIGH':>6s}{'median $k':>12s}{'% of value':>11s}"
      f"{'OLS logs, unweighted':>24s}{'OLS logs, weighted':>23s}{'PPML levels':>22s}")
rows=[]
for q in [1,2,3,4,5]:
    d=HL[HL['q']==q]
    u=d.groupby('pair').first()
    a=fit(d,'ols'); b=fit(d,'ols','wpre'); c=fit(d,'ppml')
    print(f"{q:>9d}{len(u):>7d}{int(u['HIGH'].sum()):>6d}{u['wpre'].median():>12,.0f}"
          f"{100*u['wpre'].sum()/tot:>10.1f}%{cell(a):>24s}{cell(b):>23s}{cell(c):>22s}")
    rows.append(dict(q=q,units=len(u),high=int(u['HIGH'].sum()),
                     med=u['wpre'].median(),valshare=u['wpre'].sum()/tot,
                     ols=a[0],ols_se=a[1],ols_p=a[2],
                     olsw=b[0],olsw_se=b[1],olsw_p=b[2],
                     ppml=c[0],ppml_se=c[1],ppml_p=c[2]))
a=fit(HL,'ols'); b=fit(HL,'ols','wpre'); c=fit(HL,'ppml')
print(f"{'-'*112}")
print(f"{'ALL':>9s}{HL['pair'].nunique():>7d}"
      f"{int(HL.groupby('pair')['HIGH'].first().sum()):>6d}"
      f"{HL.groupby('pair')['wpre'].first().median():>12,.0f}{100.0:>10.1f}%"
      f"{cell(a):>24s}{cell(b):>23s}{cell(c):>22s}")
pd.DataFrame(rows).to_csv("sc16_sizestrata.csv",index=False)
print(f"\n  §6 reference: unweighted −0.084 (p=0.17), weighted +0.382 (p=0.018), "
      f"PPML +0.193 (p=0.009)")
log("DONE")
