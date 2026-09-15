"""At what level should the B2 standard errors be clustered? Two separate questions:
(1) where is the residual dependence, measured directly from the model's own residuals; and
(2) what does the ATT's standard error look like when the bootstrap is blocked at each
candidate level. Primary spec: lnS, threshold 6+, r = 2.
"""
import numpy as np, pandas as pd, sys, time
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
import gsc as G
exec(open("b2_estimate.py").read().split('print(f"\\n{\'=\'*104}\\nB2.4')[0])
t0=time.time()
KEY,R,THR='lnS',2,6
Y,X,W,ct,se_,s_=build(THR,KEY)
g=G.gsc(Y,X,W,T0,R)
co=~W
div=np.array([s[:2] for s in se_])      # ISIC 2-digit division from the 4-digit code
att=g['att'][T0:].mean()
print(f"\n{'='*100}\nPANEL GEOMETRY\n{'='*100}")
for lab,v in [('country',ct),('ISIC 4-digit sector',se_),('ISIC 2-digit division',div),
              ('unit (country-sector)',np.arange(len(ct)))]:
    print(f"  {lab:24s} all {len(np.unique(v)):5,} groups | "
          f"TREATED {len(np.unique(v[W])):5,} groups, "
          f"mean treated group size {W.sum()/len(np.unique(v[W])):6.1f}")

print(f"\n{'='*100}\nWHERE THE DEPENDENCE ACTUALLY IS -- intraclass correlation of the\n"
      f"  control-side residuals (2,223 clean controls x 18 years, r = 2 model)\n{'='*100}")
fit=(X[:,co,:]@g['beta'] + g['a_c'][None,:] + g['xi'][:,None] + g['F']@g['Lam'].T)
res=Y[:,co]-fit
def icc(r2d, grp):
    """One-way ICC: mean within-group cross-product over total variance, averaged over years."""
    num=den=0.0; npair=0
    grp=pd.factorize(np.asarray(grp))[0]
    order=np.argsort(grp); gs=grp[order]
    bounds=np.r_[0,np.where(np.diff(gs)!=0)[0]+1,len(gs)]
    for t in range(r2d.shape[0]):
        x=r2d[t][order]; x=x-x.mean(); den+=(x**2).sum(); npair_t=0; num_t=0.0
        for a,b in zip(bounds[:-1],bounds[1:]):
            n=b-a
            if n<2: continue
            s=x[a:b].sum(); num_t+=s*s-(x[a:b]**2).sum(); npair_t+=n*(n-1)
        num+=num_t; npair+=npair_t
    return (num/npair)/(den/(r2d.shape[0]*r2d.shape[1]))
for lab,v in [('country',ct[co]),('ISIC 4-digit sector',se_[co]),
              ('ISIC 2-digit division',div[co])]:
    r=icc(res,v); mbar=len(v)/len(np.unique(v))
    print(f"  within-{lab:22s} rho = {r:+.4f}   mean group size {mbar:6.1f}   "
          f"Moulton inflation sqrt(1+(m-1)rho) = {np.sqrt(max(1+(mbar-1)*r,0)):.2f}x")
rr=np.corrcoef(res[:T0].T@np.ones(1) if False else res[:T0])   # year-to-year, within unit
print(f"  within-UNIT serial correlation of residuals, mean |lag-1| = "
      f"{np.mean([np.corrcoef(res[t],res[t+1])[0,1] for t in range(T0-1)]):+.3f}")
print("    (Xu's bootstrap resamples WHOLE residual series per unit, so serial correlation of")
print("     any form is already handled; it is the CROSS-SECTIONAL term that his Assumption 5")
print("     assumes away and that the numbers above measure.)")

print(f"\n{'='*100}\nSE ON THE POOLED ATT ({att:+.4f}) BY BLOCKING LEVEL\n{'='*100}")
ROWS=[]
for lab,bl in [('unit (Xu default)',None),('ISIC 4-digit sector',se_),
               ('ISIC 2-digit division',div),('country',ct)]:
    a_b,sd,A=G.bootstrap(Y,X,W,T0,R,B=200,rng=np.random.default_rng(77),blocks=bl,max_loo=60)
    s=A[:,T0:].mean(axis=1).std(ddof=1)
    ng=len(np.unique(bl[W])) if bl is not None else int(W.sum())
    t=att/s; from scipy import stats
    p=2*(1-stats.t.cdf(abs(t),max(ng-1,1)))
    print(f"  {lab:24s} treated clusters {ng:5,}  se {s:.4f}  t {t:+5.2f}  "
          f"p(t,{max(ng-1,1)} df) {p:.3f}   [{time.time()-t0:.0f}s]")
    ROWS.append(dict(level=lab,n_clusters=ng,se=s,t=t,p=p))
D=pd.DataFrame(ROWS); D.to_csv("b2_clustering.csv",index=False)
v=dict(zip(D.level,D.se**2))
v2=v['country']+v['ISIC 4-digit sector']-v['unit (Xu default)']
print(f"\n  two-way (country + sector, Cameron-Gelbach-Miller): "
      f"se {np.sqrt(max(v2,0)):.4f}  t {att/np.sqrt(max(v2,1e-12)):+.2f}")
print(f"[{time.time()-t0:.0f}s] saved b2_clustering.csv")
