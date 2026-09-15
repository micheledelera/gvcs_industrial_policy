"""lnX with the bootstrap blocked at SECTOR rather than country. Neither Xu nor Cunningham
prescribes a clustering level -- Cunningham uses randomisation inference and no standard
errors, Xu's bootstrap assumes cross-sectional independence -- so the level is a judgement
call and this runs the sector version of it. Same design otherwise: threshold 6+, r = 2,
covariate Dec_k x Post. p-values computed the same three ways as b2_pvalues.py, with the
t reference set by the number of TREATED sector clusters.
"""
import numpy as np, pandas as pd, sys, time
from scipy import stats
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
import gsc as G
exec(open("b2_estimate.py").read().split('print(f"\\n{\'=\'*104}\\nB2.4')[0])
t0=time.time()
KEY,R,THR='lnX',2,6
Y,X,W,ct,se_,s_=build(THR,KEY)
div=np.array([s[:2] for s in se_])
g=G.gsc(Y,X,W,T0,R)
ROWS=[]
for lab,bl in [('ISIC 4-digit sector',se_),('ISIC 2-digit division',div),
               ('country (for reference)',ct)]:
    ng=len(np.unique(bl[W]))
    att,sd,A=G.bootstrap(Y,X,W,T0,R,B=400,rng=np.random.default_rng(101),max_loo=60,blocks=bl)
    d=A[:,T0:].mean(axis=1)
    a=att[T0:].mean(); s=d.std(ddof=1)
    p_perc=float((np.abs(d-d.mean())>=abs(a)).mean())
    p_norm=2*(1-stats.norm.cdf(abs(a/s))); p_t=2*(1-stats.t.cdf(abs(a/s),df=ng-1))
    ps=A.std(axis=0,ddof=1)
    sig=[y for i,y in enumerate(YRS) if y>=2018 and abs(att[i])>1.96*ps[i]]
    print(f"\n{'='*96}\nlnX  r={R}   ATT {a:+.4f}   blocked at {lab} ({ng} treated clusters)"
          f"\n{'='*96}")
    print(f"  bootstrap se                {s:.4f}   t = {a/s:+.2f}")
    print(f"    p, normal approx          {p_norm:.3f}")
    print(f"    p, t with {ng-1:3d} df         {p_t:.3f}")
    print(f"    p, bootstrap percentile   {p_perc:.3f}   (400 draws)")
    print(f"    95% CI                    [{a-1.96*s:+.3f}, {a+1.96*s:+.3f}] log points"
          f"  = [{100*(np.exp(a-1.96*s)-1):+.0f}%, {100*(np.exp(a+1.96*s)-1):+.0f}%]")
    print(f"  post years whose own 95% band excludes zero: {sig if sig else 'NONE'}",flush=True)
    ROWS.append(dict(outcome=KEY,r=R,level=lab,n_clusters=ng,att=a,se=s,
                     p_norm=p_norm,p_t=p_t,p_percentile=p_perc))
    if lab=='ISIC 4-digit sector':
        np.save("b2_paths_lnX_sector.npy",
                np.vstack([Y[:,W].mean(axis=1),g['Y0'].mean(axis=1),att,ps]))
pd.DataFrame(ROWS).to_csv("b2_lnx_sector.csv",index=False)
print(f"\n[{time.time()-t0:.0f}s] saved b2_lnx_sector.csv and b2_paths_lnX_sector.npy")
