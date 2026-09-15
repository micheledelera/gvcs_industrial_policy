"""Statistical significance of the B2 ATTs, computed three ways so the reader can see how
much the answer depends on the inference scheme rather than the estimate.

  normal / t approximation from the country-blocked bootstrap se
  bootstrap PERCENTILE p-value: share of bootstrap draws on the far side of zero, doubled
  the unit-level (non-clustered) version, shown only to expose how anti-conservative it is

25 treated countries, so a t distribution with 24 df is the honest reference rather than
the normal. NOTE: the ADH-style randomisation test (Part C) is a different and stronger
test and has NOT been run for this design yet.
"""
import numpy as np, pandas as pd, sys, time
from scipy import stats
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
import gsc as G
exec(open("b2_estimate.py").read().split('print(f"\\n{\'=\'*104}\\nB2.4')[0])
t0=time.time()
ROWS=[]
for KEY,R in [('lnS',1),('lnS',2),('lnD',2),('lnX',2)]:
    Y,X,W,ct,se_,s_=build(6,KEY)
    ntr_c=len(set(ct[W]))
    res={}
    for lab,kw in [('block',dict(blocks=ct)),('unit',dict(blocks=None))]:
        att,sd,A=G.bootstrap(Y,X,W,T0,R,B=400,rng=np.random.default_rng(101),
                             max_loo=60,**kw)
        d=A[:,T0:].mean(axis=1)                    # bootstrap draws of the post-2018 ATT
        res[lab]=dict(att=att[T0:].mean(),se=d.std(ddof=1),draws=d,path=att,
                      pathse=A.std(axis=0,ddof=1))
    a=res['block']['att']; sb=res['block']['se']; db=res['block']['draws']
    # bootstrap distribution is centred on the estimate under the null-imposed DGP (delta=0),
    # so recentre on zero and ask how often |draw| >= |estimate|
    dc=db-db.mean()
    p_perc=float((np.abs(dc)>=abs(a)).mean())
    p_norm=2*(1-stats.norm.cdf(abs(a/sb)))
    p_t=2*(1-stats.t.cdf(abs(a/sb),df=ntr_c-1))
    su=res['unit']['se']
    print(f"\n{'='*96}\n{KEY}  r={R}   ATT {a:+.4f}   ({ntr_c} treated countries)\n{'='*96}")
    print(f"  country-blocked bootstrap se {sb:.4f}   t = {a/sb:+.2f}")
    print(f"    p, normal approx            {p_norm:.3f}")
    print(f"    p, t with {ntr_c-1:2d} df           {p_t:.3f}")
    print(f"    p, bootstrap percentile     {p_perc:.3f}   (400 draws)")
    print(f"    95% CI                      [{a-1.96*sb:+.3f}, {a+1.96*sb:+.3f}] log points"
          f"  = [{100*(np.exp(a-1.96*sb)-1):+.0f}%, {100*(np.exp(a+1.96*sb)-1):+.0f}%]")
    print(f"  for contrast, UNIT-level se   {su:.4f}   t = {a/su:+.2f}   "
          f"p = {2*(1-stats.norm.cdf(abs(a/su))):.3f}   <- anti-conservative, ignores clustering")
    pp=res['block']['path']; ps=res['block']['pathse']
    sig=[y for i,y in enumerate(YRS) if y>=2018 and abs(pp[i])>1.96*ps[i]]
    print(f"  post years whose own 95% band excludes zero: {sig if sig else 'NONE'}")
    ROWS.append(dict(outcome=KEY,r=R,att=a,se_block=sb,p_norm=p_norm,p_t=p_t,
                     p_percentile=p_perc,se_unit=su,n_ctry=ntr_c))
    print(f"  [{time.time()-t0:.0f}s]",flush=True)
pd.DataFrame(ROWS).to_csv("b2_pvalues.csv",index=False)
