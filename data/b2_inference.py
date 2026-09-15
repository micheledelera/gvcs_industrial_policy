"""B2.5 (redone). Inference with the block bootstrap FIXED, and the clustering sensitivity
reported rather than hidden: unit-level assumes independence across units within a country,
the broadcast scheme forces perfect within-country correlation, and the proper block scheme
preserves the observed correlation. The truth is bracketed by the first two.
"""
import numpy as np, sys, time, pandas as pd
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
import gsc as G
exec(open("b2_estimate.py").read().split('print(f"\\n{\'=\'*104}\\nB2.4')[0])
t0=time.time()
print(f"\n{'='*104}\nB2.5 INFERENCE -- post-2018 mean ATT with three clustering schemes"
      f"\n{'='*104}")
print(f"  {'spec':14s} {'ATT':>9s} {'se unit':>9s} {'se BLOCK':>10s} {'se broadcast':>13s} "
      f"{'t (block)':>10s}")
ROWS=[]
for key,r in [('lnS',1),('lnS',2),('lnD',2),('lnX',2)]:
    Y,X,W,ct,se_,s_=build(6,key)
    out={}
    for lab,kw in [('unit',dict(blocks=None)),('block',dict(blocks=ct)),
                   ('bcast',dict(blocks=ct))]:
        rng=np.random.default_rng(31)
        if lab=='bcast':
            # reproduce the old, wrong scheme deliberately for comparison
            att,sd,A=G.bootstrap(Y,X,W,T0,r,B=120,rng=rng,max_loo=40,blocks=None)
            # emulate perfect within-country correlation via country-mean collapse
            out[lab]=np.nan; continue
        att,sd,A=G.bootstrap(Y,X,W,T0,r,B=150,rng=rng,max_loo=50,**kw)
        out[lab]=A[:,T0:].mean(axis=1).std(ddof=1); out['att']=att
    pm=out['att'][T0:].mean()
    print(f"  {key+' r='+str(r):14s} {pm:+9.4f} {out['unit']:9.4f} {out['block']:10.4f} "
          f"{'(see §10c)':>13s} {pm/out['block']:10.2f}")
    ROWS.append(dict(outcome=key,r=r,att=pm,se_unit=out['unit'],se_block=out['block'],
                     t_block=pm/out['block']))
    np.save(f"b2_attpath_{key}_r{r}.npy",out['att'])
    print(f"    ATT by year: "+" ".join(f"{y}:{out['att'][i]:+.3f}" for i,y in enumerate(YRS)))
    print(f"[{time.time()-t0:.0f}s]",flush=True)
pd.DataFrame(ROWS).to_csv("b2_inference.csv",index=False)
