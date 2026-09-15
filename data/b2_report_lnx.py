"""The B4/B5 figures for lnX -- log US imports, the A0 CHECK outcome -- alongside the primary
lnS. Same merged design: threshold 6+, r = 2, covariate Dec_k x Post, country-blocked
bootstrap. Saves paths for plot_b2_lnx.py and prints the pieces the figure cannot carry.
"""
import numpy as np, pandas as pd, sys, time
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
import gsc as G
exec(open("b2_estimate.py").read().split('print(f"\\n{\'=\'*104}\\nB2.4')[0])
t0=time.time()
KEY,R,THR='lnX',2,6
Y,X,W,ct,se_,s_=build(THR,KEY)
g=G.gsc(Y,X,W,T0,R)
co=~W
att=g['att']
print(f"\n{'='*100}\nlnX  r={R}  thr={THR}: {int(W.sum()):,} treated, {int(co.sum()):,} clean "
      f"controls, {len(np.unique(ct[W])):,} treated countries\n{'='*100}")
print(f"  beta on Dec_k x Post = {g['beta'][0]:+.6f}")
fit=(X[:,co,:]@g['beta'] + g['a_c'][None,:] + g['xi'][:,None] + g['F']@g['Lam'].T)
res=Y[:,co]-fit
print(f"  control-side fit: RMSE {np.sqrt((res**2).mean()):.4f} on sd {Y[:,co].std():.4f} "
      f"-> R2 = {1-res.var()/Y[:,co].var():.3f}")
pre_rmse=np.sqrt((g['gap'][:T0]**2).mean(axis=0))
print(f"  per-unit PRE projection RMSE: median {np.median(pre_rmse):.4f} "
      f"({100*np.median(pre_rmse)/Y.std():.0f}% of the outcome sd {Y.std():.3f})")
print(f"  ATT (post mean) {att[T0:].mean():+.4f};  by year: "
      +" ".join(f"{y}:{att[i]:+.3f}" for i,y in enumerate(YRS)))

print(f"\n  BALANCE, lnX")
W_=[('2007-09',[2007,2008,2009]),('2010-12',[2010,2011,2012]),
    ('2013-14',[2013,2014]),('2015-17',[2015,2016,2017])]
print(f"  {'window':22s} {'treated':>10s} {'synthetic':>11s} {'donor pool':>12s} {'tr-syn':>9s}")
for lab,ys in W_:
    ii=[YRS.index(y) for y in ys]
    a=Y[:,W][ii].mean(); b=g['Y0'][ii].mean(); c=Y[:,co][ii].mean()
    print(f"  lnX {lab:18s} {a:10.3f} {b:11.3f} {c:12.3f} {a-b:+9.3f}")

# the decomposition the primary figure carries in its footnote: how much of the gap is the
# counterfactual falling rather than the treated rising
d_tr=Y[:,W].mean(axis=1)[T0:].mean()-Y[:,W].mean(axis=1)[:T0].mean()
d_cf=g['Y0'].mean(axis=1)[T0:].mean()-g['Y0'].mean(axis=1)[:T0].mean()
print(f"\n  pre->post change, treated      {d_tr:+.4f} log points")
print(f"  pre->post change, counterfactual {d_cf:+.4f}")
print(f"  share of the gap that is the counterfactual moving: "
      f"{abs(d_cf)/(abs(d_tr)+abs(d_cf)):.1%}")

# size quintiles, to see whether §10d's small-unit story survives in levels
q=pd.qcut(s_[W],5,labels=False,duplicates='drop')
print(f"\n  ATT by 2015-17 size quintile (lnX)")
for k in range(q.max()+1):
    m=np.zeros(len(W),bool); m[np.where(W)[0][q==k]]=True
    gk=g['gap'][:,q==k]
    print(f"    Q{k+1}  n {int((q==k).sum()):5,}  mean 2015-17 imports "
          f"${np.exp(np.log(s_[W][q==k]).mean())/1e3:9.1f}m   ATT {gk[T0:].mean():+.4f}")

att_b,sd_b,A=G.bootstrap(Y,X,W,T0,R,B=200,rng=np.random.default_rng(77),blocks=ct,max_loo=60)
np.save("b2_report_paths_lnX.npy",
        np.vstack([Y[:,W].mean(axis=1),g['Y0'].mean(axis=1),att,sd_b]))
print(f"\n  country-blocked se on the post ATT: {A.mean(axis=0)[T0:].mean():.4f}" if False else "")
print(f"[{time.time()-t0:.0f}s] saved b2_report_paths_lnX.npy")
