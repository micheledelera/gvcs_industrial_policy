"""§10e. The reporting steps skipped when I jumped to B2.7: the three GSC estimation steps
displayed rather than merely executed, the balance table (B2), and the data for the paths
(B4) and gap (B5) figures for the MERGED design. Primary spec: lnS, threshold 6+, r = 2.
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
print(f"\n{'='*100}\nB2.1  STEP 1 -- interactive fixed effects on CONTROLS ONLY\n{'='*100}")
print(f"  controls {int(co.sum()):,} x {len(YRS)} years; r = {R}; covariate Dec_k x Post")
print(f"  beta on Dec_k x Post = {g['beta'][0]:+.6f}  "
      f"(per percentage point of China's 2015-17 share, post-2018)")
print(f"    -> a sector 10pp more Chinese-exposed gives every exporter in it "
      f"{10*g['beta'][0]:+.4f} log points of share after 2018")
fit=(X[:,co,:]@g['beta'] + g['a_c'][None,:] + g['xi'][:,None] + g['F']@g['Lam'].T)
res=Y[:,co]-fit
print(f"  control-side fit: RMSE {np.sqrt((res**2).mean()):.4f} on an outcome with sd "
      f"{Y[:,co].std():.4f}  -> R2 = {1-res.var()/Y[:,co].var():.3f}")
print(f"  control loadings: factor 1 mean {g['Lam'][:,0].mean():+.3f} sd "
      f"{g['Lam'][:,0].std():.3f} | factor 2 mean {g['Lam'][:,1].mean():+.3f} sd "
      f"{g['Lam'][:,1].std():.3f}")

print(f"\n{'='*100}\nB2.2  STEP 2 -- treated loadings by projection on the eleven pre-2018 "
      f"outcomes\n{'='*100}")
Lt=g['Lam_t']
print(f"  treated loadings: factor 1 mean {Lt[:,0].mean():+.3f} sd {Lt[:,0].std():.3f} | "
      f"factor 2 mean {Lt[:,1].mean():+.3f} sd {Lt[:,1].std():.3f}")
pre_rmse=np.sqrt((g['gap'][:T0]**2).mean(axis=0))
print(f"  per-unit PRE-period projection RMSE: median {np.median(pre_rmse):.4f}, "
      f"p90 {np.percentile(pre_rmse,90):.4f}, max {pre_rmse.max():.4f}")
print(f"    (outcome sd is {Y.std():.3f}, so the median treated unit's pre-period is fitted "
      f"to {100*np.median(pre_rmse)/Y.std():.0f}% of the outcome's dispersion)")

print(f"\n{'='*100}\nB2.3  STEP 3 -- imputed counterfactual and ATT_t\n{'='*100}")
att=g['att']
print(f"  treated observed mean (post) {Y[:,W][T0:].mean():+.3f};  imputed counterfactual "
      f"mean (post) {g['Y0'][T0:].mean():+.3f};  ATT {att[T0:].mean():+.4f}")
print(f"  ATT by year: "+" ".join(f"{y}:{att[i]:+.3f}" for i,y in enumerate(YRS)))

print(f"\n{'='*100}\nB2  BALANCE TABLE (A3's version: pre-period outcomes and loadings, since"
      f"\n     time-invariant covariates are absorbed by the loadings)\n{'='*100}")
W_ = [('2007-09',[2007,2008,2009]),('2010-12',[2010,2011,2012]),
      ('2013-14',[2013,2014]),('2015-17',[2015,2016,2017])]
print(f"  {'variable':22s} {'treated':>10s} {'synthetic':>11s} {'donor pool':>12s} "
      f"{'tr - syn':>10s}")
for lab,ys in W_:
    ii=[YRS.index(y) for y in ys]
    a=Y[:,W][ii].mean(); b=g['Y0'][ii].mean(); c=Y[:,co][ii].mean()
    print(f"  lnS {lab:18s} {a:10.3f} {b:11.3f} {c:12.3f} {a-b:+10.3f}")
decu=dec[(pre_yrs>=THR)|clean]
print(f"  {'Dec_k (China share %)':22s} {decu[W].mean():10.1f} {'--':>11s} "
      f"{decu[co].mean():12.1f} {'--':>10s}")
for d in range(R):
    print(f"  {'loading on factor '+str(d+1):22s} {Lt[:,d].mean():10.3f} {'--':>11s} "
          f"{g['Lam'][:,d].mean():12.3f} {'--':>10s}")

print(f"\n{'='*100}\nB3  CORRECTION -- GSC does not yield implied donor weights\n{'='*100}")
print("  In the merge table I wrote that implied per-donor weights 'can be derived' because")
print("  the imputation is linear in control outcomes through F-hat and Lambda-hat. That was")
print("  too optimistic. The treated counterfactual is a_tr + xi + F lambda_tr, and controls")
print("  enter ONLY through F-hat and xi-hat; F-hat comes from an eigendecomposition of the")
print("  control residual matrix, so the map from an individual donor's outcomes to the")
print("  treated counterfactual is not linear and no clean per-donor weight exists.")
print("  GSC weights TIME PERIODS and FACTOR DIRECTIONS, not donors.")
print("  The honest analogue -- controls closest to the treated group in loading space:")
dist=np.sqrt(((g['Lam']-Lt.mean(axis=0))**2).sum(axis=1))
ctc=ct[co]; sec=se_[co]
for j in np.argsort(dist)[:8]:
    print(f"    {nm(ctc[j]):16s} {str(sec[j]):>6s}   loading distance {dist[j]:.4f}")

att_b,sd_b,A=G.bootstrap(Y,X,W,T0,R,B=200,rng=np.random.default_rng(77),blocks=ct,max_loo=60)
np.save("b2_report_paths.npy",np.vstack([Y[:,W].mean(axis=1),g['Y0'].mean(axis=1),att,sd_b]))
print(f"\n[{time.time()-t0:.0f}s] saved paths + per-year block-bootstrap se")
