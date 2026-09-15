"""Diagnosing the B2 bootstrap. Two suspicious features:
  (a) the pre-period mean gap has se 0.0000 -- exactly zero in every draw
  (b) the post-period se is 0.57-0.73, larger than the estimate itself
"""
import numpy as np, sys, time
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
import gsc as G
exec(open("b2_estimate.py").read().split('print(f"\\n{\'=\'*104}\\nB2.4')[0])
Y,X,W,ct,se_,s_=build(6,'lnS')
print(f"panel {Y.shape}, treated {int(W.sum())}, countries {len(set(ct))}\n")

# (a) is the pre-period mean gap zero BY CONSTRUCTION?
g=G.gsc(Y,X,W,T0,2)
r=2
print("(a) step 2 regresses the treated pre-period on [1, F_pre] -- an intercept is included,")
print("    so the pre-period residuals sum to zero per unit by OLS. Check:")
gap_pre=g['gap'][:T0]
print(f"    mean pre-period gap per treated unit: max |mean| = "
      f"{np.abs(gap_pre.mean(axis=0)).max():.2e}")
print(f"    -> the MEAN pre-period gap is mechanically zero. It is NOT a pre-trend test.")
print(f"    year-by-year pre gaps are informative; their sd across years = "
      f"{g['att'][:T0].std():.4f}")

# (b) unit-level vs my block scheme
print("\n(b) bootstrap standard error of the post-2018 mean ATT, three schemes:")
for lab,kw in [("unit-level (blocks=None)",dict(blocks=None)),
               ("my block scheme (one series broadcast per country)",dict(blocks=ct))]:
    t=time.time()
    att,sd,A=G.bootstrap(Y,X,W,T0,r,B=120,rng=np.random.default_rng(5),max_loo=40,**kw)
    pm=att[T0:].mean(); ps=A[:,T0:].mean(axis=1).std(ddof=1)
    print(f"    {lab:52s} ATT {pm:+.4f}  se {ps:.4f}  ({time.time()-t:.0f}s)")
print("\n    My block scheme assigns ONE residual series to EVERY unit in a country, which")
print("    imposes perfect within-country correlation rather than preserving the observed")
print("    correlation. That is not a block bootstrap; it is an upper bound on clustering,")
print("    and it is what inflates the se to 0.57-0.73.")
