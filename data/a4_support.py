"""A4. Common support. Under GSC this is FACTOR-LOADING OVERLAP (Xu Fig. 3b), not the
convex hull of outcomes -- the diagnostic Xu names as essential precisely because GSC will
happily extrapolate where canonical SC would visibly fail.

Design-stage legitimate: step 1 fits the IFE model on CONTROL data only, and each treated
unit's loadings come from its ELEVEN PRE-2018 outcomes. No treated post-treatment outcome
is touched, so the Rubin/ADH separation of design from estimation holds.

Reports, for r = 1..4 (per §9c we do not trust the CV's pick):
  the CV curve on our actual panel
  per-dimension overlap of treated loadings within the controls' range
  strict convex-hull containment of treated loadings in the control cloud
  Mahalanobis position of treated loadings in the control distribution
  whether imputed counterfactuals stay inside the observed data range (Xu's other check)
"""
import numpy as np, pandas as pd, sys, gc
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
import gsc as G
from scipy.spatial import ConvexHull, Delaunay
exec(open("a1_treated.py").read().split('print(f"sample:')[0])
nev=(pre_yrs==0); clean=nev&(post_yrs==0); tr=(pre_yrs>=6)
use=tr|clean
lnS_u=lnS[use]; CTu=CT[use]; SEu=SE[use]; TRu=tr[use]; DECu=dec[use]
T0=len(PRE); T=len(YRS)
post=np.array([1.0 if y>=2018 else 0.0 for y in YRS])
X=(DECu[None,:,None]*post[:,None,None])                      # T x N x 1 : Dec_k x Post
Y=lnS_u.T                                                    # T x N
print(f"panel: {Y.shape[1]:,} units ({int(TRu.sum()):,} treated, "
      f"{int((~TRu).sum()):,} clean controls) x {T} years; X = Dec_k x Post\n")

print(f"{'='*100}\nCV curve for r on the ACTUAL panel (per §9c, reported not obeyed)\n{'='*100}")
cv,best=G.cv_r(Y,X,TRu,T0,rmax=5)
print("  "+"   ".join(f"r={r}: {v:.5f}" for r,v in cv.items()))
print(f"  CV minimiser: r = {best}"
      f"   (§9c: at T0 = 11 the CV picks the truth only 37-50% of the time and errs DOWNWARD)")

print(f"\n{'='*100}\nFACTOR-LOADING OVERLAP, treated vs clean controls\n{'='*100}")
print(f"  {'r':>2s} {'per-dim inside':>15s} {'in convex hull':>15s} "
      f"{'median Mahalanobis pctile':>26s} {'counterfactual in range':>24s}")
rows=[]
for r in [1,2,3,4]:
    g=G.gsc(Y,X,TRu,T0,r)
    Lc=g['Lam']; Lt=g['Lam_t']
    ins=np.mean([(Lt[:,d]>=Lc[:,d].min())&(Lt[:,d]<=Lc[:,d].max()) for d in range(r)],axis=0)
    perdim=100*np.mean(ins==1.0)
    if r==1:
        hull=100*np.mean((Lt[:,0]>=Lc[:,0].min())&(Lt[:,0]<=Lc[:,0].max()))
    else:
        try:
            D=Delaunay(Lc, qhull_options="QJ")
            hull=100*np.mean(D.find_simplex(Lt)>=0)
        except Exception as e:
            hull=np.nan
    mu=Lc.mean(axis=0); S=np.cov(Lc,rowvar=False)+1e-12*np.eye(r)
    Si=np.linalg.inv(np.atleast_2d(S))
    md=lambda A: np.sqrt(np.einsum('ij,jk,ik->i',A-mu,Si,A-mu))
    mc,mt=md(Lc),md(Lt)
    pct=100*np.mean(mc[None,:]<=mt[:,None],axis=1)
    # Xu's other diagnostic: do imputed counterfactuals stay inside the observed range
    lo,hi=Y.min(),Y.max()
    inr=100*np.mean((g['Y0']>=lo)&(g['Y0']<=hi))
    print(f"  {r:2d} {perdim:14.1f}% {hull:14.1f}% {np.median(pct):25.1f}% {inr:23.1f}%")
    rows.append(dict(r=r,perdim=perdim,hull=hull,mahal=np.median(pct),inrange=inr))
    if r==2:
        np.save("a4_load_ctrl.npy",Lc); np.save("a4_load_trt.npy",Lt)
        np.save("a4_factors.npy",g['F']); np.save("a4_xi.npy",g['xi'])
        np.save("a4_trt_country.npy",CTu[TRu])
pd.DataFrame(rows).to_csv("a4_support.csv",index=False)
print("\n  per-dim inside = treated units inside the controls' min-max range on EVERY")
print("  dimension; in convex hull = strict containment in the control loading cloud;")
print("  Mahalanobis pctile = where the median treated unit sits in the control distribution")

g2=G.gsc(Y,X,TRu,T0,2)
print(f"\n{'='*100}\nESTIMATED FACTORS at r = 2 (rescaled by their loading sd)\n{'='*100}")
F=g2['F']; sd=g2['Lam'].std(axis=0)
for d in range(2):
    print(f"  factor {d+1}: "+" ".join(f"{y}:{F[i,d]*sd[d]:+.2f}" for i,y in enumerate(YRS)))
print(f"  additive year effects xi: "+" ".join(f"{y}:{g2['xi'][i]:+.2f}"
      for i,y in enumerate(YRS)))
