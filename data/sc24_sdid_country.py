"""§8d. Two fixes to §8c: identify omega by trimming, and permute at the COUNTRY level.

§8c found omega uniform at every zeta, including zeta = 0 (effective n 2,429 of 2,497).
So it is NOT the regularisation: with 2,497 donors fitting an 11-period pre-path the unit
weights are massively underdetermined -- §4d's degeneracy -- and the solver returns the
near-uniform solution its initialisation favours. tau_sdid is then just the lambda-weighted
DiD, and the unit weights do no work. §5b trimmed to K = 20 donors for exactly this reason.

Fix 1  sweep the donor pool: controls inside the §5b bands (Chinese share within 10pp and
       log size within 2.0 of the TREATED GROUP mean), then the K nearest on MVA/GDP, log
       MVA per capita, ECI and export share. K in {20, 50, 100, 250, all}.

Fix 2  §8c's randomisation permuted the treatment across units. Real treatment is
       country-clustered (25 countries, largest 11% of treated units), so a permutation that
       scatters treated units across 139 countries destroys exactly the clustering that
       generates the correlation -- it is anti-conservative against a country-level
       confound, which is the margin every design in §5-§8 has died on. So: permute whole
       COUNTRIES into treatment until the treated unit count matches.
"""
import pandas as pd, numpy as np, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
rng=np.random.default_rng(23)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])
PREY9=list(range(2009,2018))
raw2=pd.read_pickle("agg_for_estimation.pkl")
sp=raw2[['i','ISIC4c','t','n_policies']].drop_duplicates(subset=['i','ISIC4c','t'])
sp['n_policies']=pd.to_numeric(sp['n_policies'],errors='coerce').fillna(0.0)
sp['t']=sp['t'].astype(int); del raw2
P=(sp[sp['t'].isin(PREY9)].pivot_table(index=['i','ISIC4c'],columns='t',
   values='n_policies',aggfunc='sum').reindex(M.index).fillna(0.0))
yrs=(P>0).sum(axis=1).values
TRT0=(yrs>=6); NEV=(yrs==0); keep=TRT0|NEV
Y=np.hstack([Lp,Lq])[keep]; W0=TRT0[keep]
ct=CT[keep]; se=SE[keep]; Zk=Z[keep]; Dk=Dv[keep]; Sk=Sv[keep]
NP,NQ=len(PRE),len(POST)
log(f"panel {Y.shape}: {int(W0.sum()):,} treated / {int((~W0).sum()):,} controls")

def proj(v):
    w=np.sort(v)[::-1]; c=np.cumsum(w)-1.0; r=np.arange(1,len(v)+1)
    m=w-c/r>0
    if not m.any(): return np.ones_like(v)/len(v)
    return np.maximum(v-c[m][-1]/r[m][-1],0.0)
def simplex_ls(A,b,lam2,it=2000):
    n=A.shape[1]; w=np.ones(n)/n; L=np.linalg.norm(A,2)**2+lam2+1e-12
    for _ in range(it): w=proj(w-(A.T@(A@w-b)+lam2*w)/L)
    return w

def donors(W, K):
    """control indices inside the §5b bands around the treated group mean, K nearest"""
    co=np.where(~W)[0]
    if K is None: return co
    dbar,sbar,zbar=Dk[W].mean(),Sk[W].mean(),Zk[W].mean(axis=0)
    m=(np.abs(Dk[co]-dbar)<=DECB)&(np.abs(Sk[co]-sbar)<=SZB)
    cand=co[m] if m.sum()>=K else co
    return cand[np.argsort(np.sqrt(((Zk[cand]-zbar)**2).sum(axis=1)))[:K]]

def sdid(W, K, zmult=1.0):
    co=donors(W,K); Yc,Yt=Y[co],Y[W]
    tr=Yt[:,:NP].mean(axis=0)
    sig=np.diff(Yc[:,:NP],axis=1).std(ddof=1)
    zeta=zmult*((W.sum()*NQ)**0.25)*sig
    A=Yc[:,:NP].T; om=simplex_ls(A-A.mean(axis=0,keepdims=True),tr-tr.mean(),(zeta**2)*NP)
    B=Yc[:,:NP]; c=Yc[:,NP:].mean(axis=1)
    lam=simplex_ls(B-B.mean(axis=1,keepdims=True),c-c.mean(),((1e-6*sig)**2)*len(Yc))
    # tau in closed form: the lambda-weighted, omega-weighted double difference
    tr_post=Yt[:,NP:].mean(); tr_pre=float(tr@lam)
    sy_post=float((Yc[:,NP:].mean(axis=1))@om); sy_pre=float((Yc[:,:NP].T@om)@lam)
    return (tr_post-tr_pre)-(sy_post-sy_pre), om, lam, co

print(f"\n{'='*100}\nDONOR-POOL SWEEP: does omega do any work once the pool is trimmed?"
      f"\n{'='*100}")
print(f"  {'K':>6s} {'tau_sdid':>10s} {'eff n':>7s} {'max w':>8s} {'pre-fit RMSE':>13s}")
ROWS=[]
for K in [20,50,100,250,None]:
    tt,om,lam,co=sdid(W0,K)
    tr=Y[W0][:,:NP].mean(axis=0); fit=Y[co][:,:NP].T@om
    rm=float(np.sqrt((((tr-tr.mean())-(fit-fit.mean()))**2).mean()))
    print(f"  {str(K) if K else 'all':>6s} {tt:>+10.4f} {1/(om**2).sum():7.0f} "
          f"{om.max():8.4f} {rm:13.4f}")
    ROWS.append(dict(K=K if K else 0,tau=tt,effn=1/(om**2).sum(),maxw=om.max(),rmse=rm))
pd.DataFrame(ROWS).to_csv("sc24_ksweep.csv",index=False)

KSEL=20
tau,om,lam,co=sdid(W0,KSEL)
print(f"\n  -> K = {KSEL}: tau_sdid = {tau:+.4f}")
dn=pd.Series(om,index=[f"{nm(ct[j])} {se[j]}" for j in co]).sort_values(ascending=False)
print("  donors with weight >= 1%:")
for k,v in dn[dn>=0.01].items(): print(f"    {k:34s} {100*v:5.1f}%")

# ---- country-level randomisation ----
ucty,ucnt=np.unique(ct,return_counts=True)
target=int(W0.sum())
def perm_country():
    order=rng.permutation(len(ucty)); cum=0; picked=[]
    for j in order:
        picked.append(ucty[j]); cum+=ucnt[j]
        if cum>=target: break
    return np.isin(ct,picked)
print(f"\n{'='*100}\nRANDOMISATION AT THE COUNTRY LEVEL ({len(ucty)} countries, "
      f"target {target} treated units)\n{'='*100}")
for K in [KSEL,None]:
    Pv=[]
    for r in range(300):
        Wp=perm_country()
        if Wp.sum()<50 or (~Wp).sum()<50: continue
        try:
            v=sdid(Wp,K)[0]
            if np.isfinite(v): Pv.append(v)
        except Exception: pass
    Pv=np.array(Pv); tk=sdid(W0,K)[0]
    p=float((1+(np.abs(Pv)>=abs(tk)).sum())/(1+len(Pv)))
    print(f"  K={str(K) if K else 'all':>4s}  tau {tk:+.4f}   placebo n={len(Pv)} "
          f"mean {Pv.mean():+.4f} sd {Pv.std():.4f}   two-sided p = {p:.4f}")
    np.save(f"sc24_permctry_{K if K else 0}.npy",Pv)
    log(f"K={K} country permutations done")
pd.DataFrame({'year':YRS,'treated':Y[W0].mean(axis=0),
              'synthetic':Y[co].T@om,'lambda':list(lam)+[np.nan]*NQ}
            ).to_csv("sc24_paths.csv",index=False)
log("DONE")
