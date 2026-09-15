"""§8c. Synthetic difference-in-differences on the §8b persistence treatment.

Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021). The estimator §8b's design calls
for: it keeps the DiD differencing (so a level gap between groups is removed, unlike
Abadie) and adds BOTH unit weights and TIME weights, so it does not assume parallel
pre-trends -- it reweights pre-periods toward those that predict the post-period. §8b
documented a parallel-trends violation (joint p = 0.012-0.036), which is the case this is
built for; §3x/§3y failed with a fuzzy treatment, and a sharp binary with 2,497 donors is
the condition it needs.

  treated   any intervention in >= 6 of 2009-2017        1,083 units, 25 countries
  controls  no intervention in any year 2009-2017        2,497 units
  outcome   ln US imports, 2007-2024, balanced
  omega     control unit weights, simplex + free intercept, ridge zeta as in ADHIW eq (5)
  lambda    pre-period time weights, simplex + free intercept, fitted on CONTROLS to
            predict their own post-period mean
  tau       weighted two-way least squares with weights omega_i * lambda_t

Inference, deliberately not the fit (§5i's lesson -- a flat SC pre-period is a property of
the weights, not evidence):
  jackknife over countries, as elsewhere in this file
  randomisation inference, 500 permutations of the treatment vector STRATIFIED BY SECTOR,
    so each sector keeps its number of treated units and the permutation randomises the
    country dimension -- the margin every design in §5-§8 has died on
  the same, unstratified, as a second version
"""
import pandas as pd, numpy as np, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
rng=np.random.default_rng(11)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])
PREY9=list(range(2009,2018))
raw2=pd.read_pickle("agg_for_estimation.pkl")
sp=raw2[['i','ISIC4c','t','n_policies']].drop_duplicates(subset=['i','ISIC4c','t'])
sp['n_policies']=pd.to_numeric(sp['n_policies'],errors='coerce').fillna(0.0)
sp['t']=sp['t'].astype(int); del raw2
P=(sp[sp['t'].isin(PREY9)].pivot_table(index=['i','ISIC4c'],columns='t',
   values='n_policies',aggfunc='sum').reindex(M.index).fillna(0.0))
yrs=(P>0).sum(axis=1).values
TRT=(yrs>=6); NEV=(yrs==0); keep=TRT|NEV
Y=np.hstack([Lp,Lq])[keep]; W=TRT[keep]; ct=CT[keep]; se=SE[keep]
NP,NQ=len(PRE),len(POST)
log(f"panel {Y.shape}: {int(W.sum()):,} treated, {int((~W).sum()):,} controls, "
    f"{len(set(ct))} countries")

def proj(v):
    w=np.sort(v)[::-1]; c=np.cumsum(w)-1.0; r=np.arange(1,len(v)+1)
    m=w-c/r>0
    if not m.any(): return np.ones_like(v)/len(v)
    return np.maximum(v-c[m][-1]/r[m][-1],0.0)
def simplex_ls(A,b,lam2,it=4000):
    """min ||A w - b||^2 + lam2 ||w||^2 over the simplex; A already intercept-demeaned"""
    n=A.shape[1]; w=np.ones(n)/n
    L=np.linalg.norm(A,2)**2+lam2+1e-12
    for _ in range(it):
        g=A.T@(A@w-b)+lam2*w
        w=proj(w-g/L)
    return w

def sdid(Y,W,zmult=1.0):
    """returns tau, omega (controls), lambda (pre-periods), and the two group paths.
    zmult scales ADHIW's zeta: with N_tr in the thousands the published formula
    (N_tr*T_post)^(1/4)*sigma drives omega to the uniform simplex, so the dependence on it
    has to be shown rather than assumed."""
    Yc,Yt=Y[~W],Y[W]
    tr_pre=Yt[:,:NP].mean(axis=0)
    # zeta as in ADHIW: (N_tr * T_post)^(1/4) * sd of control first differences in pre
    dif=np.diff(Yc[:,:NP],axis=1)
    sig=dif.std(ddof=1)
    zeta=zmult*((W.sum()*NQ)**0.25)*sig
    # omega: match the treated pre-period path, free intercept -> demean over pre-periods
    A=Yc[:,:NP].T                                   # T_pre x N_co
    Ad=A-A.mean(axis=0,keepdims=True); bd=tr_pre-tr_pre.mean()
    om=simplex_ls(Ad,bd,(zeta**2)*NP)
    # lambda: controls' pre-periods predicting their own post mean, free intercept
    B=Yc[:,:NP]                                     # N_co x T_pre
    c=Yc[:,NP:].mean(axis=1)
    Bd=B-B.mean(axis=1,keepdims=True); cd=c-c.mean()
    zl=1e-6*sig
    lam=simplex_ls(Bd,cd,(zl**2)*len(Yc))
    # tau: weighted two-way least squares, weights omega_i * lambda_t
    wi=np.empty(len(Y)); wi[~W]=om; wi[W]=1.0/W.sum()
    wt=np.empty(Y.shape[1]); wt[:NP]=lam; wt[NP:]=1.0/NQ
    Wt=np.outer(wi,wt)
    D=np.zeros_like(Y); D[np.ix_(W,np.arange(NP,NP+NQ))]=1.0
    y,x=Y.copy(),D.copy()
    rs,cs_=Wt.sum(axis=1),Wt.sum(axis=0)
    rok,cok=rs>0,cs_>0                              # zero-weight rows/years are excluded
    for _ in range(1000):                           # weighted alternating projections
        mx=0.0
        for arr in (y,x):
            rm=np.zeros(len(rs))
            rm[rok]=(Wt[rok]*arr[rok]).sum(axis=1)/rs[rok]; arr-=rm[:,None]
            cm=np.zeros(len(cs_))
            cm[cok]=(Wt[:,cok]*arr[:,cok]).sum(axis=0)/cs_[cok]; arr-=cm[None,:]
            mx=max(mx,np.abs(rm).max(),np.abs(cm).max())
        if mx<1e-10: break
    dd=(Wt*x*x).sum()
    tau=float((Wt*x*y).sum()/dd) if dd>0 else np.nan
    return tau,om,lam,Yt.mean(axis=0),(Yc.T@om)

print(f"\n{'='*100}\nSYNTHETIC DiD -- dependence on ADHIW's zeta\n{'='*100}")
print(f"  {'zeta':>12s} {'tau_sdid':>10s} {'donors>0':>9s} {'eff n':>7s} {'max w':>8s}")
SW={}
for zm in [0.0,0.01,0.1,0.5,1.0]:
    tt,oo,ll,_,_=sdid(Y,W,zm)
    SW[zm]=(tt,oo,ll)
    print(f"  {zm:10.2f}x  {tt:>+10.4f} {int((oo>1e-6).sum()):9d} "
          f"{1/(oo**2).sum():7.0f} {oo.max():8.4f}")
ZM=0.1
tau,om,lam,pt,ps=sdid(Y,W,ZM)
print(f"\n  -> reporting zeta = {ZM}x ADHIW, the largest value at which omega is not "
      f"driven to the uniform simplex")
print(f"  tau_sdid = {tau:+.4f} log points")
eff=1/ (om**2).sum()
print(f"  omega: {int((om>1e-6).sum()):,} donors with positive weight of {int((~W).sum()):,}; "
      f"effective n {eff:.0f}; largest weight {om.max():.4f}; "
      f"top 10 donors hold {100*np.sort(om)[::-1][:10].sum():.1f}%")
dc=pd.Series(om,index=[nm(c) for c in ct[~W]]).groupby(level=0).sum().sort_values(ascending=False)
print(f"  omega by donor country: "+",  ".join(f"{k} {100*v:.1f}%" for k,v in dc.head(8).items()))
print(f"\n  lambda (time weights on the pre-period):")
print("    "+"  ".join(f"{y}:{w:.3f}" for y,w in zip(PRE,lam)))
print(f"    years carrying weight: {[y for y,w in zip(PRE,lam) if w>0.01]}")
pre_gap=float((pt[:NP]*lam).sum()-(ps[:NP]*lam).sum())
post_gap=float(pt[NP:].mean()-ps[NP:].mean())
print(f"\n  lambda-weighted pre-period gap {pre_gap:+.4f};  post-period gap {post_gap:+.4f};"
      f"  difference {post_gap-pre_gap:+.4f}")

# ---- comparison estimators on the identical sample ----
def did(Y,W):
    a=Y[W][:,NP:].mean()-Y[W][:,:NP].mean(); b=Y[~W][:,NP:].mean()-Y[~W][:,:NP].mean()
    return a-b
def sc_only(Y,W):
    Yc,Yt=Y[~W],Y[W]; tr=Yt[:,:NP].mean(axis=0)
    A=Yc[:,:NP].T
    w=simplex_ls(A,tr,0.0)
    return float(Yt[:,NP:].mean()-(Yc[:,NP:].T@w).mean()), w
sc_tau,wsc=sc_only(Y,W)
print(f"\n  for comparison, same sample: plain DiD {did(Y,W):+.4f}   "
      f"Abadie-style SC (no time weights, no intercept) {sc_tau:+.4f}")

# ---- inference ----
cs=np.unique(ct[W])
jk=np.array([sdid(Y[ct!=c],W[ct!=c],ZM)[0] for c in cs])
jk=jk[np.isfinite(jk)]
se_jk=np.sqrt((len(cs)-1)/len(cs)*((jk-jk.mean())**2).sum())
log(f"jackknife over {len(cs)} treated countries done")
print(f"\n{'='*100}\nINFERENCE\n{'='*100}")
print(f"  jackknife over countries: se {se_jk:.4f}  t = {tau/se_jk:+.2f}")

NPERM=300
def perm_strat(W,se_):
    out=np.zeros(len(W),bool)
    for s in np.unique(se_):
        idx=np.where(se_==s)[0]; k=int(W[idx].sum())
        if k: out[rng.choice(idx,size=k,replace=False)]=True
    return out
for lab,fn in [("stratified by sector",lambda: perm_strat(W,se)),
               ("unstratified",lambda: np.isin(np.arange(len(W)),
                    rng.choice(len(W),size=int(W.sum()),replace=False)))]:
    P_=[]
    for r in range(NPERM):
        Wp=fn()
        if Wp.sum()==0 or (~Wp).sum()<20: continue
        try:
            v=sdid(Y,Wp,ZM)[0]
            if np.isfinite(v): P_.append(v)
        except Exception: pass
    P_=np.array(P_)
    assert np.isfinite(P_).all() and np.isfinite(tau), "NaN in the placebo comparison"
    p2=float((1+(np.abs(P_)>=abs(tau)).sum())/(1+len(P_)))
    print(f"  randomisation, {lab:22s} n={len(P_)}  placebo mean {P_.mean():+.4f} "
          f"sd {P_.std():.4f}   two-sided p = {p2:.4f}")
    np.save(f"sc23_perm_{lab.split()[0]}.npy",P_)
    log(f"{lab} permutations done")

pd.DataFrame({'year':YRS,'treated':pt,'synthetic':ps,
              'lambda':list(lam)+[np.nan]*NQ}).to_csv("sc23_sdid_paths.csv",index=False)
pd.DataFrame({'omega':om,'country':[nm(c) for c in ct[~W]],
              'sector':[str(x) for x in se[~W]]}).to_csv("sc23_omega.csv",index=False)
log("DONE")
