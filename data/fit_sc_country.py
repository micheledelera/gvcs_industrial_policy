"""Synthetic control, done properly: per treated unit, no free intercept, richer covariates.

§3af showed the §3aa-§3ae estimator is not really a synthetic control. Its weights are
near-uniform (median effective donors 13.4 of 21) because it matches a GROUP MEAN with a
FREE INTERCEPT: a smooth target, and the level mismatch is absorbed for nothing. That is
why Turkmenistan and Suriname sit in the same donor pool as Vietnam and Turkiye.

Three fixes, which together are the honest version of the design:

  1. PER-UNIT MATCHING. One synthetic control per treated pair (i,k) against that
     sector's IP=0 donors, then aggregate. Abadie's machinery is built for one treated
     unit; averaging 16 pairs first destroys the variation that makes weights informative.
  2. NO FREE INTERCEPT. The donor combination must match the LEVEL as well as the shape.
     This is what makes "only a truly similar unit could track this closely" true, and
     it forces donors of comparable export size.
  3. RICHER COVARIATES. log MVA per capita, MVA/GDP, ECI, log total exports, and the
     sector's share of the country's own exports -- the last being the specialisation
     margin Juhasz et al. say governments select on.

And a second specification that changes the QUESTION rather than the estimator:

  4. POLICY-ACTIVE DONORS ONLY. §3ac found 69.5% of donors are countries doing no policy
     at all, which makes the contrast country-level rather than sector-targeting.
     Restricting donors to policy-active countries that did not target THIS sector makes
     it a genuine targeting test, at the cost of ~a third of the donor pool.

Reported for each: pre-period fit, effective donors, tau, and the placebo on
(2015-17)-(2010-12). The placebo is the test that matters.
"""
import pandas as pd, numpy as np, gc, sys, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("fit_sc_mvasweep2.py").read().split("def run(mw):")[0])
C2ISO={699:356,757:756,579:578,251:250,842:840,381:380,490:158,729:729}
XLSX=("/root/.claude/uploads/618110ef-d894-58be-8e00-e7f2bf548b03/"
      "4ad6510b-0154963EF4289D02DEA410D8D38E4E6A8C6D.xlsx")

# ---- covariates ----
raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['t']=raw['t'].astype('int32')
raw['share_frac_policies']=pd.to_numeric(raw['share_frac_policies'],errors='coerce').fillna(0)
pre=raw[raw['t'].isin(PRE)]
X=(pre.groupby(['i','ISIC4c'],observed=True)['imports'].sum()/3).unstack(fill_value=0.0)
anyp=(pre.groupby('i')['share_frac_policies'].max()>0)
del raw,pre; gc.collect()
shm=X.div(X.sum(axis=1).replace(0,np.nan),axis=0); world=X.sum(axis=0)/X.sum().sum()
M_=((shm.div(world,axis=1))>=1).astype(float); M_=M_.loc[M_.sum(axis=1)>0,M_.sum(axis=0)>0]
kc,kp=M_.sum(axis=1),M_.sum(axis=0)
vals,vecs=np.linalg.eig(((M_.div(kc,axis=0))@(M_.div(kp,axis=1).T)).values)
K=np.real(vecs[:,np.argsort(-np.real(vals))[1]])
ECI=pd.Series((K-K.mean())/K.std(),index=M_.index)
if ECI.corr(pd.Series(kc,index=M_.index))<0: ECI=-ECI
u=pd.read_excel(XLSX,sheet_name='Data'); u=u[u['Year'].isin(PRE)]
U=u.pivot_table(index='CountryCode',columns='VariableCode',values='Value',aggfunc='mean')

mm=m.copy()
ctry=sorted({i for i,_ in mm.index}); iso=[C2ISO.get(c,c) for c in ctry]
mm['mva_pc']=[np.log(U['NV_IND_MANFPC'].reindex(iso).values[ctry.index(i)]) for i,_ in mm.index]
mm['eci']=[ECI.get(i,np.nan) for i,_ in mm.index]
mm['lexp']=[np.log(X.sum(axis=1).get(i,np.nan)) for i,_ in mm.index]
mm['spec']=[(X.loc[i,k]/X.loc[i].sum() if i in X.index and X.loc[i].sum()>0 else np.nan)
            for i,k in mm.index]
mm['anyp']=[bool(anyp.get(i,False)) for i,_ in mm.index]
COV=['mvaz','mva_pc','eci','lexp','spec']
for c in COV[1:]:
    mm[c]=(mm[c]-mm[c].mean())/mm[c].std()
mm=mm.dropna(subset=COV)
Yv2=Yv.loc[mm.index]
log(f"pairs {len(mm):,} | treated {int(mm['T'].sum()):,} | "
    f"sectors {mm.index.get_level_values('ISIC4c').nunique()}")

def solve_noint(A,b,zeta2,iters=1200):
    """min ||A'w - b||^2 + zeta2||w||^2 on the simplex. No intercept: the donor
    combination must match the LEVEL as well as the shape."""
    n=A.shape[0]; w=np.ones(n)/n
    L=np.linalg.norm(A,2)**2/max(A.shape[1],1)+zeta2+1e-12
    for _ in range(iters):
        g=A@(A.T@w-b)*2.0/max(A.shape[1],1)+2.0*zeta2*w
        w=proj_simplex(w-g/L)
    return w

def run(per_unit, intercept, covs, donors_policy_only, label, cw=1.0, zs=1e-6,
        country_treat=False):
    """country_treat=True redefines treatment at the COUNTRY level: every sector of a
    policy-using country is treated, donors are countries with no recorded policy
    anywhere. That is the policy-users-vs-non-users comparison -- the actual research
    question -- rather than 'did this country target this sector', which is a
    within-policy-user targeting question."""
    dp=(Yv2[POST].mean(axis=1)-Yv2[PRE].mean(axis=1))
    dl=(Yv2[PRE].mean(axis=1)-Yv2[PRE0].mean(axis=1))
    taus, placs, rmses, effs, ndon = [], [], [], [], []
    for k,g in mm.groupby(level='ISIC4c', observed=True):
        if country_treat:
            t_=g[g['anyp']]; d_=g[~g['anyp']]
        else:
            t_=g[g['T']==1]; d_=g[g['T']==0]
            if donors_policy_only: d_=d_[d_['anyp']]
        if len(t_)==0 or len(d_)<MIN_DON: continue
        P=Yv2.loc[d_.index,PATH].values
        Cd=(cw*d_[covs].values.T) if covs else np.zeros((0,len(d_)))
        sig=np.diff(P,axis=1).std()
        targets = [t_] if not per_unit else [t_.iloc[[j]] for j in range(len(t_))]
        for tt in targets:
            bt=Yv2.loc[tt.index,PATH].values.mean(axis=0)
            bc=(cw*tt[covs].values.mean(axis=0)) if covs else np.zeros(0)
            A=np.hstack([P,Cd.T]); b=np.concatenate([bt,bc])
            z2=(((len(tt)*len(POST))**0.5)*(sig**2))*zs+1e-10
            w = solve_noint(A,b,z2) if not intercept else solve_w2(A,b,z2,len(PATH))[0]
            fit=P.T@w
            off = 0.0 if not intercept else float(np.mean(bt-fit))
            rmses.append(float(np.sqrt(((fit+off-bt)**2).mean())))
            effs.append(1.0/(w**2).sum()); ndon.append(len(d_))
            taus.append(dp[tt.index].mean()-float(w@dp[d_.index].values))
            placs.append(dl[tt.index].mean()-float(w@dl[d_.index].values))
    if len(taus)<5: print(f"{label}: too few"); return None
    taus=np.array(taus); placs=np.array(placs)
    def jk(v):
        n=len(v); a=np.array([np.delete(v,i).mean() for i in range(n)])
        return v.mean(), np.sqrt((n-1)/n*((a-a.mean())**2).sum())
    a=jk(taus); b_=jk(placs)
    print(f"{label:<46s}{len(taus):>6d}{np.median(rmses):>9.3f}"
          f"{np.median(effs):>7.1f}/{int(np.median(ndon)):<4d}"
          f"{a[0]:>+9.4f} ({a[1]:.4f}) t={a[0]/a[1]:>+5.2f}"
          f"{b_[0]:>+9.4f} ({b_[1]:.4f}) t={b_[0]/b_[1]:>+5.2f}")
    return {'spec':label,'n':len(taus),'rmse':np.median(rmses),'eff':np.median(effs),
            'tau':a[0],'se':a[1],'plac':b_[0],'plac_se':b_[1]}

print(f"\n{'spec':<46s}{'n':>6s}{'RMSE':>9s}{'eff/don':>12s}{'':>4s}{'tau':>18s}{'':>10s}{'placebo':>14s}")
print("="*128)
rows=[]
for args in [
  (False,True, ['mvaz'],                False,"0. as §3aa: group, intercept, MVA only"),
  (False,True, COV,                     False,"1. + richer covariates"),
  (False,False,COV,                     False,"2. + no intercept"),
  (True, False,COV,                     False,"3. + PER-UNIT   <- the honest version"),
  (True, True, COV,                     False,"3b. per-unit but keeping the intercept"),
  (True, False,COV,                     True, "4. per-unit, POLICY-ACTIVE donors only"),
]:
    r=run(*args)
    if r: rows.append(r)
print("-"*128)
print("COUNTRY-LEVEL TREATMENT: every sector of a policy-using country is treated;")
print("donors are countries with NO recorded industrial policy in any sector.")
print("-"*128)
for args in [
  (False,True, ['mvaz'],  False,"C0. group, intercept, MVA only"),
  (False,False,COV,       False,"C1. group, no intercept, rich covariates"),
  (True, False,COV,       False,"C2. PER-UNIT, no intercept, rich covariates"),
  (True, True, COV,       False,"C3. per-unit, intercept kept"),
]:
    r=run(*args, country_treat=True)
    if r: rows.append(r)
for args in []:
    r=run(*args)
    if r: rows.append(r)
pd.DataFrame(rows).to_csv("sc_country_treat.csv",index=False)
log("DONE")
