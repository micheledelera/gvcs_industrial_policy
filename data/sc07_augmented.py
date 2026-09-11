"""ROUTE B. Augmented synthetic control (Ben-Michael, Feller & Rothstein 2021).

§4f showed the canonical estimator fails where it matters: Mexico's pre-period gap is
+3.94 log points on log US exports, and even the "credible" subsets carry a significantly
positive pre-period gap that non-negative weights summing to one cannot remove. ASCM is
the estimator for exactly that case -- it estimates the matching bias with a ridge outcome
model fitted on the donors and subtracts it, at the cost of admitting negative weights,
i.e. extrapolation.

  1. canonical SC weights w (non-negative, sum one, no intercept) -- as §4e/§4f
  2. ridge outcome model on the DONORS ONLY: regress each donor's post-period outcome on
     its centred pre-period path,  eta = (Xc Xc' + lam I)^-1 Xc y_post,  lam by
     leave-one-out CV over a grid
  3. tau_aug = tau_scm - (X_1 - X_0 w)' eta

The correction term is the pre-period imbalance projected through the outcome model. When
the fit is perfect the imbalance is zero and ASCM collapses to classic SC -- so the SIZE OF
THE CORRECTION is a direct readout of how much extrapolation is being bought. That is
reported alongside every estimate.

Inference: the same in-space placebo, with the augmented estimator applied identically to
each donor. Outcomes: log US exports (where the hull fails hardest), log US exports in
decoupling sectors (§4f's most promising), and log orientation (comparable to §4e).
"""
import pandas as pd, numpy as np, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc05_trimmed.py").read().split("print(f\"\\n{'='*104}\\nDONOR-POOL SIZE SWEEP")[0])
K=20; BASE=[2015,2016,2017]
LAMS=np.array([1e-4,1e-3,1e-2,1e-1,1,10,100,1000])

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
us2=raw[raw['j']==USA]
tot_k=us2.groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
chn_k=us2[us2['i']==CHINA].groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
tot_k.columns=[int(c) for c in tot_k.columns]; chn_k.columns=[int(c) for c in chn_k.columns]
DEC=set((chn_k[BASE].mean(axis=1)/tot_k[BASE].mean(axis=1)*100).pipe(lambda s:s[s>=25]).index)
usdec=us2[us2['ISIC4c'].isin(DEC)].groupby(['i','t'],observed=True)['imports'].sum().unstack('t')
usdec.columns=[int(c) for c in usdec.columns]
del raw,us2; gc.collect()

OUT={'A. log US exports':np.log(usi[YRS]),
     'C. log US exports, decoupling sectors':np.log(usdec.reindex(columns=YRS)),
     'D. log US orientation':Y}

def ridge_eta(Xc, ypost, lam):
    T=Xc.shape[0]
    return np.linalg.solve(Xc@Xc.T+lam*np.eye(T), Xc@ypost)
def pick_lam(Xc, ypost):
    """leave-one-out CV over the donor pool."""
    n=Xc.shape[1]; best=(np.inf,LAMS[0])
    for lam in LAMS:
        err=0.0
        for j in range(n):
            m=np.ones(n,bool); m[j]=False
            e=ridge_eta(Xc[:,m], ypost[m], lam)
            err+=(ypost[j]-Xc[:,j]@e)**2
        if err<best[0]: best=(err,lam)
    return best[1]

def run(lab, Yraw):
    Yo=Yraw.reindex(Y.index)
    good=Yo.notna().all(axis=1) & np.isfinite(Yo).all(axis=1)
    Yo=Yo[good]
    tr=[i for i in TR if i in Yo.index]; dn=[i for i in DN if i in Yo.index]
    if len(tr)<10 or len(dn)<K+5: return None
    def one(i, pool):
        Xp=Yo.loc[pool,PRE].values.T                 # T_pre x K
        x1=Yo.loc[i,PRE].values
        A=np.hstack([Yo.loc[pool,PRE].values, COV_W*Cz.loc[pool].values])
        b=np.concatenate([x1, COV_W*Cz.loc[i].values])
        w=sc(A,b)
        pre_=rms(x1, Xp@w)
        ypost=Yo.loc[pool,POST].values.mean(axis=1)
        mu=Xp.mean(axis=1, keepdims=True); Xc=Xp-mu
        lam=pick_lam(Xc, ypost - ypost.mean())
        eta=ridge_eta(Xc, ypost-ypost.mean(), lam)
        imb = x1 - Xp@w                               # pre-period imbalance
        corr= float(imb@eta)
        tau_scm=float(Yo.loc[i,POST].mean() - ypost@w)
        return w, pre_, tau_scm, tau_scm-corr, corr, lam, float(np.abs(imb).mean())
    # placebo distribution, augmented estimator applied identically
    pl_scm=[]; pl_aug=[]; pl_pre=[]
    for d in dn:
        pool=nearest(d,[x for x in dn if x!=d],K)
        _,p_,ts,ta,c_,_,_=one(d,pool); pl_scm.append(ts); pl_aug.append(ta); pl_pre.append(p_)
    pl_scm=np.abs(np.array(pl_scm)); pl_aug=np.abs(np.array(pl_aug)); pl_pre=np.array(pl_pre)
    rows=[]
    for i in tr:
        pool=nearest(i,dn,K)
        w,p_,ts,ta,c_,lam,imb=one(i,pool)
        rows.append({'i':i,'name':nm(i),'pre':p_,'fit_x':p_/np.median(pl_pre),
                     'tau_scm':ts,'tau_aug':ta,'corr':c_,'lam':lam,'imb':imb,
                     'p_scm':float((1+(pl_scm>=abs(ts)).sum())/(1+len(pl_scm))),
                     'p_aug':float((1+(pl_aug>=abs(ta)).sum())/(1+len(pl_aug))),
                     'val':float(usi.loc[i,BASE].mean()) if i in usi.index else 0.0})
    R=pd.DataFrame(rows); R['credible']=R['fit_x']<=2.0; R['outcome']=lab
    def jk(v):
        v=np.asarray(v); n=len(v)
        if n<3: return np.nan,np.nan
        a=np.array([np.delete(v,j).mean() for j in range(n)])
        return v.mean(), np.sqrt((n-1)/n*((a-a.mean())**2).sum())
    from scipy import stats
    print(f"{'='*112}\n{lab}   {len(tr)} treated, {len(dn)} donors, K={K}\n{'='*112}")
    print(f"  {'country':<15s}{'pre-RMSPE':>11s}{'x plac':>8s}{'tau_scm':>10s}"
          f"{'correction':>12s}{'tau_AUG':>10s}{'p_scm':>8s}{'p_aug':>8s}")
    for _,x in R.sort_values('val',ascending=False).head(12).iterrows():
        s1='*' if x['p_scm']<.10 else ' '; s2='*' if x['p_aug']<.10 else ' '
        print(f"  {x['name']:<15s}{x['pre']:>11.3f}{x['fit_x']:>8.1f}{x['tau_scm']:>+10.3f}"
              f"{x['corr']:>+12.3f}{x['tau_aug']:>+10.3f}{x['p_scm']:>8.3f}{s1}{x['p_aug']:>7.3f}{s2}")
    for slab,sub in [("ALL treated",R),("CREDIBLE-FIT subset",R[R['credible']])]:
        if len(sub)<3: continue
        a=jk(sub['tau_scm'].values); b=jk(sub['tau_aug'].values)
        fa=-2*np.log(sub['p_aug'].clip(1e-6)).sum()
        print(f"\n  {slab} (n={len(sub)})")
        print(f"    classic  tau {a[0]:>+8.4f} ({a[1]:.4f}) t={a[0]/a[1]:>+5.2f}   "
              f"median p {sub['p_scm'].median():.3f}")
        print(f"    AUGMENTED tau {b[0]:>+8.4f} ({b[1]:.4f}) t={b[0]/b[1]:>+5.2f}   "
              f"median p {sub['p_aug'].median():.3f}   "
              f"p<0.10 {int((sub['p_aug']<0.10).sum())}/{len(sub)}   "
              f"Fisher p={1-stats.chi2.cdf(fa,2*len(sub)):.4f}")
        print(f"    extrapolation: median |correction| {sub['corr'].abs().median():.3f}"
              f"  = {100*sub['corr'].abs().median()/max(abs(a[0]),1e-9):.0f}% of the classic estimate"
              f"   median pre-period imbalance {sub['imb'].median():.3f}")
    print()
    return R

res=[r for r in (run(l,v) for l,v in OUT.items()) if r is not None]
pd.concat(res).to_csv("sc07_augmented.csv",index=False)
log("DONE")
