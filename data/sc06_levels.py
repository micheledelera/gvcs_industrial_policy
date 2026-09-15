"""ROUTE A with NON-scale-free outcomes. Same canonical machinery as §4e, magnitude
outcomes, and pre-RMSPE as the filter.

§4a showed the big exporters lie outside the donor hull on levels, and trimming the pool
can only shrink the hull. So this is not an escape from that problem -- it is the
alternative Abadie himself uses: run it, report the fit, and let fit decide who gets an
answer. Abadie, Diamond and Hainmueller drop placebo units whose pre-RMSPE exceeds twice
the treated unit's; the analogue for choosing which TREATED units to trust is to compare
each one's pre-RMSPE with the placebo distribution's.

Outcomes (all magnitude, none scale-free):
  A. log US exports                    the natural magnitude
  B. log share of total US imports     "did they capture US market share"
  C. log US exports, decoupling sectors only (China >=25% of the US market, 2015-17)

Machinery identical to §4e: donors ranked by covariate distance (log total exports,
MVA/GDP, log MVA per capita, ECI), K nearest kept, matched on eleven lagged outcomes plus
those covariates, non-negative weights summing to one, no intercept. In-space placebo from
121 donors each given their own K nearest.

REPORTED: per-country pre-RMSPE against the placebo distribution's, the aggregate on all
countries, and the aggregate on the subset whose fit is credible.
"""
import pandas as pd, numpy as np, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc05_trimmed.py").read().split("print(f\"\\n{'='*104}\\nDONOR-POOL SIZE SWEEP")[0])
K=20
BASE=[2015,2016,2017]

# rebuild the three magnitude outcomes on the same country set
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

OUT={}
OUT['A. log US exports']       = np.log(usi[YRS])
OUT['B. log share of US mkt']  = np.log(usi.div(usmkt,axis=1)[YRS])
OUT['C. log US exports, decoupling sectors'] = np.log(usdec.reindex(columns=YRS))
print(f"decoupling sectors: {len(DEC)} of 125\n")

def run_outcome(lab, Yraw):
    Yo=Yraw.reindex(Y.index)
    good=Yo.notna().all(axis=1) & (np.isfinite(Yo).all(axis=1))
    Yo=Yo[good]
    tr=[i for i in TR if i in Yo.index]; dn=[i for i in DN if i in Yo.index]
    if len(tr)<10 or len(dn)<K+5: print(f"{lab}: too few"); return None
    def fit(i,pool):
        A=np.hstack([Yo.loc[pool,PRE].values, COV_W*Cz.loc[pool].values])
        b=np.concatenate([Yo.loc[i,PRE].values, COV_W*Cz.loc[i].values])
        w=sc(A,b)
        return (w, rms(Yo.loc[i,PRE].values, Yo.loc[pool,PRE].values.T@w),
                   rms(Yo.loc[i,POST].values, Yo.loc[pool,POST].values.T@w))
    pl_ratio=[]; pl_pre=[]
    for d in dn:
        pool=nearest(d,[x for x in dn if x!=d],K)
        _,p_,q_=fit(d,pool); pl_ratio.append(q_/max(p_,1e-8)); pl_pre.append(p_)
    pl_ratio=np.array(pl_ratio); pl_pre=np.array(pl_pre)
    rows=[]
    for i in tr:
        pool=nearest(i,dn,K)
        w,p_,q_=fit(i,pool); ratio=q_/max(p_,1e-8)
        rows.append({'i':i,'name':nm(i),'pre':p_,'post':q_,'ratio':ratio,
                     'p':float((1+(pl_ratio>=ratio).sum())/(1+len(pl_ratio))),
                     'gap_pre':float((Yo.loc[i,PRE]-(Yo.loc[pool,PRE].values.T@w)).mean()),
                     'gap_post':float((Yo.loc[i,POST]-(Yo.loc[pool,POST].values.T@w)).mean()),
                     'n_pos':int((w>0.01).sum()),
                     'val':float(usi.loc[i,BASE].mean()) if i in usi.index else 0.0})
    R=pd.DataFrame(rows); R['fit_ratio']=R['pre']/np.median(pl_pre)
    R['credible']=R['fit_ratio']<=2.0
    def jk(v):
        v=np.asarray(v); n=len(v)
        if n<3: return np.nan,np.nan
        a=np.array([np.delete(v,j).mean() for j in range(n)])
        return v.mean(), np.sqrt((n-1)/n*((a-a.mean())**2).sum())
    from scipy import stats
    print(f"{'='*106}\n{lab}   {len(tr)} treated, {len(dn)} donors, K={K}\n{'='*106}")
    print(f"  median placebo pre-RMSPE {np.median(pl_pre):.4f}   "
          f"median treated pre-RMSPE {R['pre'].median():.4f}")
    print(f"  treated units with credible fit (pre-RMSPE <= 2x median placebo): "
          f"{int(R['credible'].sum())}/{len(R)}  "
          f"({100*R.loc[R['credible'],'val'].sum()/R['val'].sum():.1f}% of treated US trade value)")
    print(f"\n  {'country':<15s}{'pre-RMSPE':>11s}{'x placebo':>11s}{'credible':>10s}"
          f"{'post/pre':>10s}{'p':>8s}{'gap pre':>9s}{'gap post':>10s}")
    for _,x in R.sort_values('val',ascending=False).head(14).iterrows():
        st='*' if x['p']<0.10 else ' '
        print(f"  {x['name']:<15s}{x['pre']:>11.4f}{x['fit_ratio']:>11.2f}"
              f"{('yes' if x['credible'] else 'NO'):>10s}{x['ratio']:>10.2f}{x['p']:>8.3f}{st}"
              f"{x['gap_pre']:>+9.3f}{x['gap_post']:>+10.3f}")
    for slab,sub in [("ALL treated",R),("CREDIBLE-FIT subset",R[R['credible']])]:
        if len(sub)<3: continue
        a=jk(sub['gap_pre'].values); b=jk(sub['gap_post'].values)
        fi=-2*np.log(sub['p'].clip(1e-6)).sum()
        print(f"\n  {slab}  (n={len(sub)})")
        print(f"    gap pre  {a[0]:>+8.4f} ({a[1]:.4f}) t={a[0]/a[1]:>+5.2f}    "
              f"gap post {b[0]:>+8.4f} ({b[1]:.4f}) t={b[0]/b[1]:>+5.2f}")
        print(f"    placebo p: median {sub['p'].median():.3f}   "
              f"p<0.10 {int((sub['p']<0.10).sum())}/{len(sub)} ({100*(sub['p']<0.10).mean():.0f}%, null 10%)"
              f"   Fisher chi2({2*len(sub)})={fi:.1f}, p={1-stats.chi2.cdf(fi,2*len(sub)):.4f}")
    print()
    R['outcome']=lab
    return R

res=[r for r in (run_outcome(l,Yv_) for l,Yv_ in OUT.items()) if r is not None]
pd.concat(res).to_csv("sc06_levels.csv",index=False)
log("DONE")
