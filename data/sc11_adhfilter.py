"""§5c. The ADH placebo fit filter applied to the §5b design.

§5b found median pre-RMSPE of 0.45-0.53 for placebo units against 0.27-0.33 for treated
ones. Placebos fit ~1.6x worse, so their post-period deviations are mechanically larger,
the placebo distribution is too wide, and the randomization test is under-powered against
its own comparison set.

Abadie, Diamond & Hainmueller handle exactly this: "they've dropped any state unit from the
graph whose pretreatment RMSPE is more than two times that of California's." Applied here
PER TREATED UNIT -- each treated unit has its own pre-RMSPE, hence its own admissible
placebo set:

    p_i = (1 + #{admissible placebos with statistic >= unit i's}) / (1 + #admissible)
    admissible: pre-RMSPE(placebo) <= CUT x pre-RMSPE(unit i)

The cutoff is SWEPT (1x, 2x, 3x, 5x, no filter) rather than chosen, since a single cutoff
is exactly the kind of specification search Ferman, Pinto & Possebom warn produces false
positives. ADH's headline is 2x. Also reported: how many placebos survive each cutoff,
since a filter that leaves five placebos cannot deliver a p-value below 0.17.

Statistics: Abadie's post/pre RMSPE ratio for the classic estimator, and |tau_aug| for the
ridge-augmented one.
"""
import pandas as pd, numpy as np, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])
CUTS=[1.0,2.0,3.0,5.0,np.inf]

from scipy import stats
out=[]
for meas,lab in [('vol','VOLUME (n_policies)'),('tgt','TARGETING (share_frac_policies)')]:
    v=M[meas].values; pos=v[v>0]
    q75,q25=np.quantile(pos,.75),np.quantile(pos,.25)
    hi=np.where(v>=q75)[0]; lo=np.where(v<=q25)[0]
    pl=rng.choice(lo,size=min(NPLAC,len(lo)),replace=False)
    P=[]
    for a in pl:
        r=one(a,lo[lo!=a])
        if r and np.isfinite(r['ratio']) and np.isfinite(r['tau_aug']): P.append(r)
    ppre=np.array([x['pre'] for x in P]); prat=np.array([x['ratio'] for x in P])
    ptau=np.abs([x['tau_aug'] for x in P])
    T=[]
    for a in hi:
        r=one(a,lo)
        if not r or not np.isfinite(r['ratio']) or not np.isfinite(r['tau_aug']): continue
        r.update(i=CT[a],k=SE[a],name=nm(CT[a]),meas=meas)
        T.append(r)
    R=pd.DataFrame(T)
    log(f"{lab}: {len(P)} placebos (median pre-RMSPE {np.median(ppre):.3f}), "
        f"{len(R)} treated (median {R['pre'].median():.3f})")
    print(f"\n{'='*116}\n{lab}   {len(R)} treated units, {R['i'].nunique()} countries, "
          f"{len(P)} placebos\n{'='*116}")
    print(f"{'ADH cutoff':>12s}{'median adm.':>13s}{'-- CLASSIC: post/pre ratio --':>34s}"
          f"{'-- AUGMENTED: |tau| --':>32s}")
    print(f"{'':12s}{'placebos':>13s}{'median p':>12s}{'p<.10':>11s}{'Fisher p':>11s}"
          f"{'median p':>12s}{'p<.10':>11s}{'Fisher p':>11s}")
    for CUT in CUTS:
        adm_n=[]; ps=[]; pa=[]
        for _,x in R.iterrows():
            m=ppre<=CUT*x['pre'] if np.isfinite(CUT) else np.ones(len(ppre),bool)
            n=int(m.sum()); adm_n.append(n)
            if n<5: ps.append(np.nan); pa.append(np.nan); continue
            ps.append((1+(prat[m]>=x['ratio']).sum())/(1+n))
            pa.append((1+(ptau[m]>=abs(x['tau_aug'])).sum())/(1+n))
        ps=np.array(ps,float); pa=np.array(pa,float)
        ok=np.isfinite(ps)
        fs=-2*np.log(np.clip(ps[ok],1e-6,1)).sum(); fa=-2*np.log(np.clip(pa[ok],1e-6,1)).sum()
        nn=int(ok.sum())
        clab='none' if not np.isfinite(CUT) else f"{CUT:.0f}x"
        print(f"{clab:>12s}{np.median(adm_n):>13.0f}"
              f"{np.nanmedian(ps):>12.3f}{f'{100*np.nanmean(ps[ok]<.10):.0f}%':>11s}"
              f"{1-stats.chi2.cdf(fs,2*nn):>11.4f}"
              f"{np.nanmedian(pa):>12.3f}{f'{100*np.nanmean(pa[ok]<.10):.0f}%':>11s}"
              f"{1-stats.chi2.cdf(fa,2*nn):>11.4f}")
        out.append({'meas':meas,'cut':clab,'med_adm':np.median(adm_n),'n_used':nn,
                    'med_p_scm':np.nanmedian(ps),'rej_scm':np.nanmean(ps[ok]<.10),
                    'fisher_scm':1-stats.chi2.cdf(fs,2*nn),
                    'med_p_aug':np.nanmedian(pa),'rej_aug':np.nanmean(pa[ok]<.10),
                    'fisher_aug':1-stats.chi2.cdf(fa,2*nn)})
    # at ADH's 2x, which countries reject most
    m2=[]
    for _,x in R.iterrows():
        m=ppre<=2.0*x['pre']; n=int(m.sum())
        m2.append(np.nan if n<5 else (1+(ptau[m]>=abs(x['tau_aug'])).sum())/(1+n))
    R['p_aug_2x']=m2
    byc=(R.dropna(subset=['p_aug_2x']).groupby('name')
           .agg(n=('p_aug_2x','size'), rej=('p_aug_2x',lambda s:(s<.10).mean()),
                tau=('tau_aug','mean')).query('n>=8').sort_values('rej',ascending=False))
    print(f"\n  by country at ADH 2x (countries with >=8 treated sectors), "
          f"share of sectors with p_aug<0.10:")
    print("   "+byc.head(10).round(3).to_string().replace("\n","\n   "))
    print()
pd.DataFrame(out).to_csv("sc11_adhfilter.csv",index=False)
log("DONE")
