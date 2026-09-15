"""§5f. Country-demeaned synthetic control — reintroducing within-country without
abandoning the between-country donor pool.

§5e showed the §5b-§5d estimates were country effects: Vietnam's UNtargeted sectors beat
their synthetics by +0.86 against +1.06 for the targeted ones, difference p = 0.604. The
between-country design attributes a country-wide boom to whichever unit is labelled treated.

Fix: demean the outcome by the country's own cross-sector average each year,

    y~_ikt = y_ikt - mean_k y_ikt

so any country-year shock is removed BEFORE matching. The treated unit becomes "Vietnam's
computers relative to Vietnam"; donors become "Cambodia's footwear relative to Cambodia".
tau then asks whether a targeted sector rose relative to its own country by more than
comparable sectors rose relative to theirs. The donor pool stays between-country, which is
the estimand; the confound is differenced out. This is Ferman-Pinto demeaning applied at
country level, and the synthetic-control analogue of gravity's alpha_it -- presumably why
gravity survived when these designs did not.

Everything else identical to §5b: donors LOW policy, DIFFERENT country, Chinese US share
within 10pp, log size within 2.0, K = 20 nearest on MVA/GDP, log MVA per capita, ECI and
export share; weights on eleven lagged (demeaned) outcomes plus those four; canonical and
ridge-augmented; 2018; placebo over ~495 low-policy units; and the §5e high-vs-low check
repeated on the demeaned outcome.
"""
import pandas as pd, numpy as np, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])

# ---- country-demean the outcome, in levels of log exports, year by year ----
Lfull=np.hstack([Lp,Lq])                       # units x years, log US exports
ctry_mean=pd.DataFrame(Lfull,index=pd.Index(CT,name='i')).groupby(level='i').transform('mean').values
LD=Lfull-ctry_mean
Lp_raw,Lq_raw=Lp.copy(),Lq.copy()
Lp,Lq=LD[:,:len(PRE)],LD[:,len(PRE):]
log(f"demeaned: sd of raw log exports {Lfull.std():.3f} -> demeaned {LD.std():.3f}")
print(f"  country-year means removed for {len(set(CT))} countries\n")

from scipy import stats
def jkc(df,col):
    cs=df['i'].unique(); m=df[col].mean()
    a=np.array([df[df['i']!=c][col].mean() for c in cs])
    return m, np.sqrt((len(cs)-1)/len(cs)*((a-a.mean())**2).sum())

allrows=[]
for meas,lab in [('vol','VOLUME (n_policies)'),('tgt','TARGETING (share_frac_policies)')]:
    v=M[meas].values; pos=v[v>0]
    q75,q25=np.quantile(pos,.75),np.quantile(pos,.25)
    hi=np.where(v>=q75)[0]; lo=np.where(v<=q25)[0]
    pl=rng.choice(lo,size=min(NPLAC,len(lo)),replace=False)
    P=[]
    for a in pl:
        r=one(a,lo[lo!=a])
        if r and np.isfinite(r['ratio']) and np.isfinite(r['tau_aug']): P.append(r)
    ppre=np.array([x['pre'] for x in P]); ptau=np.abs([x['tau_aug'] for x in P])
    T=[]
    for a in hi:
        r=one(a,lo)
        if not r or not np.isfinite(r['tau_aug']): continue
        m2=ppre<=2.0*r['pre']
        r.update(i=CT[a],k=str(SE[a]),name=nm(CT[a]),
                 p_aug=((1+(ptau[m2]>=abs(r['tau_aug'])).sum())/(1+int(m2.sum()))
                        if m2.sum()>=5 else np.nan),
                 fit_x=r['pre']/np.median(ppre))
        T.append(r)
    R=pd.DataFrame(T); R['meas']=meas; allrows.append(R)
    a_=jkc(R,'tau_scm'); b_=jkc(R,'tau_aug'); g_=jkc(R,'gap_pre')
    ok=R['p_aug'].notna(); n=int(ok.sum())
    fa=-2*np.log(R.loc[ok,'p_aug'].clip(1e-6)).sum()
    print(f"{'='*100}\n{lab}  (country-demeaned outcome)   treated {len(R)}, "
          f"{R['i'].nunique()} countries, {len(P)} placebos\n{'='*100}")
    print(f"  median pre-RMSPE   treated {R['pre'].median():.3f}   placebo {np.median(ppre):.3f}")
    print(f"  gap pre      {g_[0]:>+8.4f} ({g_[1]:.4f}) t={g_[0]/g_[1]:>+5.2f}")
    print(f"  tau classic  {a_[0]:>+8.4f} ({a_[1]:.4f}) t={a_[0]/a_[1]:>+5.2f}   "
          f"median p {R['p_aug'].median():.3f}")
    print(f"  tau AUG      {b_[0]:>+8.4f} ({b_[1]:.4f}) t={b_[0]/b_[1]:>+5.2f}   "
          f"p<.10 {int((R.loc[ok,'p_aug']<.10).sum())}/{n} "
          f"({100*(R.loc[ok,'p_aug']<.10).mean():.0f}%, null 10%)   "
          f"Fisher p={1-stats.chi2.cdf(fa,2*n):.4f}")

    # §5e check repeated on the demeaned outcome
    VN=704
    rr=[]
    for a in range(len(M)):
        if CT[a]!=VN: continue
        g='HIGH' if a in set(hi) else ('LOW' if a in set(lo) else None)
        if not g: continue
        r=one(a, lo[lo!=a])
        if r and np.isfinite(r['tau_aug']): rr.append((g,r['tau_aug']))
    if rr:
        H=np.array([t for g,t in rr if g=='HIGH']); Lw=np.array([t for g,t in rr if g=='LOW'])
        if len(H)>=3 and len(Lw)>=5:
            d=H.mean()-Lw.mean(); av=np.concatenate([H,Lw]); nH=len(H)
            pm=np.array([(lambda s:s[:nH].mean()-s[nH:].mean())(rng.permutation(av))
                         for _ in range(5000)])
            print(f"  VIETNAM on the demeaned outcome: HIGH n={len(H)} mean {H.mean():+.3f}   "
                  f"LOW n={len(Lw)} mean {Lw.mean():+.3f}   diff {d:+.3f}   "
                  f"permutation p={float((np.abs(pm)>=abs(d)).mean()):.4f}")
    print()
pd.concat(allrows).to_csv("sc14_demean.csv",index=False)
log("DONE")
