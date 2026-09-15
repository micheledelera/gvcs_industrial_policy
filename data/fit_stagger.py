"""Staggered event study, BETWEEN countries.

  X^US_ikt = exp{ sum_tau b_tau (D^k_tau x IP^pre_ik) + a_ik + mu_kt }

D^k_tau = 1[t - onset_k = tau], event time relative to sector k's decoupling onset.
IP^pre_ik = country i's mean share_n_policies in sector k over 2015-17, standardised.

Why this is between-country: mu_kt fixes the sector-year, so every comparison is
across countries supplying the SAME sector in the SAME year -- those that had
targeted k before the shock against those that had not. a_ik absorbs each pair's
level, which is industrial capability in that sector, held exactly fixed rather
than matched on observables. No alpha_it: that would absorb the between-country
variation this design exists to use.

D_tau alone varies at (k,t) and is absorbed by mu_kt, so only the interaction is
estimable -- which is the point, since the average effect of decoupling on a sector
is the mechanical part.

ONSET FROM LEVELS, NOT SHARES. Within a sector-year, shares sum to one, so "China's
share fell" and "everyone else's share rose" are the same statement; defining the
event that way and then measuring other countries' gains is close to tautological.
Onset here is the first year China's US exports in LEVELS fall >=10% below their
2015-17 average and stay there.

Sample: developing excluding China, restricted to the 72 sectors with a decoupling
onset. Keeping the 53 non-onset sectors as never-treated changes nothing: with mu_kt
saturated at sector-year and alpha_ik at pair, a non-onset sector forms a separable
block -- zero regressor variation, fixed effects estimated from its own rows alone --
so it contributes nothing to beta. Verified: every coefficient in the full-sample
version equals the restricted one times 1.056, a constant across all twelve, which is
only the IP standardiser responding to the different sample. Restricting therefore
reports the honest estimating sample, 179,964 rather than 318,852.
Zeros retained (PPML on levels) -- restricting to pairs already exporting to the US
would select on the outcome, and section 3i showed that matters.

CAVEAT: this is TWFE with event-time dummies. With heterogeneous effects across the
nine onset cohorts it can be biased; a Sun-Abraham cohort-interacted version is the
robustness check, not run here.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, os, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

USA, CHINA, OUT = 842, 156, "stagger_results.csv"
LO, HI = -6, 6      # event time binned at the endpoints

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
chn_us = (raw[(raw['j']==USA)&(raw['i']==CHINA)]
          .groupby(['ISIC4c','t'], observed=True)['imports'].sum().unstack())
base = chn_us[[2015,2016,2017]].mean(axis=1)
def onset(row):
    b = base[row.name]
    if not np.isfinite(b) or b<=0: return np.nan
    for y in range(2016,2025):
        if y not in row.index or not np.isfinite(row[y]): continue
        later=[row[z] for z in range(y,2025) if z in row.index and np.isfinite(row[z])]
        if row[y] <= b*0.90 and len(later) and max(later) <= b*1.02: return y
    return np.nan
on = chn_us.apply(onset, axis=1).rename('onset')
log(f"sectors with onset (China US exports, LEVELS): {on.notna().sum()} of {len(on)}")
print(on.value_counts().sort_index().to_string())
del raw; gc.collect()

d = pd.read_pickle("rot_panel.pkl")
d = d[(d['Advanced_i']==0) & (d['i']!=CHINA)].copy()
ip = (d[d['t'].between(2015,2017)].groupby(['i','ISIC4c'], observed=True)
      ['share_n_policies'].mean().rename('IPpre'))
d = d.merge(ip, on=['i','ISIC4c'], how='left').merge(on, on='ISIC4c', how='left')
d = d[d['onset'].notna()].copy()      # <-- keep ONLY sectors with a decoupling onset
d['IPpre'] = (d['IPpre'].fillna(0) / d['IPpre'].std()).astype('float64')
log(f"developing ex-China panel {d.shape} | pairs {d.groupby(['i','ISIC4c'],observed=True).ngroup().nunique():,}")

d['et'] = (d['t'] - d['onset']).clip(LO, HI)
d['fe_ik']=d.groupby(['i','ISIC4c'],observed=True).ngroup().astype('int32')
d['fe_kt']=d.groupby(['ISIC4c','t'],observed=True).ngroup().astype('int32')
d['cl_i']=d['i'].astype('int32')
d['cl_k']=d.groupby('ISIC4c',observed=True).ngroup().astype('int32')
names=[]
for tau in range(LO,HI+1):
    if tau==-1: continue
    nm=f"E{'m' if tau<0 else 'p'}{abs(tau)}"
    d[nm] = ((d['et']==tau).astype('float64') * d['IPpre'])
    names.append(nm)
log(f"{len(names)} event-time interactions, ref tau=-1 | treated rows "
    f"{int((d['et'].notna()).sum()):,}")

d['fe_it']  = d.groupby(['i','t'], observed=True).ngroup().astype('int32')
d['fe_ict'] = d.groupby(['i','isic2','t'], observed=True).ngroup().astype('int32')
import sys
FES = {"between": "fe_ik + fe_kt",
       "within_it":  "fe_ik + fe_kt + fe_it",     # country-year: the direct remedy
       "within_ict": "fe_ik + fe_kt + fe_ict"}[sys.argv[1]]
fit = pf.fepois(f"x_us ~ {' + '.join(names)} | {FES}", data=d,
                vcov={"CRV1":"cl_i + cl_k"},
                demeaner=pf.LsmrDemeaner(fixef_maxiter=8000, fixef_atol=1e-6,
                fixef_btol=1e-6), store_data=False, copy_data=False)
t = fit.tidy(); t.to_csv(f"stagger_{sys.argv[1]}.csv")
print(f"\n### STAGGERED EVENT STUDY x IP [{sys.argv[1]}]  N={fit._N:,}  "
      f"ref tau=-1  FE: {FES}\n")
for nm in names:
    if nm not in t.index: continue
    tau = (-1 if nm.startswith('Em') else 1)*int(nm[2:])
    r=t.loc[nm]; era="pre " if tau<0 else "POST"
    st="*" if r['Pr(>|t|)']<0.05 else ("." if r['Pr(>|t|)']<0.10 else " ")
    print(f"  {era} tau={tau:+d}  {r['Estimate']:+.4f} ({r['Std. Error']:.4f})  "
          f"p={r['Pr(>|t|)']:.3f} {st}")
pre=[n for n in names if n.startswith('Em') and n in t.index]
tp=t.loc[pre]
print(f"\npre-period: {len(pre)} coefs, mean {tp['Estimate'].mean():+.4f}, "
      f"max|t| {tp['t value'].abs().max():.2f}, n(p<.05)={int((tp['Pr(>|t|)']<.05).sum())}")
try:
    idx=[list(t.index).index(n) for n in pre]
    V=np.asarray(fit._vcov)[np.ix_(idx,idx)]; b=tp['Estimate'].values
    W=float(b@np.linalg.solve(V,b)); from scipy import stats
    print(f"joint Wald, pre-period = 0: chi2({len(b)}) = {W:.2f}, p = {stats.chi2.sf(W,len(b)):.4g}")
except Exception as e:
    log(f"joint test unavailable: {type(e).__name__}: {e}")
log("ALL DONE")
