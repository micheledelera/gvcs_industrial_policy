"""Event study on the country-sector panel: do treated cells diverge before 2018?

  us_exp_ist = exp{ sum_tau  D_tau (Treated_is x 1[t=tau])  + a_is + a_st }   tau != 2017

alpha_st compares treated cells to untreated cells IN THE SAME SECTOR-YEAR, which is
the donor restriction imposed as a fixed effect. alpha_is removes the level of each
country-sector. No country-year fixed effect: that would force identification within
country and reproduce the gravity design.

The raw aggregate shows treated cells growing 70% over 2007-2017 against 16% for
donors, but treated cells are concentrated in faster-growing sectors, which alpha_st
absorbs. This asks whether divergence survives that.

PPML on levels; standard errors clustered by country (190 clusters), which is
conservative given treatment varies at (i,s).
"""
import pandas as pd, numpy as np, pyfixest as pf, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

REF = 2017
d = pd.read_pickle("cs_panel.pkl")
years = sorted(d['t'].unique())
m = pd.DataFrame({
    'us_exp': d['us_exp'].astype('float64').values,
    'fe_is': d.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32').values,
    'fe_st': d.groupby(['ISIC4c','t'],  observed=True).ngroup().astype('int32').values,
    'cl_i':  d['i'].values,
})
names=[]
tv, tr = d['t'].values, d['Treated'].values.astype('float64')
for y in years:
    if y == REF: continue
    m[f"D_{y}"] = tr * (tv == y)
    names.append(f"D_{y}")
log(f"{m.shape} | {len(names)} event terms | fe_is {m['fe_is'].nunique():,} "
    f"fe_st {m['fe_st'].nunique():,} | clusters {m['cl_i'].nunique()}")

fit = pf.fepois(f"us_exp ~ {' + '.join(names)} | fe_is + fe_st", data=m,
                vcov={"CRV1": "cl_i"}, demeaner=pf.LsmrDemeaner(fixef_maxiter=8000,
                fixef_atol=1e-6, fixef_btol=1e-6), store_data=False, copy_data=False)
t = fit.tidy()
t.to_csv("cs_event_results.csv")
print(f"\n### EVENT STUDY  N={fit._N:,}  ref {REF}  cluster: country\n")
got = [n for n in names if n in t.index]
for n in got:
    y = int(n.split('_')[1]); r = t.loc[n]
    era = "pre " if y < 2018 else "POST"
    st = "*" if r['Pr(>|t|)']<0.05 else ("." if r['Pr(>|t|)']<0.10 else " ")
    print(f"  {era} {y}  {r['Estimate']:+.4f}  ({r['Std. Error']:.4f})  "
          f"p={r['Pr(>|t|)']:.3f} {st}")
pre = t.loc[[n for n in got if int(n.split('_')[1]) < 2018]]
print(f"\npre-period: {len(pre)} coefs, mean {pre['Estimate'].mean():+.4f}, "
      f"max|t| {pre['t value'].abs().max():.2f}, n(p<.05)={int((pre['Pr(>|t|)']<.05).sum())}")
try:
    idx=[list(t.index).index(n) for n in pre.index]
    V=np.asarray(fit._vcov)[np.ix_(idx,idx)]; b=pre['Estimate'].values
    W=float(b@np.linalg.solve(V,b)); from scipy import stats
    print(f"joint Wald, all pre-period = 0: chi2({len(b)}) = {W:.1f}, "
          f"p = {stats.chi2.sf(W,len(b)):.4g}")
except Exception as e:
    log(f"joint test unavailable: {type(e).__name__}: {e}")
log("ALL DONE")
