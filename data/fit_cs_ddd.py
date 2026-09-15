"""Country-sector DDD: does policy pay off MORE where China was being displaced?

The plain Treated x Post design cannot test this: alpha_st absorbs everything at
sector-year, and Decoupling IS a sector-year object, so the shock is swept away.
Identification requires the triple, which neither fixed effect spans:

  Treated_is x ChinaShare_s x Post_t     <- alpha_is takes Treated x ChinaShare
                                            alpha_st takes ChinaShare x Post

  us_exp_ist = exp{ b (Treated x ChinaShare x Post) + l (Treated x Post)
                    + a_is + a_st }

b is the object of interest: among policy-targeted country-sectors, did those in
sectors China dominated in the US market gain more after 2018? l is the lower-order
Treated x Post term, which the event study showed is essentially zero.

Then the event-study version, tracing the triple year by year against 2017, since
the plain event study found a decade-long convergence trend (joint Wald p=1.3e-5)
that any post-2018 estimate has to be read against.

PPML on levels, clustered by country.
"""
import pandas as pd, numpy as np, pyfixest as pf, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, USA, REF = 156, 842, 2017
d = pd.read_pickle("cs_panel.pkl")

# China's pre-period share of US imports, by sector (same object as the country panel)
raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
pre = raw[(raw['j']==USA) & (raw['t'].between(2015,2017))]
us_s  = pre.groupby('ISIC4c', observed=True)['imports'].sum()
chn_s = pre[pre['i']==CHINA].groupby('ISIC4c', observed=True)['imports'].sum()
share = (chn_s/us_s).reindex(us_s.index).fillna(0).clip(0,1).rename('ChinaShare')
del raw, pre
d = d.merge(share, on='ISIC4c', how='left')
d['ChinaShare'] = d['ChinaShare'].fillna(0)
log(f"ChinaShare: mean {d['ChinaShare'].mean():.3f}  sd {d['ChinaShare'].std():.3f}")

tr   = d['Treated'].values.astype('float64')
cs   = d['ChinaShare'].values.astype('float64')
post = (d['t'].values >= 2018).astype('float64')
base = pd.DataFrame({
    'us_exp': d['us_exp'].astype('float64').values,
    'fe_is': d.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32').values,
    'fe_st': d.groupby(['ISIC4c','t'],  observed=True).ngroup().astype('int32').values,
    'cl_i':  d['i'].values,
})

# ---- 1. collapsed DDD ----
m = base.copy()
m['TrPost']   = tr * post
m['TrCsPost'] = tr * cs * post
fit = pf.fepois("us_exp ~ TrCsPost + TrPost | fe_is + fe_st", data=m,
                vcov={"CRV1":"cl_i"}, demeaner=pf.LsmrDemeaner(fixef_maxiter=8000,
                fixef_atol=1e-6, fixef_btol=1e-6), lean=True, store_data=False, copy_data=False)
t = fit.tidy(); t.to_csv("cs_ddd_results.csv")
print(f"\n### COLLAPSED DDD  N={fit._N:,}  cluster: country\n{t.round(5).to_string()}")
log(f"b (TrCsPost) = {t.loc['TrCsPost','Estimate']:+.4f} "
    f"(se={t.loc['TrCsPost','Std. Error']:.4f}, p={t.loc['TrCsPost','Pr(>|t|)']:.4f})")
del fit, m

# ---- 2. event-study version of the triple ----
years = sorted(d['t'].unique()); tv = d['t'].values
m = base.copy(); names=[]
for y in years:
    if y == REF: continue
    m[f"T_{y}"]  = tr * (tv==y)
    m[f"TC_{y}"] = tr * cs * (tv==y)
    names.append(f"TC_{y}")
rhs = " + ".join([f"TC_{y}" for y in years if y!=REF] + [f"T_{y}" for y in years if y!=REF])
fit = pf.fepois(f"us_exp ~ {rhs} | fe_is + fe_st", data=m, vcov={"CRV1":"cl_i"},
                demeaner=pf.LsmrDemeaner(fixef_maxiter=8000, fixef_atol=1e-6,
                fixef_btol=1e-6), store_data=False, copy_data=False)
t = fit.tidy(); t.to_csv("cs_ddd_event_results.csv")
print(f"\n### TRIPLE, YEAR BY YEAR  N={fit._N:,}  ref {REF}\n")
got=[n for n in names if n in t.index]
for n in got:
    y=int(n.split('_')[1]); r=t.loc[n]
    era = "pre " if y<2018 else "POST"
    st = "*" if r['Pr(>|t|)']<0.05 else ("." if r['Pr(>|t|)']<0.10 else " ")
    print(f"  {era} {y}  {r['Estimate']:+.4f}  ({r['Std. Error']:.4f})  p={r['Pr(>|t|)']:.3f} {st}")
pre_ = t.loc[[n for n in got if int(n.split('_')[1])<2018]]
print(f"\npre-period: mean {pre_['Estimate'].mean():+.4f}, max|t| {pre_['t value'].abs().max():.2f}, "
      f"n(p<.05)={int((pre_['Pr(>|t|)']<.05).sum())}")
try:
    idx=[list(t.index).index(n) for n in pre_.index]
    V=np.asarray(fit._vcov)[np.ix_(idx,idx)]; b=pre_['Estimate'].values
    W=float(b@np.linalg.solve(V,b)); from scipy import stats
    print(f"joint Wald, pre-period triple = 0: chi2({len(b)}) = {W:.1f}, p = {stats.chi2.sf(W,len(b)):.4g}")
except Exception as e:
    log(f"joint test unavailable: {type(e).__name__}: {e}")
log("ALL DONE")
