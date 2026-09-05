"""Pre-trend test in the gravity specification itself.

The event study we ran earlier was a US-only DiD with a different FE structure and
was inconclusive. The headline is a gravity result, so it needs its own test.

Replace  dec_st = target_s x 1[t>=2018]  with a full set of year interactions:

  ES_tau  = target_s x 1[t=tau] x IP_{i,s,t-3} x US_j x dev_i      tau != 2017
  G_tau   =              1[t=tau] x IP_{i,s,t-3} x US_j x dev_i     all tau

G_tau is the year-by-year IP-gradient of the US share for developing exporters
across ALL sectors; ES_tau is the EXTRA gradient in decoupling-targeted sectors in
year tau, relative to 2017. So ES_tau is the year-by-year analogue of DDD_dev, and
the identifying assumption is testable: ES_tau should be flat and near zero for
tau < 2018.

Note that target_s x 1[t=tau] x US_j -- the targeted-sector US shock common to all
exporters -- is absorbed by alpha_jst, so it needs no explicit term.

The adv, China and Adv_Dec_US controls stay in their baseline (non-year-varying)
form. The object of interest is the developing-country path; year-varying every
control would triple the parameter count for no gain in what is being tested.

Reference year 2017: the last year before the 2018 cutoff used throughout.
"""
import pandas as pd
import numpy as np
import pyfixest as pf
import gc, time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, LAG, REF = 156, 3, 2017
MEASURE = 'share_frac_policies'
DEST = {842,276,251,380,826,528,724,56,392,124,410,36,752,40,208,246,372,620,300,579,757,554}

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i_int'] = raw['i'].astype('int32'); raw['t_int'] = raw['t'].astype('int32')
raw[MEASURE] = pd.to_numeric(raw[MEASURE], errors='coerce').fillna(0).astype('float32')
lk = (raw[['i_int','ISIC4c','t_int',MEASURE]]
      .drop_duplicates(subset=['i_int','ISIC4c','t_int']))
lk['t_int'] = lk['t_int'] + LAG
lk = lk.rename(columns={MEASURE: 'IP_lag'})
raw = raw[raw['j'].astype('int32').isin(DEST)].copy()
raw = raw.merge(lk, on=['i_int','ISIC4c','t_int'], how='left')
d = raw[raw['IP_lag'].notna()].copy()
del raw, lk
gc.collect()

years = sorted(d['t_int'].unique().tolist())
log(f"sample {d.shape} | years {years[0]}-{years[-1]} ({len(years)}) | ref {REF}")
assert REF in years, f"reference year {REF} not in sample"

ip    = (d['IP_lag'] / d['IP_lag'].std()).astype('float32')
china = (d['i_int'] == CHINA).astype('float32')
adv   = d['Advanced_i'].astype('float32')
dev   = ((1 - d['Advanced_i']) * (1 - china)).astype('float32')
us    = d['US_trade'].astype('float32')
tgt   = d['target'].astype('float32')
dec   = (tgt * (d['t_int'] >= 2018)).astype('float32')
ipusdev = (ip * us * dev).astype('float32')

m = pd.DataFrame({
    'imports':    d['imports'].astype('float32').values,
    'DDD_adv':    (dec * ip * us * adv).values,
    'IPxUS_adv':  (ip * us * adv).values,
    'IPxUS_chn':  (ip * us * china).values,
    'Dec_US_chn': (dec * us * china).values,
    'Adv_Dec_US': (adv * dec * us).values,
    'fe_ist': d.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
    'fe_jst': d.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
    'fe_ij':  d.groupby(['i','j'], observed=True).ngroup().astype('int32').values,
    'cl_is':  d.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32').values,
})

es_terms, g_terms = [], []
tv = d['t_int'].values
for y in years:
    yr = (tv == y).astype('float32')
    g = f"G_{y}"
    m[g] = (ipusdev.values * yr)
    g_terms.append(g)
    if y != REF:
        e = f"ES_{y}"
        m[e] = (ipusdev.values * tgt.values * yr)
        es_terms.append(e)

del d, ip, china, adv, dev, us, tgt, dec, ipusdev, tv
gc.collect()
log(f"model matrix {m.shape} | {len(es_terms)} event-study terms, {len(g_terms)} gradient terms")

rhs = " + ".join(es_terms + g_terms +
                 ['DDD_adv','IPxUS_adv','IPxUS_chn','Dec_US_chn','Adv_Dec_US'])
f = f"imports ~ {rhs} | fe_ist + fe_jst + fe_ij"

fit = pf.fepois(f, data=m, vcov={"CRV1": "cl_is"},
                demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                store_data=False, copy_data=False)
t = fit.tidy()
print(f"\n### PRE-TREND EVENT STUDY   N={fit._N:,}   cluster (i,s)   ref={REF}")
print(t.round(5).to_string())
t.to_csv("pretrend_results.csv")

es = t.loc[[e for e in es_terms]].copy()
es['year'] = [int(i.split('_')[1]) for i in es.index]
es = es.sort_values('year')
print("\n" + "="*66)
print(f"EVENT-STUDY PATH (ES_tau: extra IP-gradient in targeted sectors, ref {REF})")
print("="*66)
for _, r in es.iterrows():
    era = "pre " if r['year'] < 2018 else "post"
    star = "*" if r['Pr(>|t|)'] < 0.05 else (" " if r['Pr(>|t|)'] >= 0.10 else ".")
    print(f"  {era} {r['year']}  {r['Estimate']:+.4f}  ({r['Std. Error']:.4f})  "
          f"p={r['Pr(>|t|)']:.3f} {star}   [{r['2.5%']:+.4f}, {r['97.5%']:+.4f}]")

pre = es[es['year'] < 2018]
print(f"\npre-period: {len(pre)} coefficients, "
      f"max |t| = {pre['t value'].abs().max():.2f}, "
      f"mean = {pre['Estimate'].mean():+.4f}")
try:
    names = list(pre.index)
    idx = [list(t.index).index(n) for n in names]
    V = np.asarray(fit._vcov)[np.ix_(idx, idx)]
    b = pre['Estimate'].values
    W = float(b @ np.linalg.solve(V, b))
    from scipy import stats
    print(f"joint Wald test, all pre-period ES = 0: chi2({len(b)}) = {W:.2f}, "
          f"p = {stats.chi2.sf(W, len(b)):.4f}")
except Exception as e:
    log(f"joint test unavailable: {type(e).__name__}: {e}")
log("ALL DONE")
