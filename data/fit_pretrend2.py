"""Pre-trend event study, take 2 -- with IP on a scale comparable across years.

Take 1 (fit_pretrend.py) standardised IP by a single pooled SD. diag_ip_year.py
shows why that fails as an event study: with IP lagged 3 years, 2010 has ZERO
nonzero cells (hence the ES_2010/G_2010 collinearity drop) and 2011 has 219 with an
SD one-tenth the pooled value -- so ES_2011 came back at +2.10, a scaling artefact
rather than a pre-trend. From 2012 coverage is stable at 11-17.5% of cells, but the
within-year SD still ranges 0.49-1.60x pooled, so year coefficients remain on
inconsistent scales.

Fixes, run together because they trade off differently:
  B  IP standardised WITHIN YEAR. A pure rescaling -- alpha_ist already absorbs IP's
     level and trend, so no identifying variation is lost. Closest to the main
     specification's continuous-intensity estimand.
  C  IP as binary 1[IP > 0]. ~86% of cells are zero so the SD is driven by a thin
     positive tail; a binary treatment sidesteps scale entirely, and the treated
     fraction is stable across years.
  A  anchor: the pooled headline DDD with within-year IP, so the event-study path
     has a reference magnitude to average toward.

Sample restricted to 2012+ throughout. Reference year 2017 (last pre-2018 year).

Post-processing reads the coefficient names back from the fitted result rather than
from the intended list, so a collinearity drop reports rather than raising.
"""
import pandas as pd
import numpy as np
import pyfixest as pf
import gc, time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, LAG, REF, START = 156, 3, 2017, 2012
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
d = raw[raw['IP_lag'].notna() & (raw['t_int'] >= START)].copy()
del raw, lk
gc.collect()

years = sorted(d['t_int'].unique().tolist())
log(f"sample {d.shape} | years {years[0]}-{years[-1]} ({len(years)}) | ref {REF}")

# within-year standardisation; guard against a degenerate year
sd_by_year = d.groupby('t_int')['IP_lag'].transform('std')
ip_wy  = (d['IP_lag'] / sd_by_year.replace(0, np.nan)).fillna(0).astype('float32')
ip_bin = (d['IP_lag'] > 0).astype('float32')

china = (d['i_int'] == CHINA).astype('float32')
adv   = d['Advanced_i'].astype('float32')
dev   = ((1 - d['Advanced_i']) * (1 - china)).astype('float32')
us    = d['US_trade'].astype('float32')
tgt   = d['target'].astype('float32')
post  = (d['t_int'] >= 2018).astype('float32')
tv    = d['t_int'].values

base = pd.DataFrame({
    'imports': d['imports'].astype('float32').values,
    'fe_ist': d.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
    'fe_jst': d.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
    'fe_ij':  d.groupby(['i','j'], observed=True).ngroup().astype('int32').values,
    'cl_is':  d.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32').values,
})
del d
gc.collect()

def report(fit, tag, es_names):
    t = fit.tidy()
    t.to_csv(f"pretrend2_{tag}.csv")
    print(f"\n### {tag}   N={fit._N:,}   cluster (i,s)")
    print(t.round(5).to_string())
    got = [n for n in es_names if n in t.index]
    missing = [n for n in es_names if n not in t.index]
    if missing:
        log(f"{tag}: dropped by collinearity -> {missing}")
    if not got:
        return
    es = t.loc[got].copy()
    es['year'] = [int(i.split('_')[1]) for i in es.index]
    es = es.sort_values('year')
    print(f"\n--- {tag}: event-study path (ref {REF}) ---")
    for _, r in es.iterrows():
        era = "pre " if r['year'] < 2018 else "POST"
        st = "*" if r['Pr(>|t|)'] < 0.05 else ("." if r['Pr(>|t|)'] < 0.10 else " ")
        print(f"  {era} {r['year']}  {r['Estimate']:+.4f}  ({r['Std. Error']:.4f})  "
              f"p={r['Pr(>|t|)']:.3f} {st}")
    pre = es[es['year'] < 2018]
    if len(pre):
        log(f"{tag}: pre-period {len(pre)} coefs, mean {pre['Estimate'].mean():+.4f}, "
            f"max|t| {pre['t value'].abs().max():.2f}, n(p<.05)={int((pre['Pr(>|t|)']<.05).sum())}")
        try:
            idx = [list(t.index).index(n) for n in pre.index]
            V = np.asarray(fit._vcov)[np.ix_(idx, idx)]
            b = pre['Estimate'].values
            W = float(b @ np.linalg.solve(V, b))
            from scipy import stats
            log(f"{tag}: joint Wald, pre-period = 0: chi2({len(b)}) = {W:.2f}, "
                f"p = {stats.chi2.sf(W, len(b)):.4f}")
        except Exception as e:
            log(f"{tag}: joint test unavailable: {type(e).__name__}: {e}")

# ---- A. anchor: pooled DDD with within-year IP ----
m = base.copy()
m['DDD_dev']    = (post * tgt * ip_wy * us * dev).values
m['DDD_adv']    = (post * tgt * ip_wy * us * adv).values
m['IPxUS_dev']  = (ip_wy * us * dev).values
m['IPxUS_adv']  = (ip_wy * us * adv).values
m['IPxUS_chn']  = (ip_wy * us * china).values
m['Dec_US_chn'] = (post * tgt * us * china).values
m['Adv_Dec_US'] = (adv * post * tgt * us).values
log(f"A. anchor: {m.shape}")
try:
    fit = pf.fepois("imports ~ DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + IPxUS_chn "
                    "+ Dec_US_chn + Adv_Dec_US | fe_ist + fe_jst + fe_ij",
                    data=m, vcov={"CRV1": "cl_is"},
                    demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                    store_data=False, copy_data=False)
    report(fit, "A_anchor_withinyear", [])
    t = fit.tidy()
    log(f"A. anchor: DDD_dev = {t.loc['DDD_dev','Estimate']:+.4f} "
        f"(p={t.loc['DDD_dev','Pr(>|t|)']:.4f})  [pooled-SD headline was +0.0335]")
    del fit
except Exception as e:
    log(f"A FAILED: {type(e).__name__}: {e}")
del m; gc.collect()

# ---- B and C. event studies ----
for tag, ipvar in [("B_withinyear_continuous", ip_wy), ("C_binary_IPpos", ip_bin)]:
    m = base.copy()
    ipusdev = (ipvar * us * dev).values
    es_names = []
    for y in years:
        yr = (tv == y).astype('float32')
        m[f"G_{y}"] = ipusdev * yr
        if y != REF:
            m[f"ES_{y}"] = ipusdev * tgt.values * yr
            es_names.append(f"ES_{y}")
    m['DDD_adv']    = (post * tgt * ipvar * us * adv).values
    m['IPxUS_adv']  = (ipvar * us * adv).values
    m['IPxUS_chn']  = (ipvar * us * china).values
    m['Dec_US_chn'] = (post * tgt * us * china).values
    m['Adv_Dec_US'] = (adv * post * tgt * us).values
    log(f"{tag}: {m.shape}, {len(es_names)} ES terms")
    rhs = " + ".join(es_names + [f"G_{y}" for y in years] +
                     ['DDD_adv','IPxUS_adv','IPxUS_chn','Dec_US_chn','Adv_Dec_US'])
    try:
        fit = pf.fepois(f"imports ~ {rhs} | fe_ist + fe_jst + fe_ij",
                        data=m, vcov={"CRV1": "cl_is"},
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                        store_data=False, copy_data=False)
        report(fit, tag, es_names)
        del fit
    except Exception as e:
        log(f"{tag} FAILED: {type(e).__name__}: {e}")
    del m; gc.collect()

log("ALL DONE")
