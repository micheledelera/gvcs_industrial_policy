"""Exporter-level clustering for variants A and C.

Only 38 exporters ever carry a nonzero DDD_dev, so if the errors correlate within
exporter, the 2,454 (i,s) clusters used so far flatter the inference. Refit the
baseline and the pair-year specification clustering on i alone.

Same sample, same lag, same regressors as fit_geography.py -- only the vcov moves,
so any change in the standard errors is attributable to the clustering level.
"""
import pandas as pd
import numpy as np
import pyfixest as pf
import gc
import time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, LAG = 156, 3
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
log(f"restricted, lag{LAG}: {d.shape}")

ip    = (d['IP_lag'] / d['IP_lag'].std()).astype('float32')
china = (d['i_int'] == CHINA).astype('float32')
adv   = d['Advanced_i'].astype('float32')
dev   = ((1 - d['Advanced_i']) * (1 - china)).astype('float32')
us    = d['US_trade'].astype('float32')
dec   = (d['target'] * (d['t_int'] >= 2018)).astype('float32')

m = pd.DataFrame({
    'imports':    d['imports'].astype('float32').values,
    'DDD_dev':    (dec * ip * us * dev).values,
    'DDD_adv':    (dec * ip * us * adv).values,
    'IPxUS_dev':  (ip * us * dev).values,
    'IPxUS_adv':  (ip * us * adv).values,
    'IPxUS_chn':  (ip * us * china).values,
    'Dec_US_chn': (dec * us * china).values,
    'Adv_Dec_US': (adv * dec * us).values,
    'fe_ist': d.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
    'fe_jst': d.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
    'fe_ij':  d.groupby(['i','j'], observed=True).ngroup().astype('int32').values,
    'fe_ijt': d.groupby(['i','j','t'], observed=True).ngroup().astype('int32').values,
    'cl_is':  d.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32').values,
    'cl_i':   d['i_int'].values,
})
del d, ip, china, adv, dev, us, dec
gc.collect()
log(f"model matrix {m.shape} | exporter clusters: {m['cl_i'].nunique()} "
    f"| (i,s) clusters: {m['cl_is'].nunique():,}")

VARIANTS = [("A. baseline (alpha_ij)",   "fe_ist + fe_jst + fe_ij"),
            ("C. alpha_ijt (pair-year)", "fe_ist + fe_jst + fe_ijt")]

for name, fes in VARIANTS:
    f = ("imports ~ DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + IPxUS_chn "
         f"+ Dec_US_chn + Adv_Dec_US | {fes}")
    try:
        fit = pf.fepois(f, data=m, vcov={"CRV1": "cl_i"},
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                        lean=True, store_data=False, copy_data=False)
        t = fit.tidy()
        print(f"\n### {name} -- clustered on EXPORTER   N={fit._N:,}")
        print(t.round(5).to_string())
        t.to_csv(f"clus_exporter_{name[0]}.csv")
        log(f"{name}: DDD_dev = {t.loc['DDD_dev','Estimate']:+.4f} "
            f"(se={t.loc['DDD_dev','Std. Error']:.4f}, p={t.loc['DDD_dev','Pr(>|t|)']:.4f})")
        del fit
    except Exception as e:
        log(f"{name} FAILED: {type(e).__name__}: {e}")
    gc.collect()

log("ALL DONE")
