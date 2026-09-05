"""Leave-one-exporter-out sweep over the top 10 by treatment mass.

Mexico and India alone are 43% of the PPML-weighted treatment mass (see
leverage_by_exporter.csv), so the question is whether DDD_dev is a broad
regularity or a few country stories. Drop one exporter at a time from the
headline specification and watch the coefficient.

Baseline FE (alpha_ist + alpha_jst + alpha_ij), lag 3, cluster (i,s) -- the
headline spec from RESULTS section 2, so the "none" row should reproduce
+0.0335 (p=0.016).

Two design points:
  * IP is standardised ONCE on the full restricted sample and that SD is reused
    for every drop, so coefficients are comparable across rows. (fit_geography.py
    variant D recomputed the SD within its subsample; the difference is tiny at
    2% of rows but is not zero.)
  * The model matrix is built once and only ROWS are subset, so no drop can
    change anything but the sample.

Results are appended to loo_results.csv after every fit, so a container restart
loses at most one fit.
"""
import pandas as pd
import numpy as np
import pyfixest as pf
import gc, os, time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, LAG = 156, 3
MEASURE = 'share_frac_policies'
DEST = {842,276,251,380,826,528,724,56,392,124,410,36,752,40,208,246,372,620,300,579,757,554}
OUT = "loo_results.csv"

# top 10 developing exporters by PPML-weighted treatment mass, descending
DROPS = [(None, "none (baseline)"), (484, "Mexico"), (699, "India"), (704, "Vietnam"),
         (710, "South Africa"), (50, "Bangladesh"), (360, "Indonesia"), (76, "Brazil"),
         (784, "UAE"), (764, "Thailand"), (458, "Malaysia")]

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

IP_SD = float(d['IP_lag'].std())          # fixed once, reused for every drop
ip    = (d['IP_lag'] / IP_SD).astype('float32')
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
    'cl_is':  d.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32').values,
    'i_int':  d['i_int'].values,
})
del d, ip, china, adv, dev, us, dec
gc.collect()
log(f"model matrix {m.shape} | IP SD fixed at {IP_SD:.6f}")

F = ("imports ~ DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + IPxUS_chn "
     "+ Dec_US_chn + Adv_Dec_US | fe_ist + fe_jst + fe_ij")

for code, name in DROPS:
    sub = m if code is None else m[m['i_int'] != code]
    treated = int((sub['DDD_dev'] > 0).sum())
    try:
        fit = pf.fepois(F, data=sub.drop(columns='i_int'), vcov={"CRV1": "cl_is"},
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                        lean=True, store_data=False, copy_data=False)
        t = fit.tidy()
        b  = t.loc['DDD_dev','Estimate']; se = t.loc['DDD_dev','Std. Error']
        p  = t.loc['DDD_dev','Pr(>|t|)']
        print(f"\n### drop {name}   N={fit._N:,}   treated obs={treated:,}")
        print(t.round(5).to_string())
        row = pd.DataFrame([{'dropped': name, 'code': code, 'N': fit._N,
                             'treated_obs': treated, 'coef': b, 'se': se, 'p': p,
                             'ci_lo': t.loc['DDD_dev','2.5%'],
                             'ci_hi': t.loc['DDD_dev','97.5%'],
                             'IPxUS_dev': t.loc['IPxUS_dev','Estimate']}])
        row.to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
        log(f"drop {name}: DDD_dev = {b:+.4f} (se={se:.4f}, p={p:.4f})")
        del fit
    except Exception as e:
        log(f"drop {name} FAILED: {type(e).__name__}: {e}")
    del sub
    gc.collect()

if os.path.exists(OUT):
    r = pd.read_csv(OUT)
    print("\n" + "="*76)
    print("LEAVE-ONE-EXPORTER-OUT: DDD_dev")
    print("="*76)
    print(r[['dropped','N','treated_obs','coef','se','p']].round(4).to_string(index=False))
    base = r.loc[r['dropped'] == 'none (baseline)', 'coef']
    if len(base):
        print(f"\nbaseline {base.iloc[0]:+.4f} | across drops: "
              f"min {r['coef'].min():+.4f}, max {r['coef'].max():+.4f}, "
              f"median {r['coef'].median():+.4f}")
log("ALL DONE")
