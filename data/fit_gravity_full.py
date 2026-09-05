import pandas as pd
import numpy as np
import pyfixest as pf
import gc
import time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA = 156
MEASURE = 'share_frac_policies'
LAGS = [0, 3]

# FULL sample: all 229 destinations, no restriction. China retained with own terms.
# Winning treatment definition: target_s x 1[t>=2018] (author's original variable).

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i_int'] = raw['i'].astype('int32'); raw['t_int'] = raw['t'].astype('int32')
raw[MEASURE] = pd.to_numeric(raw[MEASURE], errors='coerce').fillna(0).astype('float32')
log(f"loaded full panel: {raw.shape}")

lookup = (raw[['i_int','ISIC4c','t_int',MEASURE]]
          .drop_duplicates(subset=['i_int','ISIC4c','t_int']))
log(f"(i,s,t) lookup: {len(lookup):,}")

for LAG in LAGS:
    log(f"================ LAG {LAG} ================")
    lk = lookup.copy()
    lk['t_int'] = lk['t_int'] + LAG
    lk = lk.rename(columns={MEASURE: 'IP_lag'})

    d = raw.merge(lk, on=['i_int','ISIC4c','t_int'], how='left')
    d = d[d['IP_lag'].notna()]
    log(f"after lag merge: {d.shape}")

    ipz   = (d['IP_lag'] / d['IP_lag'].std()).astype('float32')
    china = (d['i_int'] == CHINA).astype('float32')
    adv   = d['Advanced_i'].astype('float32')
    dev   = ((1 - d['Advanced_i']) * (1 - china)).astype('float32')
    us    = d['US_trade'].astype('float32')
    dec   = (d['target'] * (d['t_int'] >= 2018)).astype('float32')

    # build only what the fit needs, then drop everything else
    m = pd.DataFrame({
        'imports':    d['imports'].astype('float32'),
        'DDD_dev':    dec * ipz * us * dev,
        'DDD_adv':    dec * ipz * us * adv,
        'IPxUS_dev':  ipz * us * dev,
        'IPxUS_adv':  ipz * us * adv,
        'IPxUS_chn':  ipz * us * china,
        'Dec_US_chn': dec * us * china,
        'Adv_Dec_US': adv * dec * us,
        'fe_ist': d.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
        'fe_jst': d.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
        'fe_ij':  d.groupby(['i','j'], observed=True).ngroup().astype('int32').values,
        'cl_is':  d.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32').values,
    })
    del d, ipz, china, adv, dev, us, dec
    gc.collect()
    log(f"model matrix {m.shape}, mem {m.memory_usage(deep=True).sum()/1e9:.2f}GB | "
        f"FE: ist {m['fe_ist'].nunique():,} jst {m['fe_jst'].nunique():,} ij {m['fe_ij'].nunique():,}")

    f = ("imports ~ DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + IPxUS_chn "
         "+ Dec_US_chn + Adv_Dec_US | fe_ist + fe_jst + fe_ij")
    try:
        fit = pf.fepois(f, data=m, vcov={"CRV1": "cl_is"},
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                        lean=True, store_data=False, copy_data=False)
        t = fit.tidy()
        print(f"\n### FULL SAMPLE GRAVITY -- {MEASURE}, lag {LAG}   N={fit._N:,}   cluster (i,s)")
        print(t.round(5).to_string())
        t.to_csv(f"gravity_full_lag{LAG}.csv")
        log(f"lag {LAG}: DDD_dev = {t.loc['DDD_dev','Estimate']:+.4f} "
            f"(p={t.loc['DDD_dev','Pr(>|t|)']:.4f})")
        del fit
    except Exception as e:
        log(f"lag {LAG} FAILED: {type(e).__name__}: {e}")
    del m
    gc.collect()

log("ALL DONE")
