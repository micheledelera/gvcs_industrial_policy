"""
Build-up ladder: parsimonious DDD -> dev/adv split -> add IPxUS -> full spec.

Restricted sample (22 advanced destinations), share_frac_policies, cluster (i,s),
FE alpha_ist + alpha_jst + alpha_ij held fixed throughout. Lags 0 and 3.

China is in the sample with its own terms (Dec_US_chn, IPxUS_chn) in EVERY step,
so movement across steps is attributable to the dev/adv structure alone and not
to China shifting between groups.
"""
import pandas as pd
import numpy as np
import pyfixest as pf
import gc
import time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA = 156
MEASURE = 'share_frac_policies'
DEST = {842,276,251,380,826,528,724,56,392,124,410,36,752,40,208,246,372,620,300,579,757,554}

CHINA_TERMS = "Dec_US_chn + IPxUS_chn"
STEPS = [
    ("1. DDD pooled",        f"DDD_pooled + {CHINA_TERMS}"),
    ("2. + dev/adv split",   f"DDD_dev + DDD_adv + {CHINA_TERMS}"),
    ("3. + IPxUS",           f"DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + {CHINA_TERMS}"),
    ("4. + Adv_Dec_US (full)",
     f"DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + Adv_Dec_US + {CHINA_TERMS}"),
]

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i_int'] = raw['i'].astype('int32'); raw['t_int'] = raw['t'].astype('int32')
raw[MEASURE] = pd.to_numeric(raw[MEASURE], errors='coerce').fillna(0).astype('float32')
lookup0 = (raw[['i_int','ISIC4c','t_int',MEASURE]]
           .drop_duplicates(subset=['i_int','ISIC4c','t_int']))
raw = raw[raw['j'].astype('int32').isin(DEST)].copy()
log(f"restricted sample: {raw.shape}")

rows = []
for LAG in [0, 3]:
    lk = lookup0.copy()
    lk['t_int'] = lk['t_int'] + LAG
    lk = lk.rename(columns={MEASURE: 'IP_lag'})
    d = raw.merge(lk, on=['i_int','ISIC4c','t_int'], how='left')
    d = d[d['IP_lag'].notna()]

    ip    = (d['IP_lag'] / d['IP_lag'].std()).astype('float32')
    china = (d['i_int'] == CHINA).astype('float32')
    adv   = d['Advanced_i'].astype('float32')
    dev   = ((1 - d['Advanced_i']) * (1 - china)).astype('float32')
    us    = d['US_trade'].astype('float32')
    dec   = (d['target'] * (d['t_int'] >= 2018)).astype('float32')

    m = pd.DataFrame({
        'imports':    d['imports'].astype('float32').values,
        'DDD_pooled': (dec * ip * us * (1 - china)).values,   # dev and adv together
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
    })
    del d, ip, china, adv, dev, us, dec
    gc.collect()
    log(f"--- lag {LAG}: model matrix {m.shape} ---")

    for name, rhs in STEPS:
        try:
            fit = pf.fepois(f"imports ~ {rhs} | fe_ist + fe_jst + fe_ij", data=m,
                            vcov={"CRV1": "cl_is"},
                            demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                            lean=True, store_data=False, copy_data=False)
            t = fit.tidy()
            print(f"\n### lag {LAG} | {name}   N={fit._N:,}")
            print(t.round(5).to_string())
            for term in t.index:
                rows.append({'lag': LAG, 'step': name, 'term': term,
                             'coef': t.loc[term,'Estimate'], 'se': t.loc[term,'Std. Error'],
                             'p': t.loc[term,'Pr(>|t|)']})
            key = 'DDD_pooled' if 'DDD_pooled' in t.index else 'DDD_dev'
            log(f"lag {LAG} | {name}: {key} = {t.loc[key,'Estimate']:+.4f} (p={t.loc[key,'Pr(>|t|)']:.4f})")
            del fit
        except Exception as e:
            log(f"lag {LAG} | {name} FAILED: {type(e).__name__}: {e}")
    del m
    gc.collect()

res = pd.DataFrame(rows)
res.to_csv("buildup_results.csv", index=False)

print("\n" + "="*84)
print("BUILD-UP: key coefficient by step")
print("="*84)
for LAG in [0, 3]:
    print(f"\n--- lag {LAG} ---")
    sub = res[(res['lag'] == LAG) & (res['term'].isin(['DDD_pooled','DDD_dev','DDD_adv']))]
    piv = sub.pivot_table(index='step', columns='term', values=['coef','p'])
    print(piv.round(4).to_string())
log("DONE")
