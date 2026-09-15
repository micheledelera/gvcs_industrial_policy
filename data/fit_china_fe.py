"""
Option A: absorb China non-parametrically with a fixed effect, instead of
controlling it with the two parametric terms (Dec_US_chn, IPxUS_chn).

fe_chn_us takes a distinct level for each (sector, year) among China->US flows
and one common level for everything else. This imposes no functional form on how
China's decline relates to the treatment -- the parametric version assumes it is
linear in dec and in IP -- and reports nothing mechanical.

Dec_US_chn and IPxUS_chn are dropped: both vary at (China, s, t) for j=US, which
is exactly what the new FE spans, so they would be collinear with it.

China's non-US flows stay in the ordinary FE structure, so China still informs
alpha_ist for itself and alpha_jst for the other 21 destinations.

Compare DDD_dev against the parametric equivalent on the SAME restricted sample:
    lag 3, parametric China terms -> +0.0335 (p=0.016)
If the coefficient barely moves, the parametric terms were adequate. If it moves,
the linear-in-dec assumption was doing real work.
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

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i_int'] = raw['i'].astype('int32'); raw['t_int'] = raw['t'].astype('int32')
raw[MEASURE] = pd.to_numeric(raw[MEASURE], errors='coerce').fillna(0).astype('float32')
lookup0 = (raw[['i_int','ISIC4c','t_int',MEASURE]]
           .drop_duplicates(subset=['i_int','ISIC4c','t_int']))
raw = raw[raw['j'].astype('int32').isin(DEST)].copy()
log(f"restricted sample: {raw.shape}")

for LAG in [3, 0]:
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

    # China x US gets its own (sector, year) cell; everything else shares level 0.
    chn_us = ((d['i_int'] == CHINA) & (d['US_trade'] == 1)).values
    st_grp = d.groupby(['ISIC4c','t'], observed=True).ngroup().astype('int32').values
    fe_chn_us = np.where(chn_us, st_grp + 1, 0).astype('int32')

    m = pd.DataFrame({
        'imports':    d['imports'].astype('float32').values,
        'DDD_dev':    (dec * ip * us * dev).values,
        'DDD_adv':    (dec * ip * us * adv).values,
        'IPxUS_dev':  (ip * us * dev).values,
        'IPxUS_adv':  (ip * us * adv).values,
        'Adv_Dec_US': (adv * dec * us).values,
        'fe_ist': d.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
        'fe_jst': d.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
        'fe_ij':  d.groupby(['i','j'], observed=True).ngroup().astype('int32').values,
        'fe_chn_us': fe_chn_us,
        'cl_is':  d.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32').values,
    })
    del d, ip, china, adv, dev, us, dec
    gc.collect()
    log(f"lag {LAG}: {m.shape} | China-US FE levels: {m['fe_chn_us'].nunique():,} "
        f"(China-US obs: {chn_us.sum():,})")

    f = ("imports ~ DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + Adv_Dec_US "
         "| fe_ist + fe_jst + fe_ij + fe_chn_us")
    try:
        fit = pf.fepois(f, data=m, vcov={"CRV1": "cl_is"},
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                        lean=True, store_data=False, copy_data=False)
        t = fit.tidy()
        print(f"\n### OPTION A -- China absorbed by FE, lag {LAG}   N={fit._N:,}")
        print(t.round(5).to_string())
        t.to_csv(f"china_fe_lag{LAG}.csv")
        log(f"lag {LAG}: DDD_dev = {t.loc['DDD_dev','Estimate']:+.4f} "
            f"(p={t.loc['DDD_dev','Pr(>|t|)']:.4f})   [parametric lag3 was +0.0335, p=0.016]")
        del fit
    except Exception as e:
        log(f"lag {LAG} FAILED: {type(e).__name__}: {e}")
    del m
    gc.collect()

log("DONE")
