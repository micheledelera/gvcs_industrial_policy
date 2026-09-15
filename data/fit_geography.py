"""
Does the result survive controls for time-varying bilateral advantage?

alpha_ij absorbs the LEVEL of a country's US relationship (distance, contiguity
with China, existing supply-chain ties) but is time-invariant, so it cannot
absorb any CHANGE in the return to those features. Decoupling is precisely a
shock to the return to proximity -- to China (relocation feasibility) or to the
US (nearshoring). Nothing else in the FE structure catches it either: alpha_ist
absorbs a country's GLOBAL sectoral exports, so it catches a proximity advantage
that lifts sales everywhere, but not one that is US-specific.

Fixes, in increasing strength:
  B  alpha_ij x post  -- absorbs any post-2018 shift in the bilateral relationship
  C  alpha_ijt        -- absorbs ANY year-specific bilateral shock, whatever its
                         source (geography, nearshoring, USMCA, agreements)
Identification survives because DDD_dev varies across SECTORS within each
(i, j, t) cell while these FE are constant within it.

  D  drop Vietnam (704) and Mexico (484) -- the named-cases check readers will ask
     for. Weaker than C as a design (why those two and not Thailand?) but direct.
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
VIETNAM, MEXICO = 704, 484

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i_int'] = raw['i'].astype('int32'); raw['t_int'] = raw['t'].astype('int32')
raw[MEASURE] = pd.to_numeric(raw[MEASURE], errors='coerce').fillna(0).astype('float32')
lk = (raw[['i_int','ISIC4c','t_int',MEASURE]]
      .drop_duplicates(subset=['i_int','ISIC4c','t_int']))
lk['t_int'] = lk['t_int'] + LAG
lk = lk.rename(columns={MEASURE: 'IP_lag'})

raw = raw[raw['j'].astype('int32').isin(DEST)].copy()
raw = raw.merge(lk, on=['i_int','ISIC4c','t_int'], how='left')
raw = raw[raw['IP_lag'].notna()].copy()
log(f"restricted, lag{LAG}: {raw.shape}")

VARIANTS = [
    ("A. baseline (alpha_ij)",            "fe_ist + fe_jst + fe_ij",      False),
    ("B. alpha_ij x post",                "fe_ist + fe_jst + fe_ij_post", False),
    ("C. alpha_ijt (pair-year)",          "fe_ist + fe_jst + fe_ijt",     False),
    ("D. baseline, drop Vietnam+Mexico",  "fe_ist + fe_jst + fe_ij",      True),
]

for name, fes, drop_vm in VARIANTS:
    d = raw[~raw['i_int'].isin([VIETNAM, MEXICO])] if drop_vm else raw
    ip    = (d['IP_lag'] / d['IP_lag'].std()).astype('float32')
    china = (d['i_int'] == CHINA).astype('float32')
    adv   = d['Advanced_i'].astype('float32')
    dev   = ((1 - d['Advanced_i']) * (1 - china)).astype('float32')
    us    = d['US_trade'].astype('float32')
    post  = (d['t_int'] >= 2018)
    dec   = (d['target'] * post).astype('float32')

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
    })
    m['fe_ij_post'] = (m['fe_ij'] * 2 + post.values.astype('int32')).astype('int32')
    del ip, china, adv, dev, us, dec
    gc.collect()
    log(f"=== {name} ===  N={len(m):,}  FE levels: {fes.split(' + ')[-1]}="
        f"{m[fes.split(' + ')[-1]].nunique():,}")

    f = ("imports ~ DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + IPxUS_chn "
         f"+ Dec_US_chn + Adv_Dec_US | {fes}")
    try:
        fit = pf.fepois(f, data=m, vcov={"CRV1": "cl_is"},
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                        lean=True, store_data=False, copy_data=False)
        t = fit.tidy()
        print(f"\n### {name}   N={fit._N:,}   cluster (i,s)")
        print(t.round(5).to_string())
        t.to_csv(f"geography_{name[0]}.csv")
        log(f"{name}: DDD_dev = {t.loc['DDD_dev','Estimate']:+.4f} "
            f"(p={t.loc['DDD_dev','Pr(>|t|)']:.4f})")
        del fit
    except Exception as e:
        log(f"{name} FAILED: {type(e).__name__}: {e}")
    del m
    gc.collect()

log("ALL DONE")
