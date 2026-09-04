import pandas as pd
import numpy as np
import pyfixest as pf
import time

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

CHINA = 156
PRE_YEARS = [2015, 2016, 2017]
DEST = {842,276,251,380,826,528,724,56,392,124,410,36,752,40,208,246,372,620,300,579,757,554}

agg = pd.read_pickle("agg_for_estimation.pkl")
log(f"loaded: {agg.shape}")

agg['i_int'] = agg['i'].astype('int32')
agg['t_int'] = agg['t'].astype('int32')
agg = agg[agg['j'].astype('int32').isin(DEST) & (agg['i_int'] != CHINA)].copy()
log(f"restricted dests + China excluded: {agg.shape}")

agg['fe_ist'] = agg.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32')
agg['fe_jst'] = agg.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32')
agg['fe_ij']  = agg.groupby(['i','j'], observed=True).ngroup().astype('int32')
agg['fe_ijs'] = agg.groupby(['i','j','ISIC4c'], observed=True).ngroup().astype('int32')
agg['cl_is']  = agg.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32')
log("FE + cluster keys built")

d = 'frac_policies'
dev = (1 - agg['Advanced_i']).astype('float32')
adv = agg['Advanced_i'].astype('float32')
us  = agg['US_trade'].astype('float32')
dec = agg['Decouple_intensity_st'].astype('float32')

# pre-trade-war IP stance at (i,s)
pre = (agg[agg['t_int'].isin(PRE_YEARS)]
       .groupby(['i','ISIC4c'], observed=True)[d].mean().rename('IP_pre'))
agg = agg.merge(pre, left_on=['i','ISIC4c'], right_index=True, how='left')
agg['IP_pre'] = agg['IP_pre'].fillna(0).astype('float32')

# binary IP treatment (benchmark's baseline is a dummy, not a count)
agg['IP_bin'] = (agg[d] > 0).astype('float32')

for tag, ipvar in [('pre', agg['IP_pre']), ('bin', agg['IP_bin'])]:
    agg[f'DDD_{tag}_dev'] = (dec * ipvar * us * dev).astype('float32')
    agg[f'DDD_{tag}_adv'] = (dec * ipvar * us * adv).astype('float32')
    agg[f'IPxUS_{tag}_dev'] = (ipvar * us * dev).astype('float32')
    agg[f'IPxUS_{tag}_adv'] = (ipvar * us * adv).astype('float32')
log(f"IP_pre nonzero {(agg['IP_pre']>0).mean()*100:.1f}% | IP_bin nonzero {(agg['IP_bin']>0).mean()*100:.1f}%")

BASE = f"DDD2_{d}_dev + DDD2_{d}_adv + {d}_x_US_dev + {d}_x_US_adv + Adv_x_Decouple_x_US"

VARIANTS = [
    ("A. baseline improved (China out, cluster i,s)",
     f"imports ~ {BASE} | fe_ist + fe_jst + fe_ij"),
    ("C. pre-trade-war IP (2015-17), addresses reverse causality",
     "imports ~ DDD_pre_dev + DDD_pre_adv + IPxUS_pre_dev + IPxUS_pre_adv + Adv_x_Decouple_x_US | fe_ist + fe_jst + fe_ij"),
    ("D. binary IP treatment (benchmark-style dummy)",
     "imports ~ DDD_bin_dev + DDD_bin_adv + IPxUS_bin_dev + IPxUS_bin_adv + Adv_x_Decouple_x_US | fe_ist + fe_jst + fe_ij"),
    ("B. sector-specific pair FE (alpha_ijs, matches benchmark alpha_ijk)",
     f"imports ~ {BASE} | fe_ist + fe_jst + fe_ijs"),
]

for name, formula in VARIANTS:
    log(f"=== {name} ===")
    try:
        fit = pf.fepois(formula, data=agg, vcov={"CRV1": "cl_is"},
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                        lean=True, store_data=False, copy_data=False)
        print(f"\n### {name}\nN = {fit._N:,}  |  {formula}  |  cluster (i,s)")
        print(fit.tidy().round(5).to_string())
        fit.tidy().to_csv(f"gravity_improved_{name.split('.')[0]}.csv")
        log(f"{name}: done")
        del fit
    except Exception as e:
        log(f"{name}: FAILED -- {e}")

log("ALL DONE")
