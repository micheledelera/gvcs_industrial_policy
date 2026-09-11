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
# China STAYS in the sample -- it keeps disciplining alpha_jst and keeps the
# sector-year totals real. It is removed only from the coefficient of interest,
# by giving it its own terms.
agg = agg[agg['j'].astype('int32').isin(DEST)].copy()
log(f"restricted dests (China retained): {agg.shape}")

agg['fe_ist'] = agg.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32')
agg['fe_jst'] = agg.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32')
agg['fe_ij']  = agg.groupby(['i','j'], observed=True).ngroup().astype('int32')
agg['fe_ijs'] = agg.groupby(['i','j','ISIC4c'], observed=True).ngroup().astype('int32')
agg['cl_is']  = agg.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32')
log("FE + cluster keys built")

d = 'frac_policies'
china = (agg['i_int'] == CHINA).astype('float32')
adv   = agg['Advanced_i'].astype('float32')
dev   = ((1 - agg['Advanced_i']) * (1 - china)).astype('float32')   # developing EX-CHINA
us    = agg['US_trade'].astype('float32')
dec   = agg['Decouple_intensity_st'].astype('float32')
log(f"group shares -- China {china.mean()*100:.2f}% | dev ex-China {dev.mean()*100:.1f}% | advanced {adv.mean()*100:.1f}%")

# pre-trade-war IP stance at (i,s), and binary IP (benchmark's baseline is a dummy)
pre = (agg[agg['t_int'].isin(PRE_YEARS)]
       .groupby(['i','ISIC4c'], observed=True)[d].mean().rename('IP_pre'))
agg = agg.merge(pre, left_on=['i','ISIC4c'], right_index=True, how='left')
agg['IP_pre'] = agg['IP_pre'].fillna(0).astype('float32')
agg['IP_bin'] = (agg[d] > 0).astype('float32')

for tag, ip in [('frac', agg[d].astype('float32')),
                ('pre',  agg['IP_pre']),
                ('bin',  agg['IP_bin'])]:
    agg[f'DDD_{tag}_dev'] = (dec * ip * us * dev).astype('float32')
    agg[f'DDD_{tag}_adv'] = (dec * ip * us * adv).astype('float32')
    agg[f'IPxUS_{tag}_dev'] = (ip * us * dev).astype('float32')
    agg[f'IPxUS_{tag}_adv'] = (ip * us * adv).astype('float32')
    agg[f'IPxUS_{tag}_china'] = (ip * us * china).astype('float32')

# China's own decoupling term: the validation / first-stage check.
# If Decouple_intensity_st measures what we think, this should be strongly negative.
agg['Dec_x_US_China'] = (dec * us * china).astype('float32')
agg['Adv_x_Dec_x_US'] = (adv * dec * us).astype('float32')

def terms(tag):
    return (f"Dec_x_US_China + DDD_{tag}_dev + DDD_{tag}_adv + "
            f"IPxUS_{tag}_dev + IPxUS_{tag}_adv + IPxUS_{tag}_china + Adv_x_Dec_x_US")

VARIANTS = [
    ("A", "baseline, China own term, cluster (i,s)",
     f"imports ~ {terms('frac')} | fe_ist + fe_jst + fe_ij"),
    ("C", "pre-trade-war IP (2015-17)",
     f"imports ~ {terms('pre')} | fe_ist + fe_jst + fe_ij"),
    ("D", "binary IP (benchmark-style dummy)",
     f"imports ~ {terms('bin')} | fe_ist + fe_jst + fe_ij"),
    ("B", "sector-specific pair FE (alpha_ijs)",
     f"imports ~ {terms('frac')} | fe_ist + fe_jst + fe_ijs"),
]

for tag, name, formula in VARIANTS:
    log(f"=== {tag}. {name} ===")
    try:
        fit = pf.fepois(formula, data=agg, vcov={"CRV1": "cl_is"},
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                        lean=True, store_data=False, copy_data=False)
        print(f"\n### {tag}. {name}\nN = {fit._N:,}  |  cluster (i,s)\n{formula}")
        print(fit.tidy().round(5).to_string())
        fit.tidy().to_csv(f"gravity_improved_{tag}.csv")
        log(f"{tag}: done")
        del fit
    except Exception as e:
        log(f"{tag}: FAILED -- {e}")

log("ALL DONE")
