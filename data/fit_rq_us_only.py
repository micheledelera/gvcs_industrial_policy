import pandas as pd
import numpy as np
import pyfixest as pf
import time

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

CHINA = 156
PRE_YEARS = [2015, 2016, 2017]

agg = pd.read_pickle("agg_for_estimation.pkl")
log(f"loaded agg: {agg.shape}")

us = agg[agg['US_trade'] == 1].copy()
us['i_int'] = us['i'].astype('int32')
us['t_int'] = us['t'].astype('int32')
log(f"US-bound obs: {len(us):,}")

us = us[us['i_int'] != CHINA].copy()
log(f"after dropping China: {len(us):,}  ({us['i_int'].nunique()} exporters)")

us['fe_is'] = us.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32')
us['fe_st'] = us.groupby(['ISIC4c','t'], observed=True).ngroup().astype('int32')
us['Developing_i'] = (1 - us['Advanced_i']).astype('int8')
us['Dec'] = us['Decouple_intensity_st'].astype('float32')

# pre-trade-war IP stance, (i,s) level, from 2015-17
pre = (us[us['t_int'].isin(PRE_YEARS)]
       .groupby(['i','ISIC4c'], observed=True)['frac_policies'].mean()
       .rename('IP_pre'))
us = us.merge(pre, left_on=['i','ISIC4c'], right_index=True, how='left')
us['IP_pre'] = us['IP_pre'].fillna(0).astype('float32')
log(f"IP_pre built: mean {us['IP_pre'].mean():.4f}, nonzero share {(us['IP_pre']>0).mean()*100:.1f}%")

us['Dec_x_Dev']    = (us['Dec'] * us['Developing_i']).astype('float32')
us['Dec_x_IP']     = (us['Dec'] * us['frac_policies']).astype('float32')
us['Dec_x_IPpre']  = (us['Dec'] * us['IP_pre']).astype('float32')

dev = us[us['Developing_i'] == 1].copy()
log(f"developing ex-China sample: {len(dev):,} obs, {dev['i_int'].nunique()} exporters")

def run(name, formula, data, cl="i_int"):
    log(f"--- {name} ---")
    try:
        fit = pf.fepois(formula, data=data, vcov={"CRV1": cl},
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=2000))
        print(f"\n### {name}\nN = {fit._N:,}   formula: {formula}   cluster: {cl}")
        print(fit.tidy().round(5).to_string())
        return fit.tidy()
    except Exception as e:
        log(f"FAILED: {e}")
        return None

out = {}
out['RQ1'] = run("RQ1: decoupling -> developing share of US sourcing",
                 "imports ~ Dec_x_Dev | fe_is + fe_st", us)

out['RQ2_contemp'] = run("RQ2 (contemporaneous IP): does IP pay off more as decoupling intensifies",
                         "imports ~ frac_policies + Dec_x_IP | fe_is + fe_st", dev)

out['RQ2_pre'] = run("RQ2 (PRE-trade-war IP, causal version)",
                     "imports ~ Dec_x_IPpre | fe_is + fe_st", dev)

# event study: pre-period IP interacted with year, base year 2017
dev = dev.copy()
for yr in sorted(dev['t_int'].unique()):
    if yr == 2017:
        continue
    dev[f'IPpre_y{yr}'] = (dev['IP_pre'] * (dev['t_int'] == yr)).astype('float32')
ev_terms = " + ".join(f"IPpre_y{yr}" for yr in sorted(dev['t_int'].unique()) if yr != 2017)
out['RQ2_event'] = run("RQ2 event study: IP_pre x year (base 2017)",
                       f"imports ~ {ev_terms} | fe_is + fe_st", dev)

for k, v in out.items():
    if v is not None:
        v.to_csv(f"rq_us_only_{k}.csv")
log("DONE")
