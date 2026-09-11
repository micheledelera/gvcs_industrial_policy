import pandas as pd
import numpy as np
import pyfixest as pf
import time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, LAG = 156, 3
MEASURE = 'share_frac_policies'
DEST = {842,276,251,380,826,528,724,56,392,124,410,36,752,40,208,246,372,620,300,579,757,554}

agg = pd.read_pickle("agg_for_estimation.pkl")
agg['i_int'] = agg['i'].astype('int32'); agg['t_int'] = agg['t'].astype('int32')
agg[MEASURE] = pd.to_numeric(agg[MEASURE], errors='coerce').fillna(0).astype('float32')

look = (agg[['i_int','ISIC4c','t_int',MEASURE]]
        .drop_duplicates(subset=['i_int','ISIC4c','t_int']))
look['t_int'] = look['t_int'] + LAG
look = look.rename(columns={MEASURE: 'IP_lag'})

agg = agg[agg['j'].astype('int32').isin(DEST)].copy()   # China RETAINED, own terms
agg = agg.merge(look, on=['i_int','ISIC4c','t_int'], how='left')
agg = agg[agg['IP_lag'].notna()].copy()
log(f"restricted dests, lag{LAG} merged: {agg.shape}")

agg['IP_z'] = (agg['IP_lag'] / agg['IP_lag'].std()).astype('float32')
agg['fe_ist'] = agg.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32')
agg['fe_jst'] = agg.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32')
agg['fe_ij']  = agg.groupby(['i','j'], observed=True).ngroup().astype('int32')
agg['cl_is']  = agg.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32')

china = (agg['i_int'] == CHINA).astype('float32')
adv   = agg['Advanced_i'].astype('float32')
dev   = ((1 - agg['Advanced_i']) * (1 - china)).astype('float32')
us    = agg['US_trade'].astype('float32')
post  = (agg['t_int'] >= 2018).astype('float32')
tgt   = agg['target'].astype('float32')
dec   = (tgt * post).astype('float32')          # author's original treatment

agg['DDD_dev']    = (dec * agg['IP_z'] * us * dev).astype('float32')
agg['DDD_adv']    = (dec * agg['IP_z'] * us * adv).astype('float32')
agg['IPxUS_dev']  = (agg['IP_z'] * us * dev).astype('float32')
agg['IPxUS_adv']  = (agg['IP_z'] * us * adv).astype('float32')
agg['IPxUS_chn']  = (agg['IP_z'] * us * china).astype('float32')
agg['Dec_US_chn'] = (dec * us * china).astype('float32')
agg['Adv_Dec_US'] = (adv * dec * us).astype('float32')

f = ("imports ~ DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + IPxUS_chn "
     "+ Dec_US_chn + Adv_Dec_US | fe_ist + fe_jst + fe_ij")
log(f"fitting gravity with {MEASURE} (lag {LAG})")
log(f)

fit = pf.fepois(f, data=agg, vcov={"CRV1": "cl_is"},
                demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                lean=True, store_data=False, copy_data=False)
log("fit done")
print(f"\n### GRAVITY, {MEASURE} lag {LAG}   N={fit._N:,}   cluster (i,s)")
print(fit.tidy().round(5).to_string())
fit.tidy().to_csv("gravity_share_results.csv")
log("DONE")
