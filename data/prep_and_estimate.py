import pandas as pd
import numpy as np
import pyfixest as pf
import gc
import time

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

df = pd.read_pickle("trade_df_isic.pkl")
log(f"loaded: {df.shape}")

# --- clean sector codes: drop blank, merge ' old' variants into base code ---
df = df[df['ISIC4'] != ''].copy()
df['ISIC4c'] = df['ISIC4'].str.replace(' old', '', regex=False).astype('category')
df.drop(columns=['ISIC4'], inplace=True)

# --- coerce n_policies (object) to numeric, treat NaN as 0 (provisional assumption) ---
df['n_policies'] = pd.to_numeric(df['n_policies'], errors='coerce').astype('float32')
for v in ['n_sub','frac_policies','frac_sub',
          'share_n_policies','share_n_sub','share_frac_policies','share_frac_sub','mkt_share']:
    df[v] = df[v].astype('float32')
df[['n_policies','n_sub','frac_policies','frac_sub']] = df[['n_policies','n_sub','frac_policies','frac_sub']].fillna(0)

df['i'] = df['i'].astype('int32')
df['j'] = df['j'].astype('int32')
df['t'] = df['t'].astype('int16')
df['US_trade'] = df['US_trade'].astype('int8')
df['imports'] = df['imports'].astype('float32')
log(f"dtypes optimized, memory MB: {df.memory_usage(deep=True).sum()/1e6:.0f}")

# --- Decouple_s: China(156) -> USA(842) mkt_share, sector-level, 2019 vs 2024 (STATIC, per user's original spec) ---
cn_us = df.loc[(df['i']==156) & (df['j']==842), ['ISIC4c','t','mkt_share']]
piv = cn_us.pivot_table(index='ISIC4c', columns='t', values='mkt_share', aggfunc='max', observed=True)
decouple = ((piv[2024] < piv[2019]).astype('int8')).rename('Decouple_s')
log(f"Decouple_s built for {decouple.shape[0]} sectors; {decouple.sum()} decoupling, {(1-decouple).sum()} not")
del cn_us, piv
gc.collect()

# --- collapse duplicate (i,j,t,ISIC4c) rows created by merging old/new sector codes ---
agg = df.groupby(['i','j','t','ISIC4c'], as_index=False, observed=True).agg(
    imports=('imports','sum'),
    n_policies=('n_policies','max'),
    US_trade=('US_trade','max'),
)
del df
gc.collect()
log(f"collapsed: {agg.shape}, memory MB: {agg.memory_usage(deep=True).sum()/1e6:.0f}")

agg = agg.merge(decouple, left_on='ISIC4c', right_index=True, how='left')
agg['Decouple_s'] = agg['Decouple_s'].fillna(0).astype('int8')

# --- construct regressors (provisional: IP = n_policies) ---
agg['IP_ist'] = agg['n_policies']
agg['IP_x_US'] = (agg['IP_ist'] * agg['US_trade']).astype('float32')
agg['DDD'] = (agg['Decouple_s'] * agg['IP_ist'] * agg['US_trade']).astype('float32')

agg['ist_cluster'] = agg.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32')

agg['i'] = agg['i'].astype('category')
agg['j'] = agg['j'].astype('category')
agg['t'] = agg['t'].astype('category')

log(f"N obs for estimation: {agg.shape[0]}, memory MB: {agg.memory_usage(deep=True).sum()/1e6:.0f}")
log(f"DDD nonzero obs: {(agg['DDD']!=0).sum()}, IP_x_US nonzero obs: {(agg['IP_x_US']!=0).sum()}")

agg.to_pickle("agg_for_estimation.pkl")
log("saved agg_for_estimation.pkl")

fit = pf.fepois(
    "imports ~ DDD + IP_x_US | i^ISIC4c^t + j^ISIC4c^t + i^j^ISIC4c",
    data=agg,
    vcov={"CRV1": "ist_cluster"},
)
log("model fit done")
print(fit.summary())
fit.tidy().to_csv("ppml_provisional_results.csv")
log("DONE")
