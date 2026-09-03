import pandas as pd
import numpy as np
import pyfixest as pf
import time

t0 = time.time()
df = pd.read_pickle("trade_df_isic.pkl")
print(f"[{time.time()-t0:.0f}s] loaded: {df.shape}")

# --- clean sector codes: drop blank, merge ' old' variants into base code ---
df = df[df['ISIC4'] != ''].copy()
df['ISIC4c'] = df['ISIC4'].str.replace(' old', '', regex=False)

# --- coerce n_policies (object) to numeric, treat NaN as 0 (provisional assumption) ---
df['n_policies'] = pd.to_numeric(df['n_policies'], errors='coerce')
for v in ['n_policies','n_sub','frac_policies','frac_sub',
          'share_n_policies','share_n_sub','share_frac_policies','share_frac_sub']:
    df[v] = df[v].fillna(0)

# --- collapse duplicate (i,j,t,ISIC4c) rows created by merging old/new sector codes ---
# imports: sum. policy vars are ist-level (constant across j) -> take max within (i,ISIC4c,t) group after collapse.
agg = df.groupby(['i','j','t','ISIC4c'], as_index=False).agg(
    imports=('imports','sum'),
    mkt_share=('mkt_share','max'),          # will recompute properly below anyway for Decouple construction
    n_policies=('n_policies','max'),
    n_sub=('n_sub','max'),
    frac_policies=('frac_policies','max'),
    frac_sub=('frac_sub','max'),
    US_trade=('US_trade','max'),
)
print(f"[{time.time()-t0:.0f}s] collapsed: {agg.shape}")

# --- Decouple_s: China(156) -> USA(842) mkt_share, sector-level, 2019 vs 2024 (STATIC, per user's original spec) ---
cn_us = df[(df['i']==156) & (df['j']==842)][['ISIC4c','t','mkt_share']]
piv = cn_us.pivot_table(index='ISIC4c', columns='t', values='mkt_share', aggfunc='max')
decouple = ((piv[2024] < piv[2019]).astype(int)).rename('Decouple_s')
print(f"[{time.time()-t0:.0f}s] Decouple_s built for {decouple.shape[0]} sectors; "
      f"{decouple.sum()} decoupling, {(1-decouple).sum()} not (of sectors with both 2019 & 2024 China->US data)")

agg = agg.merge(decouple, left_on='ISIC4c', right_index=True, how='left')
agg['Decouple_s'] = agg['Decouple_s'].fillna(0).astype(int)  # sectors w/o China-US 2019/2024 data treated as 0 (flag)

# --- construct regressors (provisional: IP = n_policies) ---
agg['IP_ist'] = agg['n_policies']
agg['USdest'] = agg['US_trade']
agg['IP_x_US'] = agg['IP_ist'] * agg['USdest']
agg['DDD'] = agg['Decouple_s'] * agg['IP_ist'] * agg['USdest']

# --- FE keys ---
agg['ist'] = agg['i'].astype(str) + "_" + agg['ISIC4c'] + "_" + agg['t'].astype(str)
agg['jst'] = agg['j'].astype(str) + "_" + agg['ISIC4c'] + "_" + agg['t'].astype(str)
agg['ijs'] = agg['i'].astype(str) + "_" + agg['j'].astype(str) + "_" + agg['ISIC4c']

print(f"[{time.time()-t0:.0f}s] N obs for estimation: {agg.shape[0]}")
print("DDD nonzero obs:", (agg['DDD']!=0).sum())
print("IP_x_US nonzero obs:", (agg['IP_x_US']!=0).sum())

agg.to_pickle("agg_for_estimation.pkl")
print(f"[{time.time()-t0:.0f}s] saved agg_for_estimation.pkl")

fit = pf.fepois(
    "imports ~ DDD + IP_x_US | ist + jst + ijs",
    data=agg,
    vcov={"CRV1": "ist"},
)
print(f"[{time.time()-t0:.0f}s] model fit done")
print(fit.summary())
fit.tidy().to_csv("ppml_provisional_results.csv")
print(f"[{time.time()-t0:.0f}s] DONE")
