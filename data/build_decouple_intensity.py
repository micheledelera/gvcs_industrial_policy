import pandas as pd
import numpy as np
import gc
import time

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

BASELINE_YEARS = [2015, 2016, 2017]
TRAILING_WINDOW = 3  # years, including current year

df = pd.read_pickle("trade_df_isic.pkl")
df = df[(df['ISIC4'] != '') & (~df['ISIC4'].str.contains(' old'))].copy()
df['ISIC4c'] = df['ISIC4'].astype('category')

cn_us = df.loc[(df['i']==156) & (df['j']==842), ['ISIC4c','t','mkt_share']].copy()
piv = cn_us.pivot_table(index='ISIC4c', columns='t', values='mkt_share', aggfunc='max', observed=True)
log(f"China->USA mkt_share pivot built: {piv.shape}")

baseline_share_s = piv[BASELINE_YEARS].mean(axis=1)
log(f"baseline_share_s (2015-2017 avg) computed for {baseline_share_s.notna().sum()} sectors")

trailing_avg = piv.T.rolling(window=TRAILING_WINDOW, min_periods=1).mean().T
log("trailing average computed")

frac_lost = trailing_avg.subtract(baseline_share_s, axis=0) * -1
frac_lost = frac_lost.div(baseline_share_s.replace(0, np.nan), axis=0)
frac_lost = frac_lost.clip(lower=0).fillna(0)  # baseline=0 -> no meaningful "loss" to measure -> 0

target_s = df.groupby('ISIC4c', observed=True)['target'].first()
decouple_intensity = frac_lost.mul(target_s, axis=0).fillna(0)
log(f"Decouple_intensity_st built: {decouple_intensity.shape}")

di_long = decouple_intensity.stack().rename('Decouple_intensity_st').reset_index()
di_long.columns = ['ISIC4c', 't', 'Decouple_intensity_st']
log(f"long format: {di_long.shape}, nonzero cells: {(di_long['Decouple_intensity_st']>0).sum()}")

# quick sanity print: same three example sectors used earlier
for code in ['1010', '1020', '1030']:
    if code in decouple_intensity.index:
        print(f"--- {code} ---")
        print(decouple_intensity.loc[code].round(4).to_dict())

di_long.to_pickle("decouple_intensity_st.pkl")
log("saved decouple_intensity_st.pkl")

# --- patch agg_for_estimation.pkl ---
agg = pd.read_pickle("agg_for_estimation.pkl")
log(f"loaded agg: {agg.shape}")

di_long['t'] = di_long['t'].astype(agg['t'].dtype if agg['t'].dtype.name != 'category' else 'int64')
agg['t_int'] = agg['t'].astype('int64')
di_long['t'] = di_long['t'].astype('int64')
agg = agg.merge(di_long, left_on=['ISIC4c','t_int'], right_on=['ISIC4c','t'], how='left', suffixes=('','_di'))
agg['Decouple_intensity_st'] = agg['Decouple_intensity_st'].fillna(0).astype('float32')
agg.drop(columns=['t_int'] + ([ 't_di'] if 't_di' in agg.columns else []), inplace=True, errors='ignore')
log(f"merged Decouple_intensity_st onto agg, nonzero obs: {(agg['Decouple_intensity_st']>0).sum()}")

for ip_var in ['n_policies', 'n_sub', 'frac_policies']:
    d = ip_var
    agg[f'DDD2_{d}_dev'] = (agg['Decouple_intensity_st'] * agg[ip_var] * agg['US_trade'] * (1 - agg['Advanced_i'])).astype('float32')
    agg[f'DDD2_{d}_adv'] = (agg['Decouple_intensity_st'] * agg[ip_var] * agg['US_trade'] * agg['Advanced_i']).astype('float32')
agg['Adv_x_Decouple_x_US'] = (agg['Advanced_i'] * agg['Decouple_intensity_st'] * agg['US_trade']).astype('float32')

agg.to_pickle("agg_for_estimation.pkl")
log("saved updated agg_for_estimation.pkl with Decouple_intensity_st terms")
