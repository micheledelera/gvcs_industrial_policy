import pandas as pd
import numpy as np
import pyfixest as pf
import gc
import time

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

# IMF WEO "Advanced Economies" list, mapped to COMTRADE/BACI numeric reporter codes.
# NB: BACI codes mostly match ISO-3166 numeric but with some known exceptions
# (e.g. USA=842 not 840, confirmed empirically earlier; France=251 not 250 in
# COMTRADE-derived data). Verified against codes actually present in `i` below.
ADVANCED_CODES = {
    36: "Australia", 40: "Austria", 56: "Belgium", 124: "Canada", 196: "Cyprus",
    203: "Czech Republic", 208: "Denmark", 233: "Estonia", 246: "Finland",
    251: "France", 276: "Germany", 300: "Greece", 344: "Hong Kong SAR",
    352: "Iceland", 372: "Ireland", 376: "Israel", 380: "Italy", 392: "Japan",
    410: "Korea, Rep.", 428: "Latvia", 440: "Lithuania", 442: "Luxembourg",
    446: "Macao SAR", 470: "Malta", 528: "Netherlands", 554: "New Zealand",
    579: "Norway", 620: "Portugal", 630: "Puerto Rico", 674: "San Marino",
    702: "Singapore", 703: "Slovak Republic", 705: "Slovenia", 724: "Spain",
    752: "Sweden", 757: "Switzerland", 490: "Taiwan (Other Asia, nes)",
    826: "United Kingdom", 842: "United States",
}
# NB: 757 (not ISO-numeric 756) for Switzerland, 579 (not 578) for Norway --
# both confirmed against codes actually present in the data. See patch_advanced.py.

df = pd.read_pickle("trade_df_isic.pkl")
log(f"loaded: {df.shape}")

present = set(df['i'].unique())
missing_codes = {k: v for k, v in ADVANCED_CODES.items() if k not in present}
log(f"Advanced-economy codes NOT found in data (check these!): {missing_codes}")

# --- sector cleaning: drop blank ISIC4 AND drop ' old'-suffixed codes entirely ---
df = df[(df['ISIC4'] != '') & (~df['ISIC4'].str.contains(' old'))].copy()
df['ISIC4c'] = df['ISIC4'].astype('category')
df.drop(columns=['ISIC4'], inplace=True)

# --- sanity check: my mkt_share-based Decouple_s vs the dataset's own `target` flag ---
cn_us = df.loc[(df['i']==156) & (df['j']==842), ['ISIC4c','t','mkt_share']]
piv = cn_us.pivot_table(index='ISIC4c', columns='t', values='mkt_share', aggfunc='max', observed=True)
my_decouple = ((piv[2024] < piv[2019]).astype('int8')).rename('my_Decouple_s')
tgt = df.groupby('ISIC4c', observed=True)['target'].first().rename('target_s')
cmp_tab = pd.concat([my_decouple, tgt], axis=1).dropna()
log(f"cross-check my_Decouple_s (mkt_share 2019->2024) vs target: \n{pd.crosstab(cmp_tab['my_Decouple_s'], cmp_tab['target_s'])}")
del cn_us, piv, my_decouple, tgt, cmp_tab
gc.collect()

# --- dtypes ---
for v in ['n_policies']:
    df[v] = pd.to_numeric(df[v], errors='coerce').astype('float32')
for v in ['n_sub','frac_policies','frac_sub','target']:
    df[v] = df[v].astype('float32')
df[['n_policies','n_sub','frac_policies','frac_sub']] = df[['n_policies','n_sub','frac_policies','frac_sub']].fillna(0)
df['i'] = df['i'].astype('int32')
df['j'] = df['j'].astype('int32')
df['t'] = df['t'].astype('int16')
df['US_trade'] = df['US_trade'].astype('int8')
df['imports'] = df['imports'].astype('float32')
df['Advanced_i'] = df['i'].isin(ADVANCED_CODES.keys()).astype('int8')
log(f"dtypes optimized, memory MB: {df.memory_usage(deep=True).sum()/1e6:.0f}")
log(f"Advanced-economy exporter rows: {df['Advanced_i'].mean()*100:.1f}% of obs")

# --- collapse to (i,j,t,ISIC4c) -- should already be unique now that 'old' dupes are dropped, but sum just in case ---
agg = df.groupby(['i','j','t','ISIC4c'], as_index=False, observed=True).agg(
    imports=('imports','sum'),
    n_policies=('n_policies','max'),
    n_sub=('n_sub','max'),
    frac_policies=('frac_policies','max'),
    target=('target','max'),          # Decouple_s per Michele: sector pulling away from China in the US market
    US_trade=('US_trade','max'),
    Advanced_i=('Advanced_i','max'),
)
del df
gc.collect()
log(f"collapsed: {agg.shape}, memory MB: {agg.memory_usage(deep=True).sum()/1e6:.0f}")

agg['ist_cluster'] = agg.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32')

IP_VARS = ['n_policies', 'n_sub', 'frac_policies']
results = {}
for ip_var in IP_VARS:
    d = ip_var
    agg[f'{d}_x_US_dev'] = (agg[ip_var] * agg['US_trade'] * (1 - agg['Advanced_i'])).astype('float32')
    agg[f'{d}_x_US_adv'] = (agg[ip_var] * agg['US_trade'] * agg['Advanced_i']).astype('float32')
    agg[f'DDD_{d}_dev'] = (agg['target'] * agg[ip_var] * agg['US_trade'] * (1 - agg['Advanced_i'])).astype('float32')
    agg[f'DDD_{d}_adv'] = (agg['target'] * agg[ip_var] * agg['US_trade'] * agg['Advanced_i']).astype('float32')

agg['i'] = agg['i'].astype('category')
agg['j'] = agg['j'].astype('category')
agg['t'] = agg['t'].astype('category')
log(f"N obs for estimation: {agg.shape[0]}, memory MB: {agg.memory_usage(deep=True).sum()/1e6:.0f}")

agg.to_pickle("agg_for_estimation.pkl")
log("saved agg_for_estimation.pkl")

for ip_var in IP_VARS:
    d = ip_var
    formula = (f"imports ~ DDD_{d}_dev + DDD_{d}_adv + {d}_x_US_dev + {d}_x_US_adv "
               f"| i^ISIC4c^t + j^ISIC4c^t + i^j")
    log(f"fitting: {ip_var} ...")
    fit = pf.fepois(formula, data=agg, vcov={"CRV1": "ist_cluster"})
    log(f"[{ip_var}] fit done")
    print(fit.summary())
    fit.tidy().to_csv(f"ppml_{ip_var}_results.csv")
    del fit
    gc.collect()

log("DONE")
