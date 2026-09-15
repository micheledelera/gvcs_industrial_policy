"""Are all eight policy measures viable before committing overnight compute?

n_sub is known to be nonzero in only ~1.3% of developing ex-China US-bound
observations, so the sub-based measures may be too thin to identify anything.
Check coverage and variation in the cells that actually carry the DDD.
"""
import pandas as pd, numpy as np

CHINA, LAG = 156, 3
MEAS = ['n_policies','n_sub','frac_policies','frac_sub',
        'share_n_policies','share_n_sub','share_frac_policies','share_frac_sub']

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i_int'] = raw['i'].astype('int32'); raw['t_int'] = raw['t'].astype('int32')
for c in MEAS:
    raw[c] = pd.to_numeric(raw[c], errors='coerce').fillna(0).astype('float32')

cells = raw[['i_int','ISIC4c','t_int'] + MEAS].drop_duplicates(
    subset=['i_int','ISIC4c','t_int'])
dev = raw.loc[(raw['Advanced_i'] == 0) & (raw['i_int'] != CHINA), 'i_int'].unique()
tgt = raw.loc[raw['target'] == 1, 'ISIC4c'].unique()
sub = cells[cells['i_int'].isin(dev) & cells['ISIC4c'].isin(tgt) &
            (cells['t_int'] >= 2018 - LAG)]

print(f"developing ex-China x target-sector x (t>=2015) cells: {len(sub):,}\n")
rows = []
for c in MEAS:
    v = sub[c]
    rows.append({'measure': c, 'pct_nonzero': 100*(v > 0).mean(),
                 'mean': v.mean(), 'sd': v.std(),
                 'p90': v.quantile(.90), 'max': v.max(),
                 'sd_over_mean': v.std()/v.mean() if v.mean() else np.nan})
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("\ncorrelations among the eight (in these cells):")
print(sub[MEAS].corr().round(2).to_string())
