"""Is IP coverage stable enough over time for a year-by-year event study?"""
import pandas as pd, numpy as np

CHINA, LAG, MEASURE = 156, 3, 'share_frac_policies'
DEST = {842,276,251,380,826,528,724,56,392,124,410,36,752,40,208,246,372,620,300,579,757,554}

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i_int'] = raw['i'].astype('int32'); raw['t_int'] = raw['t'].astype('int32')
raw[MEASURE] = pd.to_numeric(raw[MEASURE], errors='coerce').fillna(0).astype('float32')
lk = raw[['i_int','ISIC4c','t_int',MEASURE]].drop_duplicates(subset=['i_int','ISIC4c','t_int'])
lk['t_int'] += LAG
lk = lk.rename(columns={MEASURE: 'IP_lag'})
raw = raw[raw['j'].astype('int32').isin(DEST)].copy()
d = raw.merge(lk, on=['i_int','ISIC4c','t_int'], how='left')
d = d[d['IP_lag'].notna()]
del raw, lk

dev = (d['Advanced_i'] == 0) & (d['i_int'] != CHINA)
us  = d['US_trade'] == 1
cells = (d[dev & us][['i_int','ISIC4c','t_int','IP_lag','target']]
         .drop_duplicates(subset=['i_int','ISIC4c','t_int']))
pooled_sd = float(d['IP_lag'].std())

g = cells.groupby('t_int').agg(
        cells=('IP_lag','size'), nonzero=('IP_lag', lambda x: int((x > 0).sum())),
        mean=('IP_lag','mean'), sd=('IP_lag','std'), mx=('IP_lag','max'))
g['pct_nonzero'] = 100 * g['nonzero'] / g['cells']
g['sd_rel_pooled'] = g['sd'] / pooled_sd
tg = cells[cells['target'] == 1].groupby('t_int')['IP_lag'].agg(
        tgt_cells='size', tgt_nonzero=lambda x: int((x > 0).sum()))
g = g.join(tg)

print(f"US-bound developing (ex-China) exporter-sector cells, IP lagged {LAG}")
print(f"pooled SD used for standardisation: {pooled_sd:.6f}\n")
print(g[['cells','nonzero','pct_nonzero','sd_rel_pooled','tgt_cells','tgt_nonzero']]
      .round(3).to_string())
