"""Rectangularise the panel so entry and exit become visible.

The delivered data holds only positive flows, so a country that stops selling a
sector to the US simply vanishes from the sample. That makes the extensive margin
unobservable and, worse, means every estimate so far is on a selected sample of
positives -- which is exactly the selection PPML with zeros is meant to avoid.

Frame: keep (i, s, t) cells where exporter i ships sector s to SOMEONE (any of the
229 destinations) in year t, then rectangularise those over the 22 advanced
destinations. Filling every (i, j, s, t) cell instead would manufacture structural
zeros -- a country that never makes a product is not "failing to serve the US".

target is sector-level, Advanced_i exporter-level, US_trade destination-level
(verified), so each merges on a single key.
"""
import pandas as pd, numpy as np, gc, time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

LAG, MEASURE, USA = 3, 'share_frac_policies', 842
DEST = np.array(sorted({842,276,251,380,826,528,724,56,392,124,410,36,752,40,208,
                        246,372,620,300,579,757,554}), dtype='int32')

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i_int'] = raw['i'].astype('int32')
raw['j_int'] = raw['j'].astype('int32')
raw['t_int'] = raw['t'].astype('int32')
raw[MEASURE] = pd.to_numeric(raw[MEASURE], errors='coerce').fillna(0).astype('float32')

# lookups, taken before the frame is narrowed
ip_lk = (raw[['i_int','ISIC4c','t_int',MEASURE]]
         .drop_duplicates(subset=['i_int','ISIC4c','t_int']).copy())
ip_lk['t_int'] += LAG
ip_lk = ip_lk.rename(columns={MEASURE: 'IP_lag'})
tgt_lk = raw[['ISIC4c','target']].drop_duplicates(subset=['ISIC4c'])
adv_lk = raw[['i_int','Advanced_i']].drop_duplicates(subset=['i_int'])

act = (raw.loc[raw['imports'] > 0, ['i_int','ISIC4c','t_int']]
       .drop_duplicates().reset_index(drop=True))
obs = (raw.loc[raw['j_int'].isin(DEST) & (raw['imports'] > 0),
               ['i_int','j_int','ISIC4c','t_int','imports']].copy())
obs['imports'] = obs['imports'].astype('float32')
del raw
gc.collect()
log(f"active cells {len(act):,} | observed positives in frame {len(obs):,}")

n, k = len(act), len(DEST)
rect = pd.DataFrame({
    'i_int':  np.repeat(act['i_int'].values, k),
    'ISIC4c': np.repeat(act['ISIC4c'].values, k),
    't_int':  np.repeat(act['t_int'].values, k),
    'j_int':  np.tile(DEST, n),
})
del act
gc.collect()
log(f"rectangularised: {len(rect):,} rows")

rect = rect.merge(obs, on=['i_int','j_int','ISIC4c','t_int'], how='left')
del obs
gc.collect()
n_pos = int(rect['imports'].notna().sum())
rect['imports'] = rect['imports'].fillna(0).astype('float32')
log(f"matched positives {n_pos:,} | zeros {len(rect)-n_pos:,} "
    f"({100*(len(rect)-n_pos)/len(rect):.1f}%)")

rect = rect.merge(tgt_lk, on='ISIC4c', how='left')
rect = rect.merge(adv_lk, on='i_int', how='left')
rect = rect.merge(ip_lk, on=['i_int','ISIC4c','t_int'], how='left')
rect['US_trade'] = (rect['j_int'] == USA).astype('int8')
rect['target'] = rect['target'].astype('float32')
rect['Advanced_i'] = rect['Advanced_i'].astype('int8')

before = len(rect)
rect = rect[rect['IP_lag'].notna()].copy()
rect['IP_lag'] = rect['IP_lag'].astype('float32')
log(f"after lag-{LAG} merge: {len(rect):,} rows (dropped {before-len(rect):,} "
    f"with no IP at t-{LAG})")

miss = {c: int(rect[c].isna().sum()) for c in ['target','Advanced_i','imports','IP_lag']}
log(f"remaining missing: {miss}")
assert sum(miss.values()) == 0, "unexpected missing values after merges"

log(f"zero share in final frame: {100*(rect['imports']==0).mean():.1f}%")
rect.to_pickle("rect_panel.pkl")
log(f"saved rect_panel.pkl | {rect.shape} | "
    f"{rect.memory_usage(deep=True).sum()/1e9:.2f} GB in memory")
