"""How big is the rectangularised panel, and are its zeros meaningful?

The data as delivered contains only positive flows, so entry and exit are invisible.
Rectangularising restores them -- but naively filling every (i, j, s, t) cell would
manufacture mostly STRUCTURAL zeros: a country that has never exported a sector to
anyone is not "failing to serve the US", it just does not make the product.

So the frame is: keep (i, s, t) cells where exporter i ships sector s to SOMEONE
(any of the 229 destinations in the raw data) in year t, then rectangularise those
over the 22 advanced destinations. A zero in that frame is a genuine "produces it,
does not sell it here", which is the object the extensive margin is about.

Reports the size, the implied zero share, and how US-bound entry moves over time.
"""
import pandas as pd, numpy as np

CHINA, LAG, MEASURE = 156, 3, 'share_frac_policies'
DEST = {842,276,251,380,826,528,724,56,392,124,410,36,752,40,208,246,372,620,300,579,757,554}
USA = 842

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i_int'] = raw['i'].astype('int32')
raw['j_int'] = raw['j'].astype('int32')
raw['t_int'] = raw['t'].astype('int32')
print(f"raw rows {len(raw):,} | exporters {raw['i_int'].nunique()} | "
      f"destinations {raw['j_int'].nunique()} | sectors {raw['ISIC4c'].nunique()} | "
      f"years {raw['t_int'].min()}-{raw['t_int'].max()}")

# (i,s,t) cells active anywhere in the world
act = raw[raw['imports'] > 0][['i_int','ISIC4c','t_int']].drop_duplicates()
print(f"\nactive (i,s,t) cells (exports to ANY destination): {len(act):,}")
print(f"rectangularised over {len(DEST)} advanced destinations: {len(act)*len(DEST):,} rows")

sub = raw[raw['j_int'].isin(DEST)]
obs = sub[sub['imports'] > 0][['i_int','ISIC4c','t_int','j_int']].drop_duplicates()
print(f"currently observed positive flows in that frame:      {len(obs):,}")
print(f"implied zero share after rectangularisation: "
      f"{100*(1 - len(obs)/(len(act)*len(DEST))):.1f}%")

# US-bound entry among active cells, by year
us_pos = (raw[(raw['j_int'] == USA) & (raw['imports'] > 0)][['i_int','ISIC4c','t_int']]
          .drop_duplicates().assign(serves_us=1))
a = act.merge(us_pos, on=['i_int','ISIC4c','t_int'], how='left')
a['serves_us'] = a['serves_us'].fillna(0)

dev_codes = set(raw.loc[(raw['Advanced_i'] == 0) & (raw['i_int'] != CHINA), 'i_int'].unique())
a['dev'] = a['i_int'].isin(dev_codes)
tab = (a[a['dev']].groupby('t_int')['serves_us']
       .agg(active_cells='size', serving_us='sum'))
tab['pct_serving_us'] = 100 * tab['serving_us'] / tab['active_cells']
print("\nDeveloping (ex-China) active (i,s,t) cells and US-bound participation")
print(tab.round(2).to_string())

# transitions across the 2018 cutoff
w = (a[a['dev']].pivot_table(index=['i_int','ISIC4c'], columns='t_int',
                             values='serves_us', aggfunc='max'))
for y0, y1 in [(2015, 2017), (2017, 2021), (2017, 2024)]:
    if y0 in w.columns and y1 in w.columns:
        p = w[[y0, y1]].dropna()
        ent = int(((p[y0] == 0) & (p[y1] == 1)).sum())
        ex  = int(((p[y0] == 1) & (p[y1] == 0)).sum())
        print(f"{y0}->{y1}: {len(p):,} cells present both years | "
              f"entry {ent:,} ({100*ent/len(p):.1f}%) | exit {ex:,} ({100*ex/len(p):.1f}%)")
