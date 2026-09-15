"""Who carries the identifying weight in DDD_dev?

PPML's score weights each observation by its fitted mean, which is close to
observed imports. So the exporters that matter for the coefficient are the ones
with large  imports x DDD_dev , not the ones with many rows. Report both.
"""
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
d = d[d['IP_lag'].notna()].copy()
del raw, lk

ip  = (d['IP_lag'] / d['IP_lag'].std()).astype('float32')
dev = ((1 - d['Advanced_i']) * (d['i_int'] != CHINA)).astype('float32')
dec = (d['target'] * (d['t_int'] >= 2018)).astype('float32')
d['DDD_dev'] = (dec * ip * d['US_trade'].astype('float32') * dev).values
d['mass']    = (d['DDD_dev'] * d['imports'].astype('float64')).values

tot_mass = d['mass'].sum()
tot_rows = (d['DDD_dev'] > 0).sum()
g = (d[d['DDD_dev'] > 0]
     .groupby('i_int', observed=True)
     .agg(rows=('DDD_dev','size'), mass=('mass','sum'), imports=('imports','sum'))
     .sort_values('mass', ascending=False))
g['mass_pct'] = 100 * g['mass'] / tot_mass
g['rows_pct'] = 100 * g['rows'] / tot_rows
g['cum_mass'] = g['mass_pct'].cumsum()

names = {704:'Vietnam',484:'Mexico',764:'Thailand',458:'Malaysia',699:'India',
         360:'Indonesia',608:'Philippines',76:'Brazil',792:'Turkey',710:'S.Africa',
         818:'Egypt',586:'Pakistan',50:'Bangladesh',288:'Ghana',404:'Kenya',
         152:'Chile',170:'Colombia',604:'Peru',32:'Argentina',516:'Namibia',
         862:'Venezuela',368:'Iraq',400:'Jordan',504:'Morocco',788:'Tunisia',
         144:'Sri Lanka',116:'Cambodia',418:'Laos',104:'Myanmar',643:'Russia'}
g['name'] = [names.get(k, str(k)) for k in g.index]

print(f"treated obs (DDD_dev>0): {tot_rows:,}   total regressor mass: {tot_mass:,.0f}\n")
print("Top 15 developing exporters by PPML-weighted treatment mass")
print(g.head(15)[['name','rows_pct','mass_pct','cum_mass']].round(2).to_string())
print(f"\nVietnam + Mexico: {g.loc[g.index.isin([704,484]),'mass_pct'].sum():.1f}% of mass, "
      f"{g.loc[g.index.isin([704,484]),'rows_pct'].sum():.1f}% of treated rows")
g.to_csv("leverage_by_exporter.csv")

# --- effective clusters and treated-cell structure ---
d['cl_is'] = d.groupby(['i','ISIC4c'], observed=True).ngroup()
tr = d[d['DDD_dev'] > 0]
print(f"\ntreated (i,s) clusters: {tr['cl_is'].nunique():,}  of  {d['cl_is'].nunique():,} total")
print(f"treated exporters: {tr['i_int'].nunique()}   treated sectors: {tr['ISIC4c'].nunique()}")
print(f"treated obs per cluster: median {tr.groupby('cl_is').size().median():.0f}, "
      f"max {tr.groupby('cl_is').size().max()}")
top2 = tr[tr['i_int'].isin([704,484])]
print(f"Vietnam+Mexico treated clusters: {top2['cl_is'].nunique():,}")
