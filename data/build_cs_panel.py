"""Country-sector panel of US-bound exports, for the event study and synthetic DiD.

Unit: (country, sector), developing excluding China. Outcome: US-bound exports in
LEVELS -- PPML, not log differences. Levels handle the zeros that appear everywhere
at this unit, weight by economic size, and keep the same functional form as the
gravity half of the project.

Frame: keep (i, s, t) cells where country i exports sector s to SOMEONE that year,
then take its US flow, zero if absent. A zero here means "makes it, does not sell it
to the US" -- an extensive-margin zero, not a structural one.

Treatment: Treated_is = 1 if country i recorded any policy in sector s during
2015-2017. Donors are NEVER-treated cells: no recorded policy in any year 2007-2024,
so the control group is not contaminated by later adoption.

The event study uses alpha_is + alpha_st. alpha_st means treated cells are compared
to untreated cells IN THE SAME SECTOR-YEAR -- the donor restriction we want, imposed
by fixed effect rather than by matching. Deliberately NO country-year fixed effect:
that would force identification within country across sectors, which is the gravity
design, and would discard exactly the between-country variation this design exists
to use.
"""
import pandas as pd, numpy as np, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, USA = 156, 842
raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32')
raw['t']=raw['t'].astype('int32')
raw['n_policies']=pd.to_numeric(raw['n_policies'],errors='coerce').fillna(0).astype('float32')
log(f"loaded {raw.shape}")

# active (i,s,t): exports sector s to any destination that year
act = raw.loc[raw['imports']>0, ['i','ISIC4c','t']].drop_duplicates()
us  = (raw.loc[(raw['j']==USA) & (raw['imports']>0), ['i','ISIC4c','t','imports']]
       .groupby(['i','ISIC4c','t'], observed=True)['imports'].sum().rename('us_exp'))
p = act.merge(us, on=['i','ISIC4c','t'], how='left')
p['us_exp'] = p['us_exp'].fillna(0.0)
log(f"active cells {len(p):,} | zero US flow {100*(p['us_exp']==0).mean():.1f}%")

# policy history per (i,s)
pol = (raw[['i','ISIC4c','t','n_policies']].drop_duplicates(subset=['i','ISIC4c','t'])
       .groupby(['i','ISIC4c'], observed=True)
       .apply(lambda g: pd.Series({
           'pol_pre':  g.loc[g['t'].between(2015,2017),'n_policies'].sum(),
           'pol_ever': g['n_policies'].sum()}), include_groups=False))
p = p.merge(pol, on=['i','ISIC4c'], how='left')
adv = raw[['i','Advanced_i']].drop_duplicates()
p = p.merge(adv, on='i', how='left')
del raw; gc.collect()

p = p[(p['Advanced_i']==0) & (p['i']!=CHINA)].copy()
p['Treated'] = (p['pol_pre']>0).astype('int8')
p['NeverTreated'] = (p['pol_ever']==0).astype('int8')
p = p[(p['Treated']==1) | (p['NeverTreated']==1)].copy()   # drop later-adopters
log(f"developing ex-China, treated or never-treated: {len(p):,} rows")
nis = p.groupby(['i','ISIC4c'], observed=True).ngroup().nunique()
tr  = p[p['Treated']==1].groupby(['i','ISIC4c'], observed=True).ngroup().nunique()
log(f"  (i,s) cells: {nis:,}   treated {tr:,}   donors {nis-tr:,}")
log(f"  countries: {p['i'].nunique()}   sectors: {p['ISIC4c'].nunique()}   "
    f"years {p['t'].min()}-{p['t'].max()}")

p.to_pickle("cs_panel.pkl")
log(f"saved cs_panel.pkl {p.shape}")

# --- raw pre-trends, indexed to 2017 ---
g = p.groupby(['Treated','t'])['us_exp'].sum().unstack(0)
g.columns = ['donor','treated']
idx = g / g.loc[2017]
print("\nUS-bound exports, total by group, indexed to 2017 = 1.00")
print(idx.round(3).to_string())
