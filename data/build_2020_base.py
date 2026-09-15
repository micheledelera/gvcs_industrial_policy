"""Common data build for the 2020-base difference and levels specifications.

Windows, all anchored on 2020 as the base:
  s_base   2018-2020   the reference period
  s_post   2022-2024   outcome period (2021 excluded as transition)
  s_lag    2015-2017   used to form the pre-trend control

Three timing variants of each policy measure, all fixed at or before 2020:
  flow   mean over 2018-2020            policy activity at the base
  stock  cumulative sum 2007-2020       accumulated policy, the "stock" question
  lag    mean over 2012-2014            six years before the base

Dec_k = China's share of US imports in sector k over 2015-2017. Pre-determined
relative to BOTH candidate breaks (2018 tariffs, 2021 reconfiguration), so it is
not a function of either episode's outcome.

Sample: developing excluding China -- the between-country comparison.
"""
import pandas as pd, numpy as np, gc
USA, CHINA = 842, 156
W_LAG, W_BASE, W_POST = [2015,2016,2017], [2018,2019,2020], [2022,2023,2024]
W_LAGIP = [2012,2013,2014]
MEAS = ['n_policies','frac_policies','share_n_policies','share_frac_policies']

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
for c in MEAS: raw[c]=pd.to_numeric(raw[c],errors='coerce').fillna(0).astype('float32')
us = raw[raw['j']==USA]

x   = us.groupby(['i','ISIC4c','t'], observed=True)['imports'].sum()
tot = x.groupby(level=['ISIC4c','t'], observed=True).transform('sum')
s   = (x/tot.replace(0,np.nan)*100).rename('s')                       # pp, by year
def wmean(years): return s[s.index.get_level_values('t').isin(years)]\
                        .groupby(level=['i','ISIC4c'], observed=True).mean()
s_lag, s_base, s_post = wmean(W_LAG), wmean(W_BASE), wmean(W_POST)
wgt = (us[us['t'].isin(W_BASE)].groupby(['i','ISIC4c'], observed=True)['imports'].sum()/3).rename('w')

chn = s.xs(CHINA, level='i')
Dec = chn[chn.index.get_level_values('t').isin(W_LAG)].groupby('ISIC4c', observed=True).mean().rename('Dec')

cells = raw[['i','ISIC4c','t']+MEAS].drop_duplicates(subset=['i','ISIC4c','t'])
ip = {}
ip['flow']  = cells[cells['t'].isin(W_BASE)].groupby(['i','ISIC4c'],observed=True)[MEAS].mean()
ip['stock'] = cells[cells['t']<=2020].groupby(['i','ISIC4c'],observed=True)[MEAS].sum()
ip['lag']   = cells[cells['t'].isin(W_LAGIP)].groupby(['i','ISIC4c'],observed=True)[MEAS].mean()
for k in ip: ip[k].columns=[f"{c}__{k}" for c in MEAS]

adv = raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
act = raw.loc[raw['imports']>0,['i','ISIC4c']].drop_duplicates()
years = sorted(raw['t'].unique())
del raw, us; gc.collect()

# ---- cross-section for the difference equation ----
d = act.set_index(['i','ISIC4c']).join(
    [s_lag.rename('s_lag'), s_base.rename('s_base'), s_post.rename('s_post'), wgt]).reset_index()
d = d.merge(Dec, on='ISIC4c', how='left')
for k in ip: d = d.merge(ip[k], on=['i','ISIC4c'], how='left')
ipcols=[c for c in d.columns if '__' in c]
for c in ['s_lag','s_base','s_post','w','Dec']+ipcols: d[c]=d[c].fillna(0.0)
d['Advanced_i']=d['i'].map(adv); d=d[(d['Advanced_i']==0)&(d['i']!=CHINA)].copy()
d=d[d['w']>0].copy()
d['d_post']=d['s_post']-d['s_base']; d['d_pre']=d['s_base']-d['s_lag']
d.to_pickle("base2020_cross.pkl")
print(f"cross-section: {len(d):,} (i,k) cells | {d['i'].nunique()} countries | "
      f"{d['ISIC4c'].nunique()} sectors")
print(f"  d_post mean {d['d_post'].mean():+.4f} sd {d['d_post'].std():.4f}")
print(f"  d_pre  mean {d['d_pre'].mean():+.4f} sd {d['d_pre'].std():.4f}")
print(f"  Dec    mean {d['Dec'].mean():.4f} sd {d['Dec'].std():.4f}")

# ---- panel for the levels equation ----
p = act.loc[act.index.repeat(len(years))].copy(); p['t']=np.tile(years,len(act))
p = p.set_index(['i','ISIC4c','t']).join(s).reset_index()
p = p.merge(wgt,on=['i','ISIC4c'],how='left').merge(Dec,on='ISIC4c',how='left')
for k in ip: p = p.merge(ip[k], on=['i','ISIC4c'], how='left')
for c in ['s','w','Dec']+ipcols: p[c]=p[c].fillna(0.0)
p['Advanced_i']=p['i'].map(adv); p=p[(p['Advanced_i']==0)&(p['i']!=CHINA)].copy()
p=p[p['w']>0].copy()
p.to_pickle("base2020_panel.pkl")
print(f"\npanel: {len(p):,} rows | {p.groupby(['i','ISIC4c'],observed=True).ngroup().nunique():,} pairs "
      f"| {years[0]}-{years[-1]}")
print("\npolicy measures, nonzero share and sd (cross-section):")
for c in ipcols:
    print(f"  {c:32s} nonzero {100*(d[c]>0).mean():5.1f}%   sd {d[c].std():.6g}")
