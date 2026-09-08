"""Between-country long difference: did policy users gain more US market share?

  d_s_ik = b IP^pre_ik + g d_s^pre_ik + a_k + X_ik'th + e_ik

s_ik  = country i's share of TOTAL US imports in sector k
d_s   = s(2022-24) - s(2015-17)          outcome
d_s^pre = s(2015-17) - s(2010-12)        pre-trend control
IP^pre  = mean share_n_policies in (i,k) over 2015-17, standardised

Each requirement of the question maps onto one piece:
  "gain US market share"      -> the dependent variable itself
  "similar China exposure"    -> alpha_k. Comparing within a sector means identical
                                 exposure by construction; no proxy, and no noisy
                                 ChinaShare interaction (which gave an SE of 0.68).
  "similar capability"        -> first-differencing removes any time-invariant (i,k)
                                 effect, which is capability in that sector.
  "used industrial policy"    -> IP fixed before 2018, so no reverse causality.
  pre-trends                  -> conditioned on rather than assumed flat. The
                                 staggered event study showed convergence that
                                 survives country-year absorption, so assuming it
                                 away is not available.
  between-country             -> alpha_k fixes the sector; variation is across
                                 countries within it.

Shares, not logs: no zero-dropping, so no selection on the extensive margin.
Weighted by pre-period US imports, else a cell going from 0.001% to 0.002% counts
as much as Vietnam. Clustered by country, since policy is chosen at country level.
"""
import pandas as pd, numpy as np, statsmodels.api as sm, gc

USA, CHINA = 842, 156
PRE0, PRE, POST = [2010,2011,2012], [2015,2016,2017], [2022,2023,2024]

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
MEAS=['n_policies','frac_policies','share_n_policies','share_frac_policies']
for c in MEAS:
    raw[c]=pd.to_numeric(raw[c],errors='coerce').fillna(0).astype('float32')
us = raw[raw['j']==USA]

def share(years):
    x = us[us['t'].isin(years)].groupby(['i','ISIC4c'], observed=True)['imports'].sum()/len(years)
    tot = x.groupby(level='ISIC4c', observed=True).transform('sum')
    return (x / tot.replace(0, np.nan)).rename('s'), x.rename('x')
s0,_    = share(PRE0)
s1,x1   = share(PRE)
s2,_    = share(POST)

ip = (raw[raw['t'].isin(PRE)][['i','ISIC4c','t']+MEAS]
      .drop_duplicates(subset=['i','ISIC4c','t'])
      .groupby(['i','ISIC4c'], observed=True)[MEAS].mean())
adv = raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
# pairs that are ever active anywhere, so zeros are genuine "does not sell to US"
act = raw.loc[raw['imports']>0, ['i','ISIC4c']].drop_duplicates()
del raw, us; gc.collect()

d = act.set_index(['i','ISIC4c']).join([s0.rename('s0'), s1.rename('s1'),
                                        s2.rename('s2'), x1, ip]).reset_index()
for c in ['s0','s1','s2','x']+MEAS: d[c]=d[c].fillna(0.0)
d['Advanced_i']=d['i'].map(adv)
d = d[(d['Advanced_i']==0) & (d['i']!=CHINA)].copy()
d['d_post'] = (d['s2']-d['s1'])*100      # percentage points
d['d_pre']  = (d['s1']-d['s0'])*100

d['s1_pp']  = d['s1']*100
print(f"developing ex-China (i,k) cells: {len(d):,}   sectors {d['ISIC4c'].nunique()}   "
      f"countries {d['i'].nunique()}")
for c in MEAS:
    print(f"  {c:22s} nonzero {int((d[c]>0).sum()):>6,} ({100*(d[c]>0).mean():4.1f}%)  sd {d[c].std():.6g}")
print(f"  outcome d_post (pp): mean {d['d_post'].mean():+.4f}  sd {d['d_post'].std():.4f}")
print(f"  pre-trend d_pre (pp): mean {d['d_pre'].mean():+.4f}  sd {d['d_pre'].std():.4f}")
print(f"  corr(d_post, d_pre) = {d['d_post'].corr(d['d_pre']):+.3f}\n")

SPECS=[("1. naive",              ['IPz']),
       ("2. + sector FE",        ['IPz']),
       ("3. + pre-trend",        ['IPz','d_pre']),
       ("4. + initial share",    ['IPz','d_pre','s1_pp'])]
sec = pd.get_dummies(d['ISIC4c'].astype(str), prefix='k', drop_first=True).astype(float)
w = d['x'].clip(lower=0).values
rows=[]
for MEASURE in MEAS:
  d['IPz'] = d[MEASURE]/d[MEASURE].std()
  print(f"\n{'='*70}\nMEASURE: {MEASURE}\n{'='*70}")
  for name, X in SPECS:
    M = d[X].copy()
    if name!="1. naive": M = pd.concat([M, sec], axis=1)
    M = sm.add_constant(M)
    m = sm.WLS(d['d_post'], M, weights=w).fit(cov_type='cluster',
                                              cov_kwds={'groups': d['i']})
    print(f"--- {name}   N={int(m.nobs):,}  R2={m.rsquared:.3f}   "
          f"IPz {m.params['IPz']:+.5f} (se {m.bse['IPz']:.5f})  p={m.pvalues['IPz']:.4f}")
    rows.append({'measure':MEASURE,'spec':name,'coef':m.params['IPz'],
                 'se':m.bse['IPz'],'p':m.pvalues['IPz'],'N':int(m.nobs)})
pd.DataFrame(rows).to_csv("longdiff_results.csv", index=False)
r=pd.DataFrame(rows)
def cell(x):
    st='***' if x['p']<.01 else '**' if x['p']<.05 else '*' if x['p']<.10 else ''
    return f"{x['coef']:+.4f} ({x['se']:.4f}){st}"
r['c']=r.apply(cell,axis=1)
print("\n"+"="*76)
print("IP COEFFICIENT BY MEASURE AND SPECIFICATION  (pp change in US market share)")
print("="*76)
print(r.pivot_table(index='measure',columns='spec',values='c',aggfunc='first')
       .reindex(MEAS).to_string())
