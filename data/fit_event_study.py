import pandas as pd
import numpy as np
import pyfixest as pf
import time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, BASE_YEAR = 156, 2017
MEASURE, LAG = 'share_frac_policies', 3

agg = pd.read_pickle("agg_for_estimation.pkl")
agg['i_int'] = agg['i'].astype('int32'); agg['t_int'] = agg['t'].astype('int32')
agg[MEASURE] = pd.to_numeric(agg[MEASURE], errors='coerce').fillna(0).astype('float32')

lookup = (agg[['i_int','ISIC4c','t_int',MEASURE]]
          .drop_duplicates(subset=['i_int','ISIC4c','t_int']))
lookup['t_int'] = lookup['t_int'] + LAG
lookup = lookup.rename(columns={MEASURE: 'IP_lag'})

us = agg[agg['US_trade'] == 1].copy(); del agg
us = us[(us['i_int'] != CHINA) & (us['Advanced_i'] == 0)].copy()
us = us.merge(lookup, on=['i_int','ISIC4c','t_int'], how='left')
us = us[us['IP_lag'].notna()].copy()
log(f"sample: {len(us):,} obs | {us['i_int'].nunique()} exporters | years {us['t_int'].min()}-{us['t_int'].max()}")

sd = us['IP_lag'].std()
us['IP_z'] = (us['IP_lag'] / sd).astype('float32')
log(f"{MEASURE} lag{LAG}: SD={sd:.6f}, nonzero {100*(us['IP_lag']>0).mean():.1f}%")

us['fe_st'] = us.groupby(['ISIC4c','t'], observed=True).ngroup().astype('int32')
us['fe_is'] = us.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32')
us['fe_it'] = us.groupby(['i','t'], observed=True).ngroup().astype('int32')

years = sorted(y for y in us['t_int'].unique() if y != BASE_YEAR)
tri, own = [], []
for y in years:
    d = (us['t_int'] == y).astype('float32')
    us[f'IPxTGTxY{y}'] = (us['IP_z'] * us['target'] * d).astype('float32')
    us[f'IPxY{y}']     = (us['IP_z'] * d).astype('float32')
    tri.append(f'IPxTGTxY{y}'); own.append(f'IPxY{y}')

# IPxY* are the lower-order (non-target) path; IPxTGTxY* is the differential in
# decoupling sectors, which is the object of interest.
formula = f"imports ~ {' + '.join(tri + own)} | fe_st + fe_is + fe_it"
log(f"fitting event study: {len(tri)} triple + {len(own)} lower-order terms")

fit = pf.fepois(formula, data=us, vcov={"CRV1": "i_int"},
                demeaner=pf.LsmrDemeaner(fixef_maxiter=2000))
t = fit.tidy()
log("fit done")

rows = []
for y in years:
    k = f'IPxTGTxY{y}'
    if k in t.index:
        rows.append({'year': y, 'coef': t.loc[k,'Estimate'], 'se': t.loc[k,'Std. Error'],
                     'p': t.loc[k,'Pr(>|t|)'], 'lo': t.loc[k,'2.5%'], 'hi': t.loc[k,'97.5%']})
ev = pd.DataFrame(rows).sort_values('year')
ev.loc[len(ev)] = {'year': BASE_YEAR, 'coef': 0, 'se': np.nan, 'p': np.nan, 'lo': np.nan, 'hi': np.nan}
ev = ev.sort_values('year').reset_index(drop=True)
ev.to_csv("event_study_results.csv", index=False)

print(f"\n{'='*78}")
print(f"EVENT STUDY: {MEASURE} (lag {LAG}) x target x year, base {BASE_YEAR}")
print(f"developing ex-China, US-bound | FE: st + is + it | cluster: exporter")
print(f"coefficient = differential effect in DECOUPLING sectors, per 1 SD")
print('='*78)
lim = max(abs(ev['coef'].min()), abs(ev['coef'].max())) or 1
for _, r in ev.iterrows():
    era = "PRE " if r['year'] < 2018 else "POST"
    if r['year'] == BASE_YEAR:
        print(f"{int(r['year'])}  {era}   0.0000  (base)")
        continue
    star = "***" if r['p']<0.01 else "**" if r['p']<0.05 else "*" if r['p']<0.1 else ""
    pos = int(30 + 28*r['coef']/lim)
    bar = " "*min(pos,59) + "#"
    print(f"{int(r['year'])}  {era}  {r['coef']:+.4f} ({r['se']:.4f}) p={r['p']:.3f} {star:<3} |{bar}")
print(" "*17 + "|" + " "*29 + "0")

pre  = ev[(ev['year'] < 2018) & (ev['year'] != BASE_YEAR)]
post = ev[ev['year'] >= 2018]
print(f"\nmean PRE-2018 coefficient:  {pre['coef'].mean():+.4f}   "
      f"({(pre['p']<0.05).sum()}/{len(pre)} significant at 5%)")
print(f"mean POST-2018 coefficient: {post['coef'].mean():+.4f}   "
      f"({(post['p']<0.05).sum()}/{len(post)} significant at 5%)")
try:
    w = fit.wald_test(R=np.eye(len(t))[[t.index.get_loc(f'IPxTGTxY{y}') for y in years if y < 2018]])
    print(f"\njoint Wald test, all pre-2018 triple coefficients = 0: {w}")
except Exception as e:
    log(f"(wald test unavailable: {e})")
log("DONE")
