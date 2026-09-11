import pandas as pd
import numpy as np
import pyfixest as pf
import time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, BASE = 156, 2017
SPECS = [('share_frac_policies', 3), ('share_frac_policies', 1), ('share_n_policies', 1)]

agg = pd.read_pickle("agg_for_estimation.pkl")
agg['i_int'] = agg['i'].astype('int32'); agg['t_int'] = agg['t'].astype('int32')
for m, _ in SPECS:
    agg[m] = pd.to_numeric(agg[m], errors='coerce').fillna(0).astype('float32')

base_us = agg[agg['US_trade'] == 1].copy()
base_us = base_us[(base_us['i_int'] != CHINA) & (base_us['Advanced_i'] == 0)].copy()

out = []
for MEASURE, LAG in SPECS:
    look = (agg[['i_int','ISIC4c','t_int',MEASURE]]
            .drop_duplicates(subset=['i_int','ISIC4c','t_int']))
    look['t_int'] = look['t_int'] + LAG
    look = look.rename(columns={MEASURE: 'IP_lag'})

    us = base_us.merge(look, on=['i_int','ISIC4c','t_int'], how='left')
    us = us[us['IP_lag'].notna()].copy()
    us['IP_z'] = (us['IP_lag'] / us['IP_lag'].std()).astype('float32')
    us['fe_st'] = us.groupby(['ISIC4c','t'], observed=True).ngroup().astype('int32')
    us['fe_is'] = us.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32')
    us['fe_it'] = us.groupby(['i','t'], observed=True).ngroup().astype('int32')

    pre_years = sorted(y for y in us['t_int'].unique() if y < BASE)
    all_years = sorted(y for y in us['t_int'].unique() if y != BASE)
    tri, own = [], []
    for y in pre_years:                       # flexible pre-period
        us[f'TGT_y{y}'] = (us['IP_z']*us['target']*(us['t_int']==y)).astype('float32'); tri.append(f'TGT_y{y}')
    us['TGT_post'] = (us['IP_z']*us['target']*(us['t_int']>=2018)).astype('float32'); tri.append('TGT_post')
    for y in all_years:                       # fully flexible lower-order path
        us[f'IP_y{y}'] = (us['IP_z']*(us['t_int']==y)).astype('float32'); own.append(f'IP_y{y}')

    f = f"imports ~ {' + '.join(tri+own)} | fe_st + fe_is + fe_it"
    log(f"=== {MEASURE} lag{LAG}: {len(pre_years)} pre-year terms + pooled post ===")
    try:
        fit = pf.fepois(f, data=us, vcov={"CRV1": "i_int"},
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=2000))
        t = fit.tidy()
        print(f"\n### {MEASURE}, lag {LAG}  (N={fit._N:,}, base {BASE})")
        rows = []
        for k in tri:
            if k in t.index:
                lab = 'POST-2018 (pooled)' if k=='TGT_post' else k.replace('TGT_y','')
                rows.append({'term': lab, 'coef': t.loc[k,'Estimate'], 'se': t.loc[k,'Std. Error'],
                             'p': t.loc[k,'Pr(>|t|)']})
        r = pd.DataFrame(rows)
        r['sig'] = np.where(r['p']<0.01,'***', np.where(r['p']<0.05,'**', np.where(r['p']<0.1,'*','')))
        print(r.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
        pre = r[r['term']!='POST-2018 (pooled)']
        print(f"  pre-period: {(pre['p']<0.05).sum()}/{len(pre)} significant at 5%; "
              f"largest |coef| {pre['coef'].abs().max():.4f}")
        r.insert(0,'lag',LAG); r.insert(0,'measure',MEASURE); out.append(r)
        del fit
    except Exception as e:
        log(f"FAILED: {e}")

if out:
    pd.concat(out).to_csv("event_hybrid_results.csv", index=False)
log("DONE")
