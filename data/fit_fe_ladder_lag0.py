import pandas as pd
import numpy as np
import pyfixest as pf
import time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA = 156
MEASURES = ['n_policies','frac_policies','n_sub','frac_sub',
            'share_n_policies','share_n_sub','share_frac_policies','share_frac_sub']
LAGS = [0]

agg = pd.read_pickle("agg_for_estimation.pkl")
agg['i_int'] = agg['i'].astype('int32'); agg['t_int'] = agg['t'].astype('int32')
for m in MEASURES:
    if m in agg.columns:
        agg[m] = pd.to_numeric(agg[m], errors='coerce').fillna(0).astype('float32')
present = [m for m in MEASURES if m in agg.columns]
log(f"measures present: {present}")

# IP is constant across j within (i,s,t) -- build the lag lookup from the FULL panel
# so lag coverage is not broken by gaps in US-bound trade.
lookup = (agg[['i_int','ISIC4c','t_int'] + present]
          .drop_duplicates(subset=['i_int','ISIC4c','t_int']))
log(f"(i,s,t) IP lookup: {len(lookup):,} rows")

us = agg[agg['US_trade'] == 1].copy(); del agg
us = us[(us['i_int'] != CHINA) & (us['Advanced_i'] == 0)].copy()
log(f"developing ex-China, US-bound: {len(us):,} obs")

us['Post'] = (us['t_int'] >= 2018).astype('float32')
us['Dec2'] = (us['target'] * us['Post']).astype('float32')
us['fe_st'] = us.groupby(['ISIC4c','t'], observed=True).ngroup().astype('int32')
us['fe_i']  = us['i_int']
us['fe_is'] = us.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32')
us['fe_it'] = us.groupby(['i','t'], observed=True).ngroup().astype('int32')

for L in LAGS:
    tmp = lookup.copy()
    tmp['t_int'] = tmp['t_int'] + L
    tmp = tmp.rename(columns={m: f'{m}_L{L}' for m in present})
    us = us.merge(tmp, on=['i_int','ISIC4c','t_int'], how='left')
    log(f"lag {L} merged; coverage {100*us[f'{present[0]}_L{L}'].notna().mean():.1f}%")

LADDER = [("L1 st","fe_st"), ("L2 +i","fe_st + fe_i"),
          ("L3 +is","fe_st + fe_is"), ("L4 +is+it","fe_st + fe_is + fe_it")]

rows = []
for L in LAGS:
    for m in present:
        col = f'{m}_L{L}'
        d = us[us[col].notna()].copy()
        sd = d[col].std()
        if not np.isfinite(sd) or sd == 0:
            log(f"SKIP {m} L{L}: zero/degenerate SD"); continue
        d['ip_z'] = (d[col] / sd).astype('float32')
        d['ip_z_xDec'] = (d['ip_z'] * d['Dec2']).astype('float32')
        for lname, fes in LADDER:
            try:
                fit = pf.fepois(f"imports ~ ip_z + ip_z_xDec | {fes}", data=d,
                                vcov={"CRV1": "fe_i"},
                                demeaner=pf.LsmrDemeaner(fixef_maxiter=2000))
                t = fit.tidy()
                rows.append({'lag': L, 'measure': m, 'ladder': lname, 'N': fit._N,
                             'IP': t.loc['ip_z','Estimate'], 'IP_p': t.loc['ip_z','Pr(>|t|)'],
                             'IPxDec': t.loc['ip_z_xDec','Estimate'],
                             'IPxDec_p': t.loc['ip_z_xDec','Pr(>|t|)']})
                del fit
            except Exception as e:
                log(f"FAILED {m} L{L} {lname}: {e}")
        r = [x for x in rows if x['measure']==m and x['lag']==L]
        if r:
            log(f"L{L} {m}: " + "  ".join(f"{x['ladder']}={x['IP']:+.3f}({x['IP_p']:.2f})" for x in r))

res = pd.DataFrame(rows)
res.to_csv("fe_ladder_lagged_results.csv", index=False)

def show(val, pval, title):
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    for L in LAGS:
        print(f"\n--- lag {L} ---")
        p = res[res['lag']==L].pivot(index='measure', columns='ladder', values=val).round(4)
        q = res[res['lag']==L].pivot(index='measure', columns='ladder', values=pval).round(3)
        out = p.astype(str) + " (" + q.astype(str) + ")"
        print(out.reindex(columns=[l for l,_ in LADDER]).to_string())

show('IP','IP_p',     "IP MAIN EFFECT, per 1 SD -- coefficient (p-value)")
show('IPxDec','IPxDec_p', "IP x (target x post2018), per 1 SD -- coefficient (p-value)")
log("DONE")
