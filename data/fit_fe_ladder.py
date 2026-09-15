import pandas as pd
import numpy as np
import pyfixest as pf
import time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA = 156
agg = pd.read_pickle("agg_for_estimation.pkl")
us = agg[agg['US_trade'] == 1].copy(); del agg
us['i_int'] = us['i'].astype('int32'); us['t_int'] = us['t'].astype('int32')
us = us[(us['i_int'] != CHINA) & (us['Advanced_i'] == 0)].copy()
log(f"developing ex-China, US-bound: {len(us):,} obs, {us['i_int'].nunique()} exporters")

us['n_policies'] = pd.to_numeric(us['n_policies'], errors='coerce').fillna(0)
us['Post'] = (us['t_int'] >= 2018).astype('float32')
us['Dec2'] = (us['target'] * us['Post']).astype('float32')   # author's original treatment

us['fe_st'] = us.groupby(['ISIC4c','t'], observed=True).ngroup().astype('int32')
us['fe_i']  = us['i_int']
us['fe_is'] = us.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32')
us['fe_it'] = us.groupby(['i','t'], observed=True).ngroup().astype('int32')

MEASURES = ['n_policies', 'frac_policies', 'n_sub']
for m in MEASURES:
    sd = us[m].std()
    us[f'{m}_z']      = (us[m] / sd).astype('float32')       # per-SD, comparable across measures
    us[f'{m}_z_xDec'] = (us[f'{m}_z'] * us['Dec2']).astype('float32')
    log(f"{m}: SD={sd:,.4f}  nonzero={100*(us[m]>0).mean():.1f}%")

LADDER = [
    ("L1  st",             "fe_st"),
    ("L2  st + i",         "fe_st + fe_i"),
    ("L3  st + is",        "fe_st + fe_is"),
    ("L4  st + is + it",   "fe_st + fe_is + fe_it"),
]

rows = []
for m in MEASURES:
    for lname, fes in LADDER:
        f = f"imports ~ {m}_z + {m}_z_xDec | {fes}"
        try:
            fit = pf.fepois(f, data=us, vcov={"CRV1": "fe_i"},
                            demeaner=pf.LsmrDemeaner(fixef_maxiter=2000))
            t = fit.tidy()
            rows.append({
                'measure': m, 'ladder': lname, 'N': fit._N,
                'IP': t.loc[f'{m}_z','Estimate'],       'IP_se': t.loc[f'{m}_z','Std. Error'],
                'IP_p': t.loc[f'{m}_z','Pr(>|t|)'],
                'IPxDec': t.loc[f'{m}_z_xDec','Estimate'], 'IPxDec_se': t.loc[f'{m}_z_xDec','Std. Error'],
                'IPxDec_p': t.loc[f'{m}_z_xDec','Pr(>|t|)'],
            })
            log(f"{m} | {lname}: IP={rows[-1]['IP']:+.4f} (p={rows[-1]['IP_p']:.3f})  "
                f"IPxDec={rows[-1]['IPxDec']:+.4f} (p={rows[-1]['IPxDec_p']:.3f})")
            del fit
        except Exception as e:
            log(f"{m} | {lname}: FAILED {e}")

res = pd.DataFrame(rows)
res.to_csv("fe_ladder_results.csv", index=False)
print("\n" + "="*110)
print("FE LADDER -- developing ex-China, US-bound. Coefficients per 1 SD of each measure.")
print("IP = main effect (matches the descriptive). IPxDec = target x post2018 interaction.")
print("="*110)
for m in MEASURES:
    print(f"\n### {m}")
    sub = res[res['measure'] == m]
    print(sub[['ladder','N','IP','IP_se','IP_p','IPxDec','IPxDec_se','IPxDec_p']]
          .to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
log("DONE")
