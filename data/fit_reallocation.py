"""Is the gravity result expansion or reallocation?

The gravity DDD carries alpha_ist, which absorbs country i's TOTAL exports in sector s
in year t. So beta1 is a pure DESTINATION margin: US-bound relative to i's other
markets. A country that merely redirected existing exports from Europe to the US would
produce exactly the same coefficient as one that expanded output. The gravity spec
cannot tell them apart, by construction.

This drops the destination dimension and asks the question directly, on a
country-sector-year panel:

  X_ist = exp[ a_it + a_st + a_is + b (IP_ist x target_s x post_t x dev_i) + ... ]

  a_it  country-year: i's overall export growth
  a_st  sector-year: world demand for s
  a_is  country-sector level

b asks whether i's exports in s grew more than i's own trend and s's global trend
predict, in decoupling sectors it had targeted. Run on three outcomes:

  (1) TOTAL exports to the world     expansion?
  (2) US-bound only                  the destination the gravity spec identifies on
  (3) NON-US only                    the source of any reallocation

  (1) ~ 0 with (2) > 0 > (3)  =>  reallocation: the US gain came out of other markets
  (1) > 0 with (2) > (3) > 0  =>  expansion, tilted toward the US
  (1) > 0 with (3) ~ 0        =>  expansion that went to the US

Same treatment as the gravity headline: share_frac_policies lagged 3, standardised,
target_s x 1[t>=2018], developing ex-China. Clustered (i,s).
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
USA, CHINA = 842, 156
MEASURE, LAG = 'share_frac_policies', 3

d = pd.read_pickle("agg_for_estimation.pkl")
d['i']=d['i'].astype('int32'); d['j']=d['j'].astype('int32'); d['t_int']=d['t'].astype('int32')
d['imports']=d['imports'].astype('float64')
d[MEASURE]=pd.to_numeric(d[MEASURE],errors='coerce').fillna(0).astype('float32')
d['target']=d['target'].astype('float32')
log(f"loaded {d.shape}")

# collapse to country-sector-year, three outcomes
g = d.groupby(['i','ISIC4c','t_int'], observed=True)
tot = g['imports'].sum().rename('X_tot')
us  = d[d['j']==USA].groupby(['i','ISIC4c','t_int'],observed=True)['imports'].sum().rename('X_us')
# policy and target are (i,s,t) / (s,t) level already -- take first
key = g[[MEASURE,'target','Advanced_i']].first()
p = pd.concat([tot, key], axis=1).join(us).reset_index()
p['X_us']=p['X_us'].fillna(0.0)
p['X_non']=(p['X_tot']-p['X_us']).clip(lower=0)
del d, g; gc.collect()
log(f"country-sector-year panel {p.shape}")

# 3-year lag of the policy measure
lk = p[['i','ISIC4c','t_int',MEASURE]].copy(); lk['t_int']=lk['t_int']+LAG
lk = lk.rename(columns={MEASURE:'IP_lag'})
p = p.merge(lk, on=['i','ISIC4c','t_int'], how='left')
p = p[p['IP_lag'].notna()].copy()
log(f"after lag-{LAG} merge {p.shape}")

p = p[(p['Advanced_i']==0)&(p['i']!=CHINA)].copy()
p['ipz'] = (p['IP_lag']/p['IP_lag'].std()).astype('float32')
p['dec'] = (p['target']*(p['t_int']>=2018)).astype('float32')
p['DDD'] = (p['dec']*p['ipz']).astype('float32')
p['IPz'] = p['ipz']
p['fe_it']=p.groupby(['i','t_int'],observed=True).ngroup().astype('int32')
p['fe_st']=p.groupby(['ISIC4c','t_int'],observed=True).ngroup().astype('int32')
p['fe_is']=p.groupby(['i','ISIC4c'],observed=True).ngroup().astype('int32')
p['cl']=p['fe_is']
log(f"developing ex-China: {p.shape} | {p['i'].nunique()} countries | "
    f"{p['ISIC4c'].nunique()} sectors | {p['t_int'].min()}-{p['t_int'].max()}")
print(f"  mean X_tot {p['X_tot'].mean():,.0f}   X_us {p['X_us'].mean():,.0f}   "
      f"X_non {p['X_non'].mean():,.0f}")
print(f"  US share of these exports: {p['X_us'].sum()/p['X_tot'].sum():.3f}\n")

res={}
for lbl,y in [("(1) TOTAL exports to world",'X_tot'),
              ("(2) US-bound only",'X_us'),
              ("(3) NON-US only",'X_non')]:
    try:
        m=pf.fepois(f"{y} ~ DDD + IPz | fe_it + fe_st + fe_is", data=p,
                    vcov={"CRV1":"cl"}, iwls_maxiter=500,
                    demeaner=pf.LsmrDemeaner(fixef_maxiter=2000, fixef_tol=1e-8),
                    lean=True, store_data=False, copy_data=False)
        t=m.tidy()
        b,se,pv=t.loc['DDD','Estimate'],t.loc['DDD','Std. Error'],t.loc['DDD','Pr(>|t|)']
        st='***' if pv<.01 else '**' if pv<.05 else '*' if pv<.10 else ''
        print(f"{lbl:30s} N={m._N:>10,}   DDD {b:+.4f} ({se:.4f}){st:3s} p={pv:.4f}")
        res[lbl]=(b,se,pv,int(m._N))
        del m; gc.collect()
    except Exception as e:
        print(f"{lbl:30s} FAILED: {type(e).__name__}: {e}")
pd.DataFrame([{'outcome':k,'coef':v[0],'se':v[1],'p':v[2],'N':v[3]} for k,v in res.items()]
            ).to_csv("reallocation.csv",index=False)
log("DONE")
