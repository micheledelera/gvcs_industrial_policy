"""Cell (2) of fit_reallocation.py -- US-bound only -- with a demeaning tolerance ladder.

X_us is zero in many country-sector-years (the US takes 18.3% of these exports), which
is why the default demeaner exhausted 2000 iterations. Ladder the iteration cap and
tolerance, and report which rung converged so the number is not silently softer than
the other two cells.
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
g = d.groupby(['i','ISIC4c','t_int'], observed=True)
tot = g['imports'].sum().rename('X_tot')
us  = d[d['j']==USA].groupby(['i','ISIC4c','t_int'],observed=True)['imports'].sum().rename('X_us')
key = g[[MEASURE,'target','Advanced_i']].first()
p = pd.concat([tot, key], axis=1).join(us).reset_index()
p['X_us']=p['X_us'].fillna(0.0)
del d, g; gc.collect()
lk = p[['i','ISIC4c','t_int',MEASURE]].copy(); lk['t_int']=lk['t_int']+LAG
lk = lk.rename(columns={MEASURE:'IP_lag'})
p = p.merge(lk, on=['i','ISIC4c','t_int'], how='left')
p = p[p['IP_lag'].notna()].copy()
p = p[(p['Advanced_i']==0)&(p['i']!=CHINA)].copy()
p['ipz']=(p['IP_lag']/p['IP_lag'].std()).astype('float32')
p['DDD']=(p['target']*(p['t_int']>=2018)*p['ipz']).astype('float32')
p['IPz']=p['ipz']
p['fe_it']=p.groupby(['i','t_int'],observed=True).ngroup().astype('int32')
p['fe_st']=p.groupby(['ISIC4c','t_int'],observed=True).ngroup().astype('int32')
p['fe_is']=p.groupby(['i','ISIC4c'],observed=True).ngroup().astype('int32')
p['cl']=p['fe_is']
print(f"panel {p.shape}   X_us zero in {100*(p['X_us']==0).mean():.1f}% of rows")

for it,atol in [(10000,1e-8),(20000,1e-7),(20000,1e-6),(30000,1e-5)]:
    try:
        m=pf.fepois("X_us ~ DDD + IPz | fe_it + fe_st + fe_is", data=p,
                    vcov={"CRV1":"cl"}, iwls_maxiter=500,
                    demeaner=pf.LsmrDemeaner(fixef_maxiter=it, fixef_atol=atol, fixef_btol=atol),
                    lean=True, store_data=False, copy_data=False)
        t=m.tidy()
        b,se,pv=t.loc['DDD','Estimate'],t.loc['DDD','Std. Error'],t.loc['DDD','Pr(>|t|)']
        st='***' if pv<.01 else '**' if pv<.05 else '*' if pv<.10 else ''
        print(f"\nCONVERGED at maxiter={it}, atol={atol:g}")
        print(f"(2) US-bound only              N={m._N:>10,}   DDD {b:+.4f} ({se:.4f}){st:3s} p={pv:.4f}")
        pd.DataFrame([{'outcome':'(2) US-bound only','coef':b,'se':se,'p':pv,
                       'N':int(m._N),'maxiter':it,'atol':atol}]).to_csv("realloc_us.csv",index=False)
        break
    except Exception as e:
        print(f"maxiter={it}, atol={atol:g}: {type(e).__name__}: {e}", flush=True)
log("DONE")
