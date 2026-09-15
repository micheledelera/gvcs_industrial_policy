"""Is the §3s / §3ab disagreement the SAMPLE or the ESTIMATOR?

§3s (PPML triple difference, all pairs)   : US +0.018, non-US −0.008, total −0.006
§3ab (SC matched long difference, 2,559)  : US +0.210, non-US +0.102, total +0.106

Run §3s's EXACT specification on three nested samples, changing nothing else:

  (a) FULL              every developing ex-China pair, all 125 sectors  -- reproduces §3s
  (b) EVENT SECTORS     restricted to the 62 sectors with a decoupling event
  (c) SC SAMPLE         (b) further restricted to pairs with X_US > 0 in every year
                        2010-2024 -- the exact pairs §3ab uses

If non-US flips positive at (c), the disagreement is the sample selection -- SC's
complete-positive-series requirement drops entrants and exiters, and it conditions on US
continuity, which is endogenous. If it stays negative through (c), the disagreement is
the estimator: PPML on levels versus OLS on logs, continuous versus binary treatment.

  X_ist = exp[ a_it + a_st + a_is + b (IP_ist x target_s x post_t) + eta IP_ist ]
  share_frac_policies lagged 3, standardised; clustered (i,s).
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
USA, CHINA = 842, 156
BASE=[2015,2016,2017]; THR=0.25
MEASURE, LAG = 'share_frac_policies', 3

d=pd.read_pickle("agg_for_estimation.pkl")
d['i']=d['i'].astype('int32'); d['j']=d['j'].astype('int32'); d['t_int']=d['t'].astype('int32')
d['imports']=d['imports'].astype('float64')
d[MEASURE]=pd.to_numeric(d[MEASURE],errors='coerce').fillna(0).astype('float32')
d['target']=d['target'].astype('float32')
g=d.groupby(['i','ISIC4c','t_int'],observed=True)
tot=g['imports'].sum().rename('X_tot')
us=d[d['j']==USA].groupby(['i','ISIC4c','t_int'],observed=True)['imports'].sum().rename('X_us')
key=g[[MEASURE,'target','Advanced_i']].first()
p=pd.concat([tot,key],axis=1).join(us).reset_index()
p['X_us']=p['X_us'].fillna(0.0); p['X_non']=(p['X_tot']-p['X_us']).clip(lower=0)

# event sectors, same definition as §3t-§3ab
sh=(d[d['j']==USA].groupby(['i','ISIC4c','t_int'],observed=True)['imports'].sum())
tt=sh.groupby(level=['ISIC4c','t_int'],observed=True).transform('sum')
S=(sh/tt.replace(0,np.nan)*100).xs(CHINA,level='i').unstack('t_int')
S.columns=[int(c) for c in S.columns]; S=S[sorted(S.columns)]
b0=S[BASE].mean(axis=1); S=S.loc[b0>=5.0]; b0=b0.loc[S.index]
below=S.lt(b0*(1-THR),axis=0)
stay=below.iloc[:,::-1].cummin(axis=1).iloc[:,::-1].astype(bool)
EV=set(stay.apply(lambda r: r.index[r.values.argmax()] if r.any() else np.nan,axis=1).dropna().index)
del d,g,sh,tt; gc.collect()

lk=p[['i','ISIC4c','t_int',MEASURE]].copy(); lk['t_int']=lk['t_int']+LAG
lk=lk.rename(columns={MEASURE:'IP_lag'})
p=p.merge(lk,on=['i','ISIC4c','t_int'],how='left')
p=p[p['IP_lag'].notna()].copy()
p=p[(p['Advanced_i']==0)&(p['i']!=CHINA)].copy()
p['ipz']=(p['IP_lag']/p['IP_lag'].std()).astype('float32')
p['DDD']=(p['target']*(p['t_int']>=2018)*p['ipz']).astype('float32')
p['IPz']=p['ipz']

# the SC sample: event sectors AND complete positive US series 2010-2024
yrs=sorted(p['t_int'].unique())
w=p.pivot_table(index=['i','ISIC4c'],columns='t_int',values='X_us',aggfunc='first')
full_yrs=[y for y in range(2010,2025) if y in w.columns]
sc_pairs=set(w.index[(w[full_yrs]>0).all(axis=1)]) & {(i,k) for i,k in w.index if k in EV}
p['pair']=list(zip(p['i'],p['ISIC4c']))
log(f"panel {p.shape} | event sectors {len(EV)} | SC pairs {len(sc_pairs):,}")

SAMP=[("a. FULL              ", p),
      ("b. EVENT SECTORS     ", p[p['ISIC4c'].isin(EV)]),
      ("c. SC SAMPLE         ", p[p['pair'].isin(sc_pairs)])]
rows=[]
for slab,pp in SAMP:
    q=pp.copy()
    q['fe_it']=q.groupby(['i','t_int'],observed=True).ngroup().astype('int32')
    q['fe_st']=q.groupby(['ISIC4c','t_int'],observed=True).ngroup().astype('int32')
    q['fe_is']=q.groupby(['i','ISIC4c'],observed=True).ngroup().astype('int32')
    q['cl']=q['fe_is']
    print(f"\n{'='*80}\n{slab.strip()}   rows {len(q):,}  pairs {q['fe_is'].nunique():,}  "
          f"sectors {q['ISIC4c'].nunique()}\n{'='*80}")
    for lbl,y in [("(1) TOTAL",'X_tot'),("(2) US-bound",'X_us'),("(3) NON-US",'X_non')]:
        try:
            m=pf.fepois(f"{y} ~ DDD + IPz | fe_it + fe_st + fe_is", data=q,
                        vcov={"CRV1":"cl"}, iwls_maxiter=500,
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=20000),
                        lean=True, store_data=False, copy_data=False)
            t=m.tidy()
            bb,se,pv=t.loc['DDD','Estimate'],t.loc['DDD','Std. Error'],t.loc['DDD','Pr(>|t|)']
            st='***' if pv<.01 else '**' if pv<.05 else '*' if pv<.10 else ''
            print(f"  {lbl:14s} N={m._N:>10,}   DDD {bb:+.4f} ({se:.4f}){st:3s} p={pv:.4f}")
            rows.append({'sample':slab.strip(),'outcome':lbl,'coef':bb,'se':se,'p':pv,'N':int(m._N)})
            del m; gc.collect()
        except Exception as e:
            print(f"  {lbl:14s} FAILED: {type(e).__name__}: {e}")
pd.DataFrame(rows).to_csv("realloc_restricted.csv",index=False)
log("DONE")
