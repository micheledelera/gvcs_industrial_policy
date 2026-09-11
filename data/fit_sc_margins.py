"""Does the synthetic-control estimator reproduce §3s's reallocation finding?

§3s used a country-sector-year PPML triple difference and found exports to the US rise
(+0.0175), exports elsewhere fall (−0.0081), and total exports do not increase
(−0.0059). If the SC design -- a completely different estimator, different sample,
different functional form -- independently reproduces US up / non-US down / total flat,
the reallocation reading is confirmed by two unrelated methods, and it explains why
§3aa's +0.21 on US exports is so much larger than gravity's 5.2% differential.

Same estimator as §3aa at its preferred settings (MVA_W = 2, zeta -> 0, intercept fitted
over path rows only), run on three outcomes on the IDENTICAL sample of pairs so the
comparison is clean:

  US       log US exports
  TOTAL    log exports to the world
  NON-US   log exports to all destinations except the US

Placebo for each: (2015-17) - (2010-12), entirely pre-treatment.
"""
import pandas as pd, numpy as np, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("fit_sc_mvasweep2.py").read().split("def run(mw):")[0])

USA_=842
raw2=pd.read_pickle("agg_for_estimation.pkl")
raw2['i']=raw2['i'].astype('int32'); raw2['j']=raw2['j'].astype('int32')
raw2['t']=raw2['t'].astype('int32')
tot=raw2.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
tot.columns=[int(c) for c in tot.columns]; tot=tot[sorted(tot.columns)]
del raw2; gc.collect()
TOT=tot.reindex(m.index); US=np.exp(Yv); NON=(TOT-US).clip(lower=0)
bad=(TOT<=0).any(axis=1) | (NON<=0).any(axis=1)
print(f"pairs dropped for a non-positive TOTAL or NON-US year: {int(bad.sum())} of {len(m)}")
keep=~bad
mm=m[keep].copy()
OUT={'US':np.log(US[keep]), 'TOTAL':np.log(TOT[keep]), 'NON-US':np.log(NON[keep])}
Ypath=OUT['US']                     # path matching always on US exports, as in §3aa
print(f"estimation sample: {len(mm):,} pairs | treated {int(mm['T'].sum()):,} | "
      f"sectors {mm.index.get_level_values('ISIC4c').nunique()}")
print(f"US share of these pairs' exports, pre-period: "
      f"{US[keep][PRE].mean(axis=1).sum()/TOT[keep][PRE].mean(axis=1).sum():.3f}\n")

MW=2.0
def run(Yout, col_post, col_plac):
    out=[]
    for k,g in mm.groupby(level='ISIC4c', observed=True):
        t_=g[g['T']==1]; d_=g[g['T']==0]
        if len(t_)==0 or len(d_)<MIN_DON: continue
        P=Ypath.loc[d_.index,PATH].values; bt=Ypath.loc[t_.index,PATH].values.mean(axis=0)
        A=np.hstack([P,(MW*d_['mvaz'].values)[:,None]]); b=np.append(bt, MW*t_['mvaz'].mean())
        sig=np.diff(P,axis=1).std()
        zeta2=(((len(t_)*len(POST))**0.5)*(sig**2))*1e-6+1e-10
        w,_=solve_w2(A,b,zeta2,len(PATH))
        dp=Yout[POST].mean(axis=1)-Yout[PRE].mean(axis=1)
        dl=Yout[PRE].mean(axis=1)-Yout[PRE0].mean(axis=1)
        out.append({'k':k,'tau':dp[t_.index].mean()-float(w@dp[d_.index].values),
                    'plac':dl[t_.index].mean()-float(w@dl[d_.index].values)})
    return pd.DataFrame(out)

def jk(o,col):
    n=len(o); mu=o[col].mean()
    a=np.array([o[col].drop(i).mean() for i in o.index])
    se=np.sqrt((n-1)/n*((a-a.mean())**2).sum())
    return mu, se, (mu/se if se>0 else np.nan)

from scipy import stats
print("="*88)
print(f"SC ON THREE MARGINS   MVA_W = {MW}, zeta -> 0, identical sample")
print("="*88)
print(f"{'outcome':>10s}{'cells':>7s}{'TRUE tau':>26s}{'PLACEBO tau':>26s}")
rows=[]
for lab,Yo in OUT.items():
    o=run(Yo,'tau','plac')
    if len(o)<3: print(f"{lab:>10s}  too few cells"); continue
    a=jk(o,'tau'); b=jk(o,'plac')
    f=lambda r: (f"{r[0]:+.4f} ({r[1]:.4f}) t={r[2]:+.2f}")
    print(f"{lab:>10s}{len(o):>7d}{f(a):>26s}{f(b):>26s}")
    rows.append({'outcome':lab,'cells':len(o),'tau':a[0],'se':a[1],'t':a[2],
                 'p':2*(1-stats.norm.cdf(abs(a[2]))),
                 'plac':b[0],'plac_se':b[1],'plac_t':b[2]})
r=pd.DataFrame(rows); r.to_csv("sc_margins.csv",index=False)
if len(r)==3:
    sh=US[keep][PRE].mean(axis=1).sum()/TOT[keep][PRE].mean(axis=1).sum()
    us_=r.set_index('outcome').loc['US','tau']; nn=r.set_index('outcome').loc['NON-US','tau']
    print(f"\nimplied total from the two components: {sh:.3f} x {us_:+.4f} + "
          f"{1-sh:.3f} x {nn:+.4f} = {sh*us_+(1-sh)*nn:+.4f}")
    print(f"estimated TOTAL: {r.set_index('outcome').loc['TOTAL','tau']:+.4f}")
log("DONE")
