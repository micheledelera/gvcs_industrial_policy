"""§5j addendum 2. Is the continuous dose-response tail-driven?

Under the saturated FE the quintile dummies are all about zero (Q1 +0.106 ... Q5 +0.023,
ses ~0.09) while the CONTINUOUS dose gives +0.029 (t = 2.24) and the dose among users
+0.046 (t = 2.30). Both cannot be true of a coarse step function, so the continuous effect
must live in WITHIN-quintile variation -- i.e. in the upper tail of the share distribution,
which is exactly the "handful of observations" problem that has recurred since §6.

Tests, all with unit + sector x year + country x year FE:
  winsorising the dose at the 99th, 95th and 90th percentile of the positive distribution
  the percentile-rank dose, which is tail-insensitive by construction
  dropping the top 1% and top 5% of dosed units outright
  a decile-of-positive step function, finer than quintiles
"""
import pandas as pd, numpy as np, pyfixest as pf, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])
ALL=[]
Lfull=np.hstack([Lp,Lq]); FE="u + t^k + t^i"

for meas,lab in [('tgt','TARGETING'),('vol','VOLUME')]:
    v=M[meas].values.astype(float); pos=v>0
    print(f"\n{'='*100}\n{lab}   dose skew: mean {v[pos].mean():.4f}, p50 "
          f"{np.median(v[pos]):.4f}, p95 {np.quantile(v[pos],.95):.4f}, "
          f"p99 {np.quantile(v[pos],.99):.4f}, max {v.max():.4f}\n{'='*100}")
    base=pd.DataFrame({'u':np.repeat(np.arange(len(M)),len(YRS)),'t':np.tile(YRS,len(M)),
                       'ly':Lfull.ravel()})
    base['i']=CT[base['u'].values]; base['k']=SE[base['u'].values]
    base['post']=(base['t']>=2018).astype(float)
    ROWS=[]
    def run(vec, keep=None, tag=""):
        d=base.copy()
        z=(vec-vec.mean())/vec.std(); d['z']=z[d['u'].values]
        if keep is not None: d=d[keep[d['u'].values]]
        m=pf.feols(f"ly ~ z:post | {FE}",data=d,vcov={'CRV1':'i'})
        b,s=m.coef()['z:post'],m.se()['z:post']
        print(f"    {tag:34s} {b:>+8.4f} ({s:.4f}) t={b/s:>+5.2f}   "
              f"({d['u'].nunique():,} units)")
        ROWS.append(dict(meas=meas,kind='spec',tag=tag,b=b,s=s,n=int(d['u'].nunique())))
    run(v, None, "dose in sd (headline)")
    for p in [99,95,90]:
        run(np.minimum(v,np.quantile(v[pos],p/100)), None, f"winsorised at p{p} of positive")
    r=pd.Series(v).rank(pct=True).values
    run(r, None, "percentile rank")
    for p in [99,95]:
        thr=np.quantile(v[pos],p/100); keep=~(v>thr)
        run(v, keep, f"top {100-p}% of dosed units dropped")
    # decile step function
    dq=np.quantile(v[pos],np.arange(1,10)/10)
    db=np.digitize(v,np.concatenate([[1e-12],dq]))
    d=base.copy()
    for b_ in range(1,11): d[f'D{b_}']=(db==b_).astype(float)[d['u'].values]
    m=pf.feols("ly ~ "+" + ".join(f"D{b_}:post" for b_ in range(1,11))+f" | {FE}",
               data=d,vcov={'CRV1':'i'})
    co,se=m.coef(),m.se()
    print("    deciles of positive (zero omitted; blank = dropped for collinearity):")
    got=[(b_,co.get(f'D{b_}:post',np.nan),se.get(f'D{b_}:post',np.nan)) for b_ in range(1,11)]
    print("      "+"  ".join(f"D{b_}{bb:+.2f}" if np.isfinite(bb) else f"D{b_}  n/a"
                             for b_,bb,ss in got))
    print("      "+"  ".join(f"({ss:.2f})" if np.isfinite(ss) else "     "
                             for b_,bb,ss in got))
    for b_,bb,ss in got:
        ROWS.append(dict(meas=meas,kind='decile',tag=f"D{b_}",b=bb,s=ss,
                         n=int((db==b_).sum())))
    ALL.extend(ROWS)
pd.DataFrame(ALL).to_csv("sc20c_tail.csv",index=False)
log("DONE")
