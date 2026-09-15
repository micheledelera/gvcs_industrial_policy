"""§5i addendum. (i) how big the §5h quartile contrast is in dose units, so the continuous
and discrete estimates can be compared, and (ii) the continuous dose interacted with
sector decoupling -- i.e. the actual research design, with no window or quartile chosen.
"""
import pandas as pd, numpy as np, pyfixest as pf, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])
Lfull=np.hstack([Lp,Lq])
d=pd.DataFrame({'u':np.repeat(np.arange(len(M)),len(YRS)),'t':np.tile(YRS,len(M)),
                'ly':Lfull.ravel()})
d['i']=CT[d['u'].values]; d['k']=SE[d['u'].values]
d['post']=(d['t']>=2018).astype(float)
dz=(M['Dec'].values-M['Dec'].values.mean())/M['Dec'].values.std()
d['dec']=dz[d['u'].values]
print(f"Chinese share of US imports, 2015-17: mean {M['Dec'].mean():.1f}%, "
      f"sd {M['Dec'].std():.1f}pp\n")
for meas,lab in [('tgt','TARGETING'),('vol','VOLUME')]:
    v=M[meas].values; sd=v.std(); pos=v[v>0]
    hi=v>=np.quantile(pos,.75); lo=v<=np.quantile(pos,.25)
    gap=(v[hi].mean()-v[lo].mean())/sd
    print(f"{lab}: HIGH mean dose {v[hi].mean()/sd:+.2f} sd, LOW {v[lo].mean()/sd:+.2f} sd "
          f"-> the §5h contrast is {gap:.2f} sd wide")
    z=(v-v.mean())/sd; d['z']=z[d['u'].values]
    for tag,fml,var in [
        ("  dose x Post, sector x yr FE ","ly ~ z:post | u + t^k","z:post"),
        ("  dose x Post, country x yr FE","ly ~ z:post | u + t^i","z:post"),
        ("  + dose x Post x decoupling  ","ly ~ z:post + z:post:dec | u + t^k","z:post:dec"),
        ("     (same spec, dose x Post) ","ly ~ z:post + z:post:dec | u + t^k","z:post"),
        ("  triple, country x yr FE     ","ly ~ z:post + z:post:dec | u + t^i","z:post:dec"),
        ("     (same spec, dose x Post) ","ly ~ z:post + z:post:dec | u + t^i","z:post")]:
        m=pf.feols(fml,data=d,vcov={'CRV1':'i'})
        b,e=m.coef()[var],m.se()[var]
        print(f"{tag} {b:>+8.4f} ({e:.4f}) t={b/e:>+5.2f}   "
              f"[{100*b:+.1f}% per sd]  implies {100*b*gap:+.1f}% over the §5h contrast"
              if var=="z:post" else
              f"{tag} {b:>+8.4f} ({e:.4f}) t={b/e:>+5.2f}   "
              f"[{100*b:+.1f}% per sd of dose x sd of decoupling]")
    print()
log("DONE")
