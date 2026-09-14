"""§5k. Anatomy of the +0.029. What is one sd, what variation identifies it, and which
country-sectors supply that variation.

The saturated spec is  ln X_ikt = a_ik + g_kt + g_it + b (IP_ik x Post_t) + e.
Because the FE are ADDITIVE, b is identified from what is left of IP_ik x Post_t after
projecting out unit, sector-year and country-year means. This residualises the regressor
exactly as the estimator does and then reports:

  the dose distribution, so "per sd" can be read in shares of HS lines
  how much of the regressor's variance survives the projection
  each unit's contribution to the numerator of b, i.e. who identifies it
  leave-one-unit-out b for the largest contributors
"""
import pandas as pd, numpy as np, pyfixest as pf, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])
Lfull=np.hstack([Lp,Lq])
v=M['tgt'].values.astype(float); sd=v.std()

print(f"\n{'='*100}\nTHE DOSE: share_frac_policies, 2015-17 mean, across all 5,166 units"
      f"\n{'='*100}")
print(f"  mean {v.mean():.5f}   sd {sd:.5f}   zeros {int((v==0).sum()):,} "
      f"({100*(v==0).mean():.0f}%)   users {int((v>0).sum()):,}")
for q in [50,75,90,95,99,99.5,100]:
    print(f"  p{q:<5} all units {np.percentile(v,q):.5f}"
          f"   |  among users {np.percentile(v[v>0],q) if q<100 else v.max():.5f}")
print(f"  mean + 1 sd = {v.mean()+sd:.5f}  ->  the {100*(v<=v.mean()+sd).mean():.1f}th "
      f"percentile of ALL units, the {100*(v[v>0]<=v.mean()+sd).mean():.1f}th among users")
print(f"  units at or above mean + 1 sd: {int((v>=v.mean()+sd).sum())} "
      f"({100*(v>=v.mean()+sd).mean():.2f}% of the sample)")

d=pd.DataFrame({'u':np.repeat(np.arange(len(M)),len(YRS)),'t':np.tile(YRS,len(M)),
                'ly':Lfull.ravel()})
d['i']=CT[d['u'].values]; d['k']=SE[d['u'].values]
d['post']=(d['t']>=2018).astype(float)
d['z']=((v-v.mean())/sd)[d['u'].values]
d['dp']=d['z']*d['post']
FE="u + t^k + t^i"
# residualise the regressor and the outcome on the FE, exactly as the estimator does.
# Own alternating projections rather than pyfixest's resid(), which drops singleton-FE rows
# and so returns a shorter vector than the panel.
gu=d['u'].values
gk=pd.factorize(pd.Series(list(zip(d['t'],d['k']))))[0]
gi=pd.factorize(pd.Series(list(zip(d['t'],d['i']))))[0]
def within(y, groups, tol=1e-11, it=2000):
    y=y.astype(float).copy()
    for _ in range(it):
        mx=0.0
        for g in groups:
            n=np.bincount(g); m=np.bincount(g,weights=y)/np.maximum(n,1)
            adj=m[g]; y-=adj; mx=max(mx,np.abs(adj).max())
        if mx<tol: break
    return y
G=[gu,gk,gi]
d['xt']=within(d['dp'].values,G); d['yt']=within(d['ly'].values,G)
b=float((d['xt']*d['yt']).sum()/(d['xt']**2).sum())
print(f"\n{'='*100}\nIDENTIFYING VARIATION\n{'='*100}")
print(f"  b recovered from residualised data: {b:+.4f}   (sc20_main reported +0.0287)")
print(f"  sd of dose x Post           {d['dp'].std():.4f}")
print(f"  sd after projecting out FE  {d['xt'].std():.4f}   -> "
      f"{100*d['xt'].var()/d['dp'].var():.1f}% of the variance survives")

num=d.groupby('u').apply(lambda g:(g['xt']*g['yt']).sum(),include_groups=False)
den=d.groupby('u').apply(lambda g:(g['xt']**2).sum(),include_groups=False)
N,D=num.sum(),den.sum()
C=pd.DataFrame({'num':num,'den':den})
C['share_den']=100*C['den']/D
C['loo_b']=(N-C['num'])/(D-C['den'])
C['dose']=v; C['ctry']=[nm(c) for c in CT]; C['sector']=SE
C['pct_dose']=100*pd.Series(v).rank(pct=True).values
top=C.sort_values('den',ascending=False).head(15)
print(f"\n{'='*100}\nWHO SUPPLIES IT: top 15 units by share of the identifying variance"
      f"\n{'='*100}")
print(f"  {'country':16s} {'sector':26s} {'dose':>8s} {'pctile':>7s} "
      f"{'% of var':>9s} {'b without it':>13s}")
for u,r in top.iterrows():
    print(f"  {r['ctry']:16s} {str(r['sector'])[:26]:26s} {r['dose']:8.4f} "
          f"{r['pct_dose']:6.1f}% {r['share_den']:8.2f}% {r['loo_b']:+12.4f}")
print(f"\n  top 15 units hold {top['share_den'].sum():.1f}% of the identifying variance; "
      f"top 50 hold {C.nlargest(50,'den')['share_den'].sum():.1f}%")
print(f"  {int((C['share_den'].sort_values(ascending=False).cumsum()<=50).sum())+1} units "
      f"hold the first 50% of it")
print(f"  effective n (Kish, 1/sum of squared variance shares): "
      f"{1/((C['den']/D)**2).sum():.0f} of {len(C):,} units")
cc=C.groupby('ctry')['den'].sum().sort_values(ascending=False)/D*100
print(f"\n  by country: "+",  ".join(f"{k} {x:.1f}%" for k,x in cc.head(8).items()))
C.to_csv("sc20d_anatomy.csv",index=False)
log("DONE")
