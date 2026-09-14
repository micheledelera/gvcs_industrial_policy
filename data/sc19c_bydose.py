"""§5i addendum 2. Pre-2018 export growth by policy bin, to locate the §5h pre-trend.

§5i test A found the CONTINUOUS dose does not predict pre-2018 growth (t = 0.08), while
§5h's top-vs-bottom-quartile contrast showed a large declining pre-trend. Both cannot be
describing the same monotone relationship. This prints pre-period growth by bin so the
non-monotonicity is visible rather than asserted.
"""
import pandas as pd, numpy as np, pyfixest as pf
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])
Lall=np.hstack([Lp,Lq])
pre=Lall[:,[YRS.index(y) for y in BASE]].mean(axis=1)-Lall[:,0]          # 2007 -> 2015-17
post=Lall[:,[YRS.index(y) for y in [2022,2023,2024]]].mean(axis=1)-Lall[:,[YRS.index(y) for y in BASE]].mean(axis=1)
for meas,lab in [('tgt','TARGETING (share_frac_policies)'),('vol','VOLUME (n_policies)')]:
    v=M[meas].values; pos=v[v>0]
    q=np.quantile(pos,[.25,.50,.75])
    bins=np.where(v<=0,0,np.where(v<=q[0],1,np.where(v<=q[1],2,np.where(v<=q[2],3,4))))
    NB={0:'zero policy',1:'Q1 (lowest positive)',2:'Q2',3:'Q3',4:'Q4 (highest)'}
    df=pd.DataFrame({'b':bins,'pre':pre,'post':post,'i':CT})
    print(f"\n{lab}\n{'-'*82}")
    print(f"  {'bin':22s} {'n':>6s}  {'d lnX 2007->2015-17':>21s}  {'d lnX 2015-17->2022-24':>23s}")
    for b in range(5):
        s=df[df['b']==b]
        def jkm(col):
            cs=s['i'].unique(); m=s[col].mean()
            a=np.array([s[s['i']!=c][col].mean() for c in cs])
            return m,np.sqrt((len(cs)-1)/len(cs)*((a-a.mean())**2).sum())
        p1,e1=jkm('pre'); p2,e2=jkm('post')
        print(f"  {NB[b]:22s} {len(s):6d}   {p1:+8.3f} ({e1:.3f})      "
              f"{p2:+8.3f} ({e2:.3f})")
    zz=df[df['b']==0]; q4=df[df['b']==4]; q1=df[df['b']==1]
    print(f"  Q4 - Q1(positive): pre {q4['pre'].mean()-q1['pre'].mean():+.3f}   "
          f"post {q4['post'].mean()-q1['post'].mean():+.3f}")
    print(f"  Q4 - zero        : pre {q4['pre'].mean()-zz['pre'].mean():+.3f}   "
          f"post {q4['post'].mean()-zz['post'].mean():+.3f}")
    print(f"  §5h LOW group is {100*(df[df['b'].isin([0,1])]['b']==0).mean():.0f}% "
          f"zero-policy units")
