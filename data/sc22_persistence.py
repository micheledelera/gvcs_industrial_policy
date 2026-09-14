"""§8b. Persistence as the treatment, divorced from the instrument split.

§8 showed the promotional half of the §7 proposal is infeasible: on the §5b sample only
7.5% of country-sectors ever have a recorded subsidy, the persistence distribution decays
monotonically rather than being bimodal, and "subsidised in 6+ of 9 years" is 20 units of
which 18 are Brazil. My §7 recommendation combined the bimodality of the ANY-POLICY measure
with the promotional content of the SUBSIDY measure; those are properties of two different
variables and do not combine.

The persistence half is separable and survives. Treatment:

  TREATED   any recorded intervention in >= 6 of 2009-2017 -- persistent, pre-determined,
            nine years not three, no quantile cut, and NO country-portfolio denominator
            (this is n_policies, not share_frac_policies, so §5k's defect is absent)
  NEVER     no recorded intervention in any year 2009-2017
  dropped   the intermittent middle

Same §5b sample and the same diagnostics run up front: concentration, NEVER-subsidised
controls rather than SC-selected donors (§5i), and a JOINT pre-trend Wald test (§3m).
"""
import pandas as pd, numpy as np, pyfixest as pf, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])
PREY9=list(range(2009,2018))
raw2=pd.read_pickle("agg_for_estimation.pkl")
sp=raw2[['i','ISIC4c','t','n_policies','n_sub']].drop_duplicates(subset=['i','ISIC4c','t'])
for c in ['n_policies','n_sub']: sp[c]=pd.to_numeric(sp[c],errors='coerce').fillna(0.0)
sp['t']=sp['t'].astype(int); del raw2
P=(sp[sp['t'].isin(PREY9)].pivot_table(index=['i','ISIC4c'],columns='t',
    values='n_policies',aggfunc='sum').reindex(M.index).fillna(0.0))
yrs=(P>0).sum(axis=1).values
print(f"\n{'='*104}\nANY-POLICY PERSISTENCE on the §5b sample ({len(M):,} units), 2009-2017"
      f"\n{'='*104}")
vc=pd.Series(yrs).value_counts().sort_index()
print("  years targeted: "+"  ".join(f"{k}:{v:,}({100*v/len(M):.1f}%)" for k,v in vc.items()))
for THR in [9,8,7,6,5]:
    tr=(yrs>=THR); nv=(yrs==0)
    print(f"  threshold {THR}+: treated {int(tr.sum()):5d} ({len(set(CT[tr]))} countries, "
          f"{len(set(SE[tr]))} sectors)  never {int(nv.sum()):,}  dropped {int((~tr&~nv).sum()):,}")
THR=6
TRT=(yrs>=THR); NEV=(yrs==0)
cc=pd.Series([nm(c) for c in CT[TRT]]).value_counts()
print(f"\n  -> {THR}+ of 9. Treated by country: "+",  ".join(
    f"{k} {v}" for k,v in cc.head(10).items()))
print(f"  concentration of the treated GROUP: largest country "
      f"{100*cc.iloc[0]/TRT.sum():.0f}%, top 3 {100*cc.head(3).sum()/TRT.sum():.0f}%, "
      f"{len(cc)} countries in total")

Lfull=np.hstack([Lp,Lq]); keep=TRT|NEV
d=pd.DataFrame({'u':np.repeat(np.arange(len(M)),len(YRS)),'t':np.tile(YRS,len(M)),
                'ly':Lfull.ravel()})
d['i']=CT[d['u'].values]; d['k']=SE[d['u'].values]
d['trt']=TRT.astype(float)[d['u'].values]
d['post']=(d['t']>=2018).astype(float); d['dp']=d['trt']*d['post']
d=d[keep[d['u'].values]].copy()
print(f"\n{'='*104}\nDiD: treated {int(TRT.sum()):,} vs never-targeted {int(NEV.sum()):,}"
      f"   {len(d):,} obs, {d['i'].nunique()} countries\n{'='*104}")
for fname,fe in [("unit + year","u + t"),("+ sector x year","u + t^k"),
                 ("+ country x year","u + t^i"),("both","u + t^k + t^i")]:
    m=pf.feols(f"ly ~ dp | {fe}",data=d,vcov={'CRV1':'i'})
    b,s=m.coef()['dp'],m.se()['dp']
    print(f"    {fname:18s} {b:>+8.4f} ({s:.4f}) t={b/s:>+5.2f}")
m=pf.feols("ly ~ dp | u + t^k",data=d,vcov={'CRV1':'i + k'})
b,s=m.coef()['dp'],m.se()['dp']
print(f"    two-way cluster (country & sector), + sector x year: {b:+.4f} ({s:.4f}) t={b/s:+.2f}")

for y in YRS:
    if y!=2017: d[f"y{y}"]=d['trt']*(d['t']==y)
rhs=" + ".join(f"y{y}" for y in YRS if y!=2017); ys=[y for y in YRS if y!=2017]
PT=[y for y in PRE if y!=2017]; EV=[]
print(f"\n  event study (2017 omitted) with a JOINT Wald test on the ten pre-2018 coefs")
for fname,fe in [("sector x year","u + t^k"),("both","u + t^k + t^i")]:
    m=pf.feols(f"ly ~ {rhs} | {fe}",data=d,vcov={'CRV1':'i'})
    co,se=m.coef(),m.se()
    b=np.array([co[f"y{y}"] for y in ys]); s=np.array([se[f"y{y}"] for y in ys])
    names=list(co.index); R=np.zeros((len(PT),len(names)))
    for r,y in enumerate(PT): R[r,names.index(f"y{y}")]=1.0
    p=float(np.asarray(m.wald_test(R=R)['pvalue']).ravel()[0])
    pre=np.array([b[ys.index(y)] for y in PT]); pst=np.array([b[ys.index(y)] for y in POST])
    print(f"    {fname:15s} "+" ".join(f"{y}:{b[ys.index(y)]:+.2f}" for y in ys))
    print(f"    {'':15s} pre mean {pre.mean():+.3f}  post mean {pst.mean():+.3f}  "
          f"JOINT p = {p:.4f}  {'*** REJECTS' if p<0.10 else 'PASSES'}")
    for y,bb,ss in zip(ys,b,s): EV.append(dict(fe=fname,t=y,b=bb,s=ss))
    EV.append(dict(fe=fname,t=2017,b=0.0,s=0.0))
pd.DataFrame(EV).to_csv("sc22_event.csv",index=False)

gk=pd.factorize(pd.Series(list(zip(d['t'],d['k']))))[0]
gi=pd.factorize(pd.Series(list(zip(d['t'],d['i']))))[0]
def within(y,groups,tol=1e-11,it=3000):
    y=y.astype(float).copy()
    for _ in range(it):
        mx=0.0
        for g in groups:
            n=np.bincount(g); mm=np.bincount(g,weights=y)/np.maximum(n,1)
            adj=mm[g]; y-=adj; mx=max(mx,np.abs(adj).max())
        if mx<tol: break
    return y
G=[pd.factorize(d['u'].values)[0],gk,gi]
d['xt']=within(d['dp'].values,G); d['yt']=within(d['ly'].values,G)
bb=float((d['xt']*d['yt']).sum()/(d['xt']**2).sum())
num=d.groupby('u').apply(lambda g:(g['xt']*g['yt']).sum(),include_groups=False)
den=d.groupby('u').apply(lambda g:(g['xt']**2).sum(),include_groups=False)
N,D=num.sum(),den.sum()
C=pd.DataFrame({'num':num,'den':den}); C['sh']=100*C['den']/D
C['loo']=(N-C['num'])/(D-C['den'])
C['ctry']=[nm(CT[u]) for u in C.index]; C['trt']=TRT[C.index.values]
print(f"\n{'='*104}\nCONCENTRATION (saturated b = {bb:+.4f})\n{'='*104}")
print(f"  Kish effective n: {1/((C['den']/D)**2).sum():.0f} of {len(C):,} units")
print(f"  top 15 units hold {C.nlargest(15,'den')['sh'].sum():.1f}%; "
      f"top 50 hold {C.nlargest(50,'den')['sh'].sum():.1f}%; "
      f"top 1% hold {C.nlargest(max(1,len(C)//100),'den')['sh'].sum():.1f}%")
byc=C.groupby('ctry')['den'].sum().sort_values(ascending=False)/D*100
print(f"  by country: "+",  ".join(f"{k} {v:.1f}%" for k,v in byc.head(8).items()))
C.to_csv("sc22_concentration.csv",index=False)
log("DONE")
