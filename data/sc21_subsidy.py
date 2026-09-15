"""§8. Persistent PROMOTIONAL policy as the treatment. Build, characterise, and the DiD.

§7 established three things. The GTA measures used so far correlate 0.24-0.72 with each
other; `n_sub` is a subsidy VALUE, nearly orthogonal to intervention counts (0.178 in
logs), and the only field that measures promotional effort in money; and targeting
persistence over 2009-2017 is bimodal, so a persistent-vs-never split is a natural binary
rather than an arbitrary quantile.

This builds the treatment those facts point to:

  TREATED   subsidy recorded in most years 2009-2017 -- promotional instrument, persistent,
            pre-determined, no arbitrary cut, no country-portfolio denominator
  NEVER     no subsidy in any year 2009-2017
  dropped   the intermittent middle

and then, BEFORE estimating anything, runs the diagnostics that voided §5j/§5k:
  who is in the treated group, and how concentrated it is
  the DiD with NEVER-subsidised controls -- never SC-selected donors (§5i's lesson)
  the event study with a JOINT Wald test on the pre-period (§3m's lesson)
  the Kish effective n and the top contributors to the estimate (§5k's lesson)

Sample and matching held to §5b so everything stays comparable.
"""
import pandas as pd, numpy as np, pyfixest as pf, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])
PREY9=list(range(2009,2018))

# ---- subsidy panel on the §5b units ------------------------------------------------------
raw2=pd.read_pickle("agg_for_estimation.pkl")
sp=(raw2[['i','ISIC4c','t','n_sub','n_policies']].drop_duplicates(subset=['i','ISIC4c','t']))
for c in ['n_sub','n_policies']: sp[c]=pd.to_numeric(sp[c],errors='coerce').fillna(0.0)
sp['t']=sp['t'].astype(int)
del raw2
S=(sp[sp['t'].isin(PREY9)].pivot_table(index=['i','ISIC4c'],columns='t',values='n_sub',
                                       aggfunc='sum').reindex(M.index).fillna(0.0))
P=(sp[sp['t'].isin(PREY9)].pivot_table(index=['i','ISIC4c'],columns='t',values='n_policies',
                                       aggfunc='sum').reindex(M.index).fillna(0.0))
yrs_sub=(S>0).sum(axis=1).values
yrs_pol=(P>0).sum(axis=1).values
val=S.mean(axis=1).values                       # mean annual subsidy value 2009-17

print(f"\n{'='*104}\nSUBSIDY PERSISTENCE on the §5b sample ({len(M):,} country-sectors), "
      f"2009-2017\n{'='*104}")
vc=pd.Series(yrs_sub).value_counts().sort_index()
print("  years with a recorded subsidy: "+"  ".join(
    f"{k}:{v:,}({100*v/len(M):.1f}%)" for k,v in vc.items()))
print(f"  ever subsidised {100*(yrs_sub>0).mean():.1f}%   "
      f"6+ years {100*(yrs_sub>=6).mean():.1f}%   all 9 {100*(yrs_sub==9).mean():.1f}%")
print(f"  for comparison, ANY policy: ever {100*(yrs_pol>0).mean():.1f}%, "
      f"6+ years {100*(yrs_pol>=6).mean():.1f}%")

for THR in [9,8,7,6,5]:
    tr=(yrs_sub>=THR); nv=(yrs_sub==0)
    print(f"  threshold {THR}+ of 9: treated {int(tr.sum()):4d} "
          f"({len(set(CT[tr]))} countries, {len(set(SE[tr]))} sectors), "
          f"never {int(nv.sum()):,}, dropped {int((~tr&~nv).sum()):,}")
THR=6
TRT=(yrs_sub>=THR); NEV=(yrs_sub==0)
print(f"\n  -> using {THR}+ of 9 years. Treated units also carrying non-subsidy policy: "
      f"{100*(yrs_pol[TRT]>yrs_sub[TRT]).mean():.0f}%.  "
      f"Never-subsidised donors that DO have other policy: "
      f"{100*(yrs_pol[NEV]>0).mean():.0f}%")

cc=pd.Series(CT[TRT]).map(lambda c: nm(c)).value_counts()
ss=pd.Series(SE[TRT]).astype(str).value_counts()
print(f"\n  treated by country: "+",  ".join(f"{k} {v}" for k,v in cc.head(10).items()))
print(f"  treated by sector:  "+",  ".join(f"{k} {v}" for k,v in ss.head(10).items()))
print(f"  concentration: largest country {100*cc.iloc[0]/TRT.sum():.0f}% of treated, "
      f"largest sector {100*ss.iloc[0]/TRT.sum():.0f}%; "
      f"top 3 countries {100*cc.head(3).sum()/TRT.sum():.0f}%")
print(f"  subsidy value among treated: median {np.median(val[TRT]):,.0f}, "
      f"p90 {np.percentile(val[TRT],90):,.0f}, max {val[TRT].max():,.0f}")

# ---- the panel ---------------------------------------------------------------------------
Lfull=np.hstack([Lp,Lq])
keep=TRT|NEV
d=pd.DataFrame({'u':np.repeat(np.arange(len(M)),len(YRS)),'t':np.tile(YRS,len(M)),
                'ly':Lfull.ravel()})
d['i']=CT[d['u'].values]; d['k']=SE[d['u'].values]
d['trt']=TRT.astype(float)[d['u'].values]
d['post']=(d['t']>=2018).astype(float)
d['dp']=d['trt']*d['post']
d=d[keep[d['u'].values]]
print(f"\n{'='*104}\nDiD: treated {int(TRT.sum())} vs never-subsidised {int(NEV.sum()):,}"
      f"   {len(d):,} obs, {d['i'].nunique()} countries\n{'='*104}")
for fname,fe in [("unit + year","u + t"),("+ sector x year","u + t^k"),
                 ("+ country x year","u + t^i"),("both","u + t^k + t^i")]:
    m=pf.feols(f"ly ~ dp | {fe}",data=d,vcov={'CRV1':'i'})
    b,s=m.coef()['dp'],m.se()['dp']
    print(f"    {fname:18s} {b:>+8.4f} ({s:.4f}) t={b/s:>+5.2f}")
m=pf.feols("ly ~ dp | u + t^k",data=d,vcov={'CRV1':'i + k'})
b,s=m.coef()['dp'],m.se()['dp']
print(f"    two-way cluster (country & sector), + sector x year: {b:+.4f} ({s:.4f}) t={b/s:+.2f}")

# ---- event study + JOINT pre-trend test -------------------------------------------------
for y in YRS:
    if y!=2017: d[f"y{y}"]=d['trt']*(d['t']==y)
rhs=" + ".join(f"y{y}" for y in YRS if y!=2017); ys=[y for y in YRS if y!=2017]
PRETEST=[y for y in PRE if y!=2017]
EV=[]
print(f"\n  event study (2017 omitted), with a JOINT Wald test on the ten pre-2018 coefs")
for fname,fe in [("sector x year","u + t^k"),("both","u + t^k + t^i")]:
    m=pf.feols(f"ly ~ {rhs} | {fe}",data=d,vcov={'CRV1':'i'})
    co,se=m.coef(),m.se()
    b=np.array([co[f"y{y}"] for y in ys]); s=np.array([se[f"y{y}"] for y in ys])
    names=list(co.index); R=np.zeros((len(PRETEST),len(names)))
    for r,y in enumerate(PRETEST): R[r,names.index(f"y{y}")]=1.0
    w=m.wald_test(R=R); p=float(np.asarray(w['pvalue']).ravel()[0])
    pre=np.array([b[ys.index(y)] for y in PRETEST]); pst=np.array([b[ys.index(y)] for y in POST])
    print(f"    {fname:15s} "+" ".join(f"{y}:{b[ys.index(y)]:+.2f}" for y in ys))
    print(f"    {'':15s} pre mean {pre.mean():+.3f}  post mean {pst.mean():+.3f}  "
          f"JOINT pre-trend p = {p:.4f}  "
          f"{'*** REJECTS' if p<0.10 else 'passes'}")
    for y,bb,ss in zip(ys,b,s): EV.append(dict(fe=fname,t=y,b=bb,s=ss))
    EV.append(dict(fe=fname,t=2017,b=0.0,s=0.0))
pd.DataFrame(EV).to_csv("sc21_event.csv",index=False)

# ---- concentration diagnostic (the §5k check, up front) ---------------------------------
gu=d['u'].values
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
G=[pd.factorize(gu)[0],gk,gi]
d['xt']=within(d['dp'].values,G); d['yt']=within(d['ly'].values,G)
bb=float((d['xt']*d['yt']).sum()/(d['xt']**2).sum())
num=d.groupby('u')[['xt','yt']].apply(lambda g:(g['xt']*g['yt']).sum())
den=d.groupby('u')[['xt']].apply(lambda g:(g['xt']**2).sum())
N,D=num.sum(),den.sum()
C=pd.DataFrame({'num':num,'den':den}); C['sh']=100*C['den']/D
C['loo']=(N-C['num'])/(D-C['den'])
C['ctry']=[nm(CT[u]) for u in C.index]; C['sector']=[str(SE[u]) for u in C.index]
C['trt']=TRT[C.index.values]
print(f"\n{'='*104}\nCONCENTRATION of the saturated estimate (b = {bb:+.4f})\n{'='*104}")
print(f"  Kish effective n: {1/((C['den']/D)**2).sum():.0f} of {len(C):,} units")
print(f"  top 15 units hold {C.nlargest(15,'den')['sh'].sum():.1f}% of the identifying "
      f"variance; top 50 hold {C.nlargest(50,'den')['sh'].sum():.1f}%")
byc=C.groupby('ctry')['den'].sum().sort_values(ascending=False)/D*100
print(f"  by country: "+",  ".join(f"{k} {v:.1f}%" for k,v in byc.head(8).items()))
print(f"\n  {'country':16s} {'sector':10s} {'trt':>4s} {'% var':>7s} {'b without':>10s}")
for u,r in C.nlargest(10,'den').iterrows():
    print(f"  {r['ctry']:16s} {r['sector']:10s} {int(r['trt']):4d} {r['sh']:6.2f}% "
          f"{r['loo']:+9.4f}")
C.to_csv("sc21_concentration.csv",index=False)
log("DONE")
