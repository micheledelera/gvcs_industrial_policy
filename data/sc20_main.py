"""§5j. The recommended design, run properly: CONTINUOUS dose, PRE-DETERMINED in a window.

§5i established the two choices. Dose continuous (no quartile cut, which turned out to be
a tail phenomenon 2.5-4x the dose-response) and pre-determined (fixed at its 2015-17 mean,
so the 2018 event stays exogenous to the unit and the policy reaction function documented
for volume cannot contaminate it). This runs that specification as the main result rather
than as a diagnostic.

    ln X_ikt = alpha_ik + gamma_t + beta (IP_ik x Post_t) + e_ikt ,   Post = 1[t >= 2018]

IP_ik standardised, so beta is the effect of one sd more policy. Then, in turn:

  1  FE ladder          unit+year; +sector x year; +country x year; both
  2  decoupling triple  beta1 (IP x Post) + beta2 (IP x Post x Dec), Dec = China's 2015-17
                        share of US imports in the sector, standardised -- the actual
                        research design: does policy pay off MORE where decoupling bit?
  3  functional form    dose in sd, in ln(1+x), and as a within-sample percentile rank,
                        because a count and a share are both right-skewed
  4  dose shape         quintiles of the POSITIVE distribution, zero-policy as the omitted
                        category -- is the response linear, or all in the top bin?
  5  event study        year-by-year, with and without country x year FE
  6  samples            the §5b balanced panel (main) and an unbalanced panel that only
                        requires positive exports in 2015-17, since the balanced
                        requirement selects on surviving to 2024

Clustered on country throughout; two-way country and sector reported for the headline.
"""
import pandas as pd, numpy as np, pyfixest as pf, time, gc
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])

def mkpanel(Lm, idx, ct, se, dec, doses):
    n,T=Lm.shape
    d=pd.DataFrame({'u':np.repeat(np.arange(n),T),'t':np.tile(YRS,n),'ly':Lm.ravel()})
    d=d[np.isfinite(d['ly'])]
    d['i']=ct[d['u'].values]; d['k']=se[d['u'].values]
    d['post']=(d['t']>=2018).astype(float)
    d['dec']=((dec-np.nanmean(dec))/np.nanstd(dec))[d['u'].values]
    for nm_,v in doses.items(): d[nm_]=v[d['u'].values]
    return d

def ladder(d, var, tag, extra=""):
    rhs=f"{var}:post"+(f" + {extra}" if extra else "")
    rows=[]
    for fname,fe in [("unit + year","u + t"),("+ sector x year","u + t^k"),
                     ("+ country x year","u + t^i"),("both","u + t^k + t^i")]:
        m=pf.feols(f"ly ~ {rhs} | {fe}",data=d,vcov={'CRV1':'i'})
        co,se_=m.coef(),m.se()
        out=f"    {tag:28s} {fname:18s}"
        for v in [f"{var}:post"]+([e.strip() for e in extra.split("+")] if extra else []):
            if v in co.index:
                b,s=co[v],se_[v]
                out+=f"   {v.replace(var,'dose'):>22s} {b:>+7.4f} ({s:.4f}) t={b/s:>+5.2f}"
        print(out)
        rows.append(dict(tag=tag,fe=fname,b=co[f"{var}:post"],s=se_[f"{var}:post"]))
    return rows

def build(balanced=True):
    """rebuild the export panel, optionally without the complete-POST requirement"""
    raw=pd.read_pickle("agg_for_estimation.pkl")
    raw['i']=raw['i'].astype('int32'); raw['t']=raw['t'].astype('int32')
    advl=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
    XX=(raw[raw['j']==USA].groupby(['i','ISIC4c','t'],observed=True)['imports'].sum()
        .unstack('t'))
    del raw; gc.collect()
    XX.columns=[int(c) for c in XX.columns]; XX=XX.reindex(columns=YRS)
    XX=XX.loc[[(i,k) for i,k in XX.index if advl.get(i,1)==0 and i!=CHINA]]
    if balanced: XX=XX[(XX[YRS]>0).all(axis=1) & XX[YRS].notna().all(axis=1)]
    else:        XX=XX[(XX[BASE]>0).all(axis=1) & XX[BASE].notna().all(axis=1)]
    return XX

ROWS=[]
for meas,lab in [('tgt','TARGETING (share of HS lines covered)'),
                 ('vol','VOLUME (number of interventions)')]:
    v=M[meas].values; sd=v.std()
    doses={'z':(v-v.mean())/sd,
           'zl':(np.log1p(v)-np.log1p(v).mean())/np.log1p(v).std(),
           'zr':(pd.Series(v).rank(pct=True).values-0.5)/pd.Series(v).rank(pct=True).std()}
    pos=v[v>0]; qs=np.quantile(pos,[.2,.4,.6,.8])
    q=np.digitize(v,np.concatenate([[1e-12],qs]))       # 0 = zero policy, 1..5 quintiles
    for b in range(1,6): doses[f'q{b}']=(q==b).astype(float)
    d=mkpanel(np.hstack([Lp,Lq]),M.index,CT,SE,M['Dec'].values,doses)
    print(f"\n{'='*118}\n{lab}   {d['u'].nunique():,} units, {len(d):,} obs, "
          f"{d['i'].nunique()} countries, {d['k'].nunique()} sectors\n{'='*118}")

    print("  1  FE ladder, dose in sd")
    ROWS+= [dict(meas=meas,**r) for r in ladder(d,'z',"dose x Post")]
    m=pf.feols("ly ~ z:post | u + t^k",data=d,vcov={'CRV1':'i + k'})
    b,s=m.coef()['z:post'],m.se()['z:post']
    print(f"      two-way cluster (country & sector), + sector x year: "
          f"{b:+.4f} ({s:.4f}) t={b/s:+.2f}")

    print("\n  2  decoupling triple")
    ROWS+=[dict(meas=meas,**r) for r in ladder(d,'z',"triple",extra="z:post:dec")]

    print("\n  3  functional form (+ sector x year FE)")
    for var,fn in [('z','dose in sd'),('zl','ln(1 + dose)'),('zr','percentile rank')]:
        m=pf.feols(f"ly ~ {var}:post | u + t^k",data=d,vcov={'CRV1':'i'})
        b,s=m.coef()[f"{var}:post"],m.se()[f"{var}:post"]
        print(f"    {fn:22s} {b:>+8.4f} ({s:.4f}) t={b/s:>+5.2f}")

    print("\n  4  dose shape: quintiles of the positive distribution, zero-policy omitted")
    rhs=" + ".join(f"q{b}:post" for b in range(1,6))
    for fname,fe in [("+ sector x year","u + t^k"),("+ country x year","u + t^i")]:
        m=pf.feols(f"ly ~ {rhs} | {fe}",data=d,vcov={'CRV1':'i'})
        co,se_=m.coef(),m.se()
        cnt=[int((q==b).sum()) for b in range(1,6)]
        print(f"    {fname:18s} "+"  ".join(
            f"Q{b}(n={cnt[b-1]}) {co[f'q{b}:post']:+.3f}({se_[f'q{b}:post']:.3f})"
            for b in range(1,6)))

    print("\n  5  event study, continuous dose, 2017 omitted")
    for y in YRS:
        if y!=2017: d[f"z{y}"]=d['z']*(d['t']==y)
    rhs=" + ".join(f"z{y}" for y in YRS if y!=2017)
    for fname,fe in [("sector x year","u + t^k"),("+ country x year","u + t^k + t^i")]:
        m=pf.feols(f"ly ~ {rhs} | {fe}",data=d,vcov={'CRV1':'i'})
        co,se_=m.coef(),m.se()
        ys=[y for y in YRS if y!=2017]
        b=np.array([co[f"z{y}"] for y in ys]); s=np.array([se_[f"z{y}"] for y in ys])
        pre=np.array([b[ys.index(y)] for y in PRE if y!=2017])
        pst=np.array([b[ys.index(y)] for y in POST])
        mt=np.abs(pre/np.array([s[ys.index(y)] for y in PRE if y!=2017])).max()
        print(f"    {fname:16s} "+" ".join(f"{y}:{b[ys.index(y)]:+.3f}" for y in YRS if y!=2017))
        print(f"    {'':16s} pre mean {pre.mean():+.4f} (max|t| {mt:.2f})   "
              f"post mean {pst.mean():+.4f}")
        for y,bb,ss in zip(ys,b,s):
            ROWS.append(dict(meas=meas,tag='event',fe=fname,t=y,b=bb,s=ss))
        ROWS.append(dict(meas=meas,tag='event',fe=fname,t=2017,b=0.0,s=0.0))

# ---- 6. unbalanced sample ---------------------------------------------------------------
log("rebuilding the unbalanced panel")
XU=build(balanced=False)
LU=np.log(XU[YRS].where(XU[YRS]>0))
iu=list(XU.index); ctu=np.array([a for a,_ in iu]); seu=np.array([b for _,b in iu])
ipu=ip.reindex(iu)
decu=np.array([DEC.get(k,np.nan) for _,k in iu])
print(f"\n{'='*118}\n6  UNBALANCED sample: {len(XU):,} units (vs {len(M):,} balanced), "
      f"{int(np.isfinite(LU.values).sum()):,} obs\n{'='*118}")
for meas,col,lab in [('tgt','share_frac_policies','TARGETING'),
                     ('vol','n_policies','VOLUME')]:
    vv=ipu[col].fillna(0.0).values
    du=mkpanel(LU.values,iu,ctu,seu,decu,{'z':(vv-vv.mean())/vv.std()})
    du=du.dropna(subset=['dec'])
    for fname,fe in [("+ sector x year","u + t^k"),("+ country x year","u + t^i")]:
        m=pf.feols("ly ~ z:post | "+fe,data=du,vcov={'CRV1':'i'})
        b,s=m.coef()['z:post'],m.se()['z:post']
        print(f"    {lab:10s} {fname:18s} {b:>+8.4f} ({s:.4f}) t={b/s:>+5.2f}   "
              f"({du['u'].nunique():,} units, {len(du):,} obs)")
pd.DataFrame(ROWS).to_csv("sc20_main.csv",index=False)
log("DONE")
