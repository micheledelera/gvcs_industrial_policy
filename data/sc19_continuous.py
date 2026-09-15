"""§5i. Continuous treatment instead of quartiles and a fixed window.

§5h found the HIGH-vs-LOW differential traces a U with its trough in 2015-17 -- the very
window the treatment is measured in. Two things could be doing that: the DISCRETISATION
(top vs bottom quartile of a 2015-17 average) or the SELECTION (policy lands on sectors
that were losing ground). This separates them.

  A  selection test        does pre-determined policy predict PRE-2018 export growth?
                           d ln X (2007 -> 2015-17) on standardised IP, sector FE
  B  continuous dose,      ln X_ukt = a_u + g_t + sum_s b_s (IP_uk x 1[t=s]) + e
     pre-determined        IP fixed at its 2015-17 mean and standardised, so the dose is
                           continuous but still pre-determined. If the pre-period still
                           slopes, the U is selection, not discretisation.
  C  continuous dose,      ln X_ukt = a_u + g_t + b IP_uk x Post_t + e    (the DiD analogue)
     pre-determined
  D  TIME-VARYING dose     ln X_ukt = a_u + g_t + b IP_ukt + e -- what "continuous treatment
                           instead of a window" most naturally means. Also the reverse
                           regression, IP_ukt on lagged export growth, because if policy
                           responds to performance this specification is picking up the
                           policy reaction function, not its effect.

Sample held to §5b's 5,166 balanced country-sectors so every number is comparable to
§5g/§5h. Both measures. SEs clustered on country throughout.
"""
import pandas as pd, numpy as np, pyfixest as pf, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])

# ---- year-by-year policy panel on the §5b units -----------------------------------------
raw2=pd.read_pickle("agg_for_estimation.pkl")
ipp=(raw2[['i','ISIC4c','t','n_policies','share_frac_policies']]
     .drop_duplicates(subset=['i','ISIC4c','t']))
for c in ['n_policies','share_frac_policies']: ipp[c]=pd.to_numeric(ipp[c],errors='coerce')
ipp['t']=ipp['t'].astype(int)
del raw2
IPP=(ipp.set_index(['i','ISIC4c','t'])[['n_policies','share_frac_policies']]
        .reindex(pd.MultiIndex.from_tuples([(i,k,t) for i,k in M.index for t in YRS],
                                           names=['i','ISIC4c','t'])).fillna(0.0))
Lfull=np.hstack([Lp,Lq])
d=pd.DataFrame({'u':np.repeat(np.arange(len(M)),len(YRS)),'t':np.tile(YRS,len(M)),
                'ly':Lfull.ravel()})
d['i']=CT[d['u'].values]; d['k']=SE[d['u'].values]
d['post']=(d['t']>=2018).astype(float)
d['vol_t']=IPP['n_policies'].values; d['tgt_t']=IPP['share_frac_policies'].values
# pre-determined dose: 2015-17 mean, standardised across units
for meas in ['vol','tgt']:
    z=(M[meas].values-M[meas].values.mean())/M[meas].values.std()
    d[f'{meas}_z']=z[d['u'].values]
# time-varying dose, standardised by the SAME constants so coefficients are comparable
for meas,col in [('vol','vol_t'),('tgt','tgt_t')]:
    d[f'{meas}_tz']=(d[col]-M[meas].values.mean())/M[meas].values.std()
log(f"panel {len(d):,} obs, {len(M):,} units")

OUT=[]
for meas,lab in [('tgt','TARGETING (share_frac_policies)'),('vol','VOLUME (n_policies)')]:
    Zc,Zt=f'{meas}_z',f'{meas}_tz'
    print(f"\n{'='*100}\n{lab}   dose = 1 sd of the 2015-17 distribution\n{'='*100}")

    # ---- A. selection: does policy land on sectors that had been losing ground? ----
    g=d[d['t'].isin([2007])].set_index('u')['ly']
    h=d[d['t'].isin(BASE)].groupby('u')['ly'].mean()
    s=pd.DataFrame({'dpre':h-g,'z':d.groupby('u')[Zc].first(),
                    'i':d.groupby('u')['i'].first(),'k':d.groupby('u')['k'].first()})
    for tag,fml in [("A1 no controls   ","dpre ~ z"),("A2 + sector FE   ","dpre ~ z | k"),
                    ("A3 + country FE  ","dpre ~ z | i + k")]:
        m=pf.feols(fml,data=s,vcov={'CRV1':'i'})
        b,e=m.coef()['z'],m.se()['z']
        print(f"  {tag} d lnX 2007->2015-17 on dose: {b:>+8.4f} ({e:.4f}) t={b/e:>+5.2f}")
    # same for the post period, for contrast
    q=d[d['t']>=2022].groupby('u')['ly'].mean()
    s['dpost']=q-h
    m=pf.feols("dpost ~ z | i + k",data=s,vcov={'CRV1':'i'})
    b,e=m.coef()['z'],m.se()['z']
    print(f"  A4 same, 2015-17 -> 2022-24               {b:>+8.4f} ({e:.4f}) t={b/e:>+5.2f}")

    # ---- B. continuous-dose event study, pre-determined ----
    for y in YRS:
        if y!=2017: d[f"z{y}"]=d[Zc]*(d['t']==y)
    ev=pf.feols("ly ~ "+" + ".join(f"z{y}" for y in YRS if y!=2017)+" | u + t^k",
                data=d,vcov={'CRV1':'i'})
    co,se=ev.coef(),ev.se()
    E=pd.DataFrame({'t':[y for y in YRS if y!=2017],
                    'b':[co[f"z{y}"] for y in YRS if y!=2017],
                    's':[se[f"z{y}"] for y in YRS if y!=2017]})
    E=pd.concat([E,pd.DataFrame({'t':[2017],'b':[0.0],'s':[0.0]})]).sort_values('t')
    E['meas']=meas; OUT.append(E)
    pre=E[(E['t']<2018)&(E['t']!=2017)]
    print(f"\n  B continuous-dose event study (unit + sector x year FE)")
    print("    "+"  ".join(f"{int(r.t)}:{r.b:+.3f}" for r in E.itertuples()))
    print(f"    pre-2018 coefs mean {pre['b'].mean():+.4f}, max |t| "
          f"{np.abs(pre['b']/pre['s']).max():.2f}")

    # ---- C. continuous dose x Post, and D. time-varying dose ----
    print(f"\n  C / D  dose-response")
    for tag,fml,var in [
        ("C1 dose x Post              ", f"ly ~ {Zc}:post | u + t",                f"{Zc}:post"),
        ("C2 + sector x year          ", f"ly ~ {Zc}:post | u + t^k",              f"{Zc}:post"),
        ("C3 + country x year         ", f"ly ~ {Zc}:post | u + t^i",              f"{Zc}:post"),
        ("D1 time-varying dose        ", f"ly ~ {Zt} | u + t",                     Zt),
        ("D2 + sector x year          ", f"ly ~ {Zt} | u + t^k",                  Zt),
        ("D3 + country x year         ", f"ly ~ {Zt} | u + t^i",                  Zt),
        ("D4 time-varying, pre-2018   ", f"ly ~ {Zt} | u + t^k",                  Zt)]:
        dd=d[d['t']<2018] if tag.startswith("D4") else d
        m=pf.feols(fml,data=dd,vcov={'CRV1':'i'})
        b,e=m.coef()[var],m.se()[var]
        print(f"    {tag} {b:>+8.4f} ({e:.4f}) t={b/e:>+5.2f}")

    # ---- reverse regression: does policy respond to past export growth? ----
    dd=d.sort_values(['u','t']).copy()
    dd['g1']=dd.groupby('u')['ly'].diff(); dd['g2']=dd.groupby('u')['g1'].shift(1)
    dd['g3']=dd.groupby('u')['g1'].shift(2)
    m=pf.feols(f"{Zt} ~ g1 + g2 + g3 | u + t",data=dd.dropna(subset=['g1','g2','g3']),
               vcov={'CRV1':'i'})
    co,se=m.coef(),m.se()
    print(f"\n  reverse regression: dose_t on its own lagged export growth")
    for v,nmv in [('g1','d lnX t-1'),('g2','d lnX t-2'),('g3','d lnX t-3')]:
        print(f"    {nmv:12s} {co[v]:>+8.4f} ({se[v]:.4f}) t={co[v]/se[v]:>+5.2f}")
    wt=m.wald_test(R=np.eye(3))
    print(f"    joint test of the three lags: p = {float(np.asarray(wt['pvalue']).ravel()[0]):.4f}")
pd.concat(OUT).to_csv("sc19_continuous.csv",index=False)
log("DONE")
