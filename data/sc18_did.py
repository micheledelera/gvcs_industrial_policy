"""§5h. Difference-in-differences on the same two groups the synthetic control uses.

The §5b/§5g figure contrasts a TREATED group (HIGH policy, top quartile of the positive
2015-17 distribution) with a PLACEBO group (LOW policy, bottom quartile or zero). The
synthetic control reweights the placebo group, one optimal donor mix per treated unit.
A DiD gives up the reweighting and compares the two groups' changes directly:

    ln X_ukt = alpha_u + gamma_t + beta (HIGH_u x Post_t) + e_ukt ,  Post = 1[t >= 2018]

beta is the DiD: (HIGH after - HIGH before) - (LOW after - LOW before). Everything else is
held to the §5b sample so the numbers are comparable: same 5,166 country-sectors, same
balanced 2007-2024 window, same two quartile definitions, both policy measures.

Three samples, because "the placebo group" can mean three things:
  A  ALL LOW units                       the full comparison group
  B  the 497 DRAWN placebos              literally the grey band in §5g
  C  LOW units that entered at least one treated unit's trimmed donor pool
                                         the SC's EFFECTIVE comparison set
                                         (different country, Chinese share within 10pp,
                                          log size within 2.0, 20 nearest on covariates)

Five specifications, each clustered on country:
  1  collapsed 2x2      pre = 2007-17 mean, post = 2018-24 mean, one obs per unit
  2  TWFE               unit FE + year FE
  3  + sector x year    absorbs every sector's own decoupling shock
  4  + country x year   the §5f demeaning as a DiD: identified within country, across
                        sectors, so a country-wide boom cannot be attributed to policy
  5  SC-weighted        LOW units weighted by the total synthetic-control weight they
                        actually received, so the DiD uses the SAME comparison mix as
                        §5b and the only remaining difference is the pre-period difference
"""
import pandas as pd, numpy as np, pyfixest as pf, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])

Lfull=np.hstack([Lp,Lq]); NP=len(PRE)

def pool_of(a, lopool):
    m=(np.abs(Dv[lopool]-Dv[a])<=DECB)&(np.abs(Sv[lopool]-Sv[a])<=SZB)&(CT[lopool]!=CT[a])
    cand=lopool[m]
    if len(cand)<8: return None,None
    pool=cand[np.argsort(np.sqrt(((Z[cand]-Z[a])**2).sum(axis=1)))[:K]]
    w=sc(np.hstack([Lp[pool],Z[pool]]), np.concatenate([Lp[a],Z[a]]))
    return pool,w

def panel(units, high, wt):
    """long panel of log US exports for the given units"""
    d=pd.DataFrame({'u':np.repeat(units,len(YRS)),'t':np.tile(YRS,len(units)),
                    'ly':Lfull[units].ravel()})
    d['i']=CT[d['u'].values]; d['k']=SE[d['u'].values]
    d['high']=high[d['u'].values].astype(float)
    d['post']=(d['t']>=2018).astype(float)
    d['dh']=d['high']*d['post']
    d['w']=wt[d['u'].values]
    return d

OUT=[]
for meas,lab in [('tgt','TARGETING (share_frac_policies)'),('vol','VOLUME (n_policies)')]:
    v=M[meas].values; pos=v[v>0]
    hi=np.where(v>=np.quantile(pos,.75))[0]; lo=np.where(v<=np.quantile(pos,.25))[0]
    drawn=rng.choice(lo,size=min(NPLAC,len(lo)),replace=False)     # same seed as §5b/§5g
    # SC weights actually received by each LOW unit, summed over treated units
    swt=np.zeros(len(M)); used=set()
    for a in hi:
        p,w=pool_of(a,lo)
        if p is None: continue
        swt[p]+=w; used.update(p[w>1e-6].tolist())
    eff=np.array(sorted(used))
    log(f"{lab}: HIGH {len(hi)}, LOW {len(lo)}, drawn placebos {len(drawn)}, "
        f"donors with positive weight {len(eff)}")
    HI=np.zeros(len(M),bool); HI[hi]=True

    for sname,ctrl in [("A  all LOW",lo),("B  drawn placebos",drawn),
                       ("C  weighted donors",eff)]:
        units=np.concatenate([hi,ctrl])
        wt=np.ones(len(M)); wt[ctrl]=swt[ctrl] if sname.startswith("C") else 1.0
        d=panel(units,HI,np.ones(len(M)))
        nh=int(HI[units].sum()); nc=len(ctrl)
        print(f"\n  {'-'*98}\n  {lab}   sample {sname}   HIGH {nh} / LOW {nc}   "
              f"{d['i'].nunique()} countries, {len(d):,} obs\n  {'-'*98}")
        # 1. collapsed 2x2, by hand as well as by regression
        c=(d.groupby(['u','i','high','post'],observed=True)['ly'].mean()
             .unstack('post').reset_index())
        c['dy']=c[1.0]-c[0.0]
        gh=c[c['high']==1]['dy'].mean(); gl=c[c['high']==0]['dy'].mean()
        m1=pf.feols("dy ~ high",data=c,vcov={'CRV1':'i'})
        b,s=m1.coef()['high'],m1.se()['high']
        print(f"    1 collapsed 2x2      {b:>+8.4f} ({s:.4f}) t={b/s:>+5.2f}   "
              f"HIGH {gh:+.3f} - LOW {gl:+.3f}")
        # 2-4. panel
        for tag,fe in [("2 TWFE              ","u + t"),
                       ("3 + sector x year   ","u + t^k"),
                       ("4 + country x year  ","u + t^i")]:
            m=pf.feols(f"ly ~ dh | {fe}",data=d,vcov={'CRV1':'i'})
            b,s=m.coef()['dh'],m.se()['dh']
            print(f"    {tag} {b:>+8.4f} ({s:.4f}) t={b/s:>+5.2f}")
        # 1b. collapsed 2x2 using only the SC's own base window as "before"
        cb=(d[d['t'].isin(BASE)|(d['t']>=2018)]
              .groupby(['u','i','high','post'],observed=True)['ly'].mean()
              .unstack('post').reset_index())
        cb['dy']=cb[1.0]-cb[0.0]
        mb=pf.feols("dy ~ high",data=cb,vcov={'CRV1':'i'})
        b,s=mb.coef()['high'],mb.se()['high']
        print(f"    1b 2x2, pre = 2015-17 {b:>+8.4f} ({s:.4f}) t={b/s:>+5.2f}")
        # 5. SC-weighted (only meaningful on the donors that carry weight)
        if sname.startswith("C"):
            dw=panel(units,HI,wt); dw=dw[dw['w']>0]
            m=pf.feols("ly ~ dh | u + t",data=dw,weights='w',vcov={'CRV1':'i'})
            b,s=m.coef()['dh'],m.se()['dh']
            print(f"    5 SC-weighted TWFE   {b:>+8.4f} ({s:.4f}) t={b/s:>+5.2f}")
            # 6-7. group-specific linear trend, unweighted and SC-weighted
            d['ht']=d['high']*(d['t']-2017); dw['ht']=dw['high']*(dw['t']-2017)
            for tag,dd,wcol in [("6 + HIGH linear trend",d,None),
                                ("7 SC-wgt + HIGH trend",dw,'w')]:
                m=pf.feols("ly ~ dh + ht | u + t",data=dd,
                           weights=wcol,vcov={'CRV1':'i'})
                b,s=m.coef()['dh'],m.se()['dh']
                bt,st=m.coef()['ht'],m.se()['ht']
                print(f"    {tag} {b:>+8.4f} ({s:.4f}) t={b/s:>+5.2f}   "
                      f"trend {bt:>+7.4f} ({st:.4f}) t={bt/st:>+5.2f}")
            # event study on the SC's effective comparison set, sector x year FE
            for y in YRS:
                if y!=2017: d[f"y{y}"]=d['high']*(d['t']==y)
            ev=pf.feols("ly ~ "+" + ".join(f"y{y}" for y in YRS if y!=2017)+" | u + t^k",
                        data=d,vcov={'CRV1':'i'})
            co,se=ev.coef(),ev.se()
            E=pd.DataFrame({'t':[y for y in YRS if y!=2017],
                            'b':[co[f"y{y}"] for y in YRS if y!=2017],
                            's':[se[f"y{y}"] for y in YRS if y!=2017]})
            E=pd.concat([E,pd.DataFrame({'t':[2017],'b':[0.0],'s':[0.0]})]).sort_values('t')
            E['meas']=meas; OUT.append(E)
            pre=E[(E['t']<2018)&(E['t']!=2017)]
            print(f"    event study (sector x year FE): pre-2018 coefs mean "
                  f"{pre['b'].mean():+.4f}, max |t| {np.abs(pre['b']/pre['s']).max():.2f}")
            print("      "+"  ".join(f"{int(r.t)}:{r.b:+.2f}" for r in E.itertuples()))
pd.concat(OUT).to_csv("sc18_did.csv",index=False)
log("DONE")
