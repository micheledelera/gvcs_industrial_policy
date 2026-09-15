"""Formal pre-trend tests for the three rungs of fit_panel_es_ctrl.py.

Counting significant pre-coefficients is a weak diagnostic: it misses a smooth drift
whose individual years are each insignificant. Two tests per path:

  JOINT   H0: all pre-REF coefficients = 0        (Wald, CRV1 by country)
  SLOPE   H0: no linear trend through them        regress b_tau on tau, GLS with the
                                                  estimated vcov -- i.e. the t on the
                                                  slope of the pre-period path

A design can pass JOINT and fail SLOPE (drift with wide yearly SEs) or vice versa.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, sys, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("fit_panel_es_ctrl.py").read().split("tv=p['t'].values")[0]
     .replace('REF = int(sys.argv[1]) if len(sys.argv)>1 else 2020','REF = 2020'))
tv=p['t'].values
MEAS2=['share_n_policies','share_frac_policies','n_policies','frac_policies']

def tests(t, V, names, ref):
    """Wald on the listed coefficients, and GLS slope through them."""
    nm=[n for n in names if n in t.index]
    b=t.loc[nm,'Estimate'].values
    idx=[list(t.index).index(n) for n in nm]
    Vs=V[np.ix_(idx,idx)]
    Vi=np.linalg.pinv(Vs)
    W=float(b @ Vi @ b); k=len(nm)
    from scipy import stats
    pj=1-stats.chi2.cdf(W,k)
    x=np.array([int(n.split('_')[-1])-ref for n in nm],dtype=float)
    Xm=np.column_stack([np.ones_like(x),x])
    A=np.linalg.pinv(Xm.T@Vi@Xm); g=A@(Xm.T@Vi@b)
    se=np.sqrt(A[1,1]); tstat=g[1]/se
    return W,k,pj,g[1],se,tstat,2*(1-stats.norm.cdf(abs(tstat)))

rows=[]
for M_ in MEAS2:
    ipz=(p[M_]/p[M_].std()).values.astype('float32')
    ni,nd=[],[]
    for y in years:
        if y==REF: continue
        p[f"I_{y}"]=(ipz*(tv==y)).astype('float32'); ni.append(f"I_{y}")
        p[f"D_{y}"]=(ipz*p['Dz'].values*(tv==y)).astype('float32'); nd.append(f"D_{y}")
    nc=[]
    for c in CTRL:
        cv=p[c].values
        for y in years:
            if y==REF: continue
            p[f"{c}_{y}"]=(cv*(tv==y)).astype('float32'); nc.append(f"{c}_{y}")
    core=" + ".join(nd+ni)
    SPECS=[("1. a_ik + a_kt",        f"s ~ {core} | fe_ik + fe_kt"),
           ("2. + capability x year",f"s ~ {core} + {' + '.join(nc)} | fe_ik + fe_kt"),
           ("3. + a_it",             f"s ~ {core} | fe_ik + fe_kt + fe_it")]
    print(f"\n{'='*94}\nMEASURE {M_}   ref {REF}   pre-period = 2007-2019 (13 years)\n{'='*94}")
    print(f"{'spec':24s}{'path':9s}{'Wald chi2(13)':>15s}{'p':>9s}"
          f"{'slope/yr':>11s}{'se':>9s}{'t':>7s}{'p':>9s}")
    for name,fml in SPECS:
        fit=pf.feols(fml,data=p,weights="w",vcov={"CRV1":"cl_i"},
                     lean=False,store_data=False,copy_data=False)
        t=fit.tidy(); V=fit._vcov
        for lbl,nms in [("IPxDec",nd),("IP",ni)]:
            pre=[n for n in nms if int(n.split('_')[-1])<REF]
            W,k,pj,g,se,ts,ps=tests(t,V,pre,REF)
            print(f"{name:24s}{lbl:9s}{W:>15.2f}{pj:>9.4f}{g:>11.4f}{se:>9.4f}{ts:>7.2f}{ps:>9.4f}")
            rows.append({'measure':M_,'spec':name,'path':lbl,'wald':W,'df':k,'p_joint':pj,
                         'slope':g,'slope_se':se,'slope_t':ts,'p_slope':ps})
        del fit; gc.collect()
    for c in ni+nd+nc: del p[c]
    gc.collect()
pd.DataFrame(rows).to_csv("panel_es_wald_2020.csv",index=False)
log("DONE")
