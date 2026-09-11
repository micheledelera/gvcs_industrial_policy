"""Rotunno frame, share treatment: is it targeting rather than incidence?

The dummy version returned a tight null on both the interaction and the IP main
effect. Section 3j established why that need not contradict the gravity result:
measures of policy INCIDENCE and VOLUME are null or weak in this data, while
measures of TARGETING -- the fraction of a country's policy effort going to a
sector -- carry the result. A dummy for "received any policy" is an incidence
measure.

This runs the same two columns with share_n_policies and share_frac_policies in
place of the dummy, each standardised by its own SD so coefficients are per-SD
comparable to the gravity estimates. share_n_policies is the measure that was
significant in all eight cells of the section 3k grid.

If the share measures turn these coefficients positive where the dummy leaves them
at zero, that is the cleanest available evidence that targeting rather than
incidence is what matters -- produced in the benchmark paper's own framework, on a
design that can estimate the policy main effect.

No country-product trends: pyfixest's varying-slope syntax is unusable (see
fit_rot.py). Two-way clustering on country and sector.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, os, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

OUT="rot_share_results.csv"
d = pd.read_pickle("rot_panel.pkl")
d['fe_ik']  = d.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32')
d['fe_kt']  = d.groupby(['ISIC4c','t'], observed=True).ngroup().astype('int32')
d['fe_ict'] = d.groupby(['i','isic2','t'], observed=True).ngroup().astype('int32')
d['cl_i']   = d['i'].astype('int32')
d['cl_k']   = d.groupby('ISIC4c', observed=True).ngroup().astype('int32')

MEAS = ['share_n_policies','share_frac_policies']
SPECS = [("B. between (no delta_ict)", "fe_ik + fe_kt"),
         ("W. within  (+ delta_ict)",  "fe_ik + fe_kt + fe_ict")]
for meas in MEAS:
    d['T']  = d['IP_'+meas]
    d['TD'] = d['IPxDec_'+meas]
    log(f"=== {meas} === sd used {d[meas].std():.6g}; "
        f"nonzero in {100*(d[meas]>0).mean():.1f}% of cell-years")
    for name, fes in SPECS:
        for tol in [1e-6, 1e-5]:
            try:
                fit = pf.fepois(f"x_us ~ TD + T | {fes}", data=d,
                                vcov={"CRV1":"cl_i + cl_k"},
                                demeaner=pf.LsmrDemeaner(fixef_maxiter=8000,
                                        fixef_atol=tol, fixef_btol=tol),
                                lean=True, store_data=False, copy_data=False)
                t = fit.tidy()
                print(f"\n### {meas} | {name} | N={fit._N:,} | tol {tol:g}")
                print(t.round(5).to_string())
                pd.DataFrame([{'measure':meas,'spec':name,'term':k,'N':fit._N,
                               'coef':t.loc[k,'Estimate'],'se':t.loc[k,'Std. Error'],
                               'p':t.loc[k,'Pr(>|t|)']} for k in t.index]).to_csv(
                    OUT, mode='a', index=False, header=not os.path.exists(OUT))
                log(f"  {name}: IPxDec = {t.loc['TD','Estimate']:+.4f} "
                    f"(se={t.loc['TD','Std. Error']:.4f}, p={t.loc['TD','Pr(>|t|)']:.4f})"
                    f"  |  IP = {t.loc['T','Estimate']:+.4f} (p={t.loc['T','Pr(>|t|)']:.4f})")
                del fit; break
            except Exception as e:
                log(f"  {name} tol {tol:g} FAILED: {type(e).__name__}: {e}"); gc.collect()
        gc.collect()
log("ALL DONE")
