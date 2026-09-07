"""Rotunno-frame estimation on US-bound exports, with and without delta_ict.

  X^US_ikt = exp{ b1 (IP x Decoup) + b2 IP + a_ik + a_ik x t + [d_ict] + mu_kt }

The two columns answer different questions and the pair is the point:

  WITH d_ict     country x ISIC2 x year absorbs all between-country variation within
                 a division-year, so the comparison is across ISIC4 sectors WITHIN a
                 country. This is the gravity design with a coarser absorber. It also
                 soaks up country-sector reporting bias in GTA, which is why Rotunno
                 include it.

  WITHOUT d_ict  between-country comparison: do policy-active country-sectors gain
                 more than others in the same sector-year, given each pair's own
                 fixed level and trend? THIS is the question -- among country-sectors
                 of similar capability (absorbed by a_ik) and similar exposure
                 (the Decoup interaction), did policy users gain more? The cost is
                 that country-year shocks are no longer controlled.

If b1 survives dropping d_ict the between-country claim has support; if it exists
only with d_ict, the finding is within-country reallocation and should be described
as such.

PPML on levels (48% of US cells are zero). Two-way clustering on country and sector.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, os, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

OUT="rot_results.csv"
d = pd.read_pickle("rot_panel.pkl")
d['fe_ik']  = d.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32')
d['fe_kt']  = d.groupby(['ISIC4c','t'], observed=True).ngroup().astype('int32')
d['fe_ict'] = d.groupby(['i','isic2','t'], observed=True).ngroup().astype('int32')
d['cl_i']   = d['i'].astype('int32')
d['cl_k']   = d.groupby('ISIC4c', observed=True).ngroup().astype('int32')
d['tt']     = (d['t'] - 2007).astype('float64')
log(f"{d.shape} | fe_ik {d['fe_ik'].nunique():,}  fe_kt {d['fe_kt'].nunique():,}  "
    f"fe_ict {d['fe_ict'].nunique():,}")

SPECS = [
    ("B. between  (no delta_ict)", "fe_ik[tt] + fe_kt"),
    ("W. within   (+ delta_ict)",  "fe_ik[tt] + fe_kt + fe_ict"),
]
for name, fes in SPECS:
    for tol in [1e-6, 1e-5]:
        try:
            fit = pf.fepois(f"x_us ~ IPxDec + IP | {fes}", data=d,
                            vcov={"CRV1":"cl_i + cl_k"},
                            demeaner=pf.LsmrDemeaner(fixef_maxiter=8000,
                                    fixef_atol=tol, fixef_btol=tol),
                            lean=True, store_data=False, copy_data=False)
            t = fit.tidy()
            print(f"\n### {name} | outcome x_us | N={fit._N:,} | tol {tol:g}")
            print(t.round(5).to_string())
            pd.DataFrame([{'spec':name,'outcome':'x_us','term':k,'N':fit._N,
                           'coef':t.loc[k,'Estimate'],'se':t.loc[k,'Std. Error'],
                           'p':t.loc[k,'Pr(>|t|)']} for k in t.index]).to_csv(
                OUT, mode='a', index=False, header=not os.path.exists(OUT))
            log(f"{name}: b1(IPxDec) = {t.loc['IPxDec','Estimate']:+.4f} "
                f"(se={t.loc['IPxDec','Std. Error']:.4f}, p={t.loc['IPxDec','Pr(>|t|)']:.4f})"
                f"  |  b2(IP) = {t.loc['IP','Estimate']:+.4f} (p={t.loc['IP','Pr(>|t|)']:.4f})")
            del fit; break
        except Exception as e:
            log(f"{name} tol {tol:g} FAILED: {type(e).__name__}: {e}"); gc.collect()
    gc.collect()
log("ALL DONE")
