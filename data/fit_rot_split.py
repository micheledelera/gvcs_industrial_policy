"""Rotunno frame, share_n_policies, split developing / advanced.

  X^US_ikt = exp{ b1 (IP x Decoup x Dev) + b2 (IP x Decoup x Adv)
                  + g1 (IP x Dev) + g2 (IP x Adv) + [...]
                  + a_ik + mu_kt + [d_ict] }

Dev excludes China, mirroring the gravity design.

Two variants, because the gravity ladder showed the omitted lower-order terms matter:

  minimal    exactly the split as asked: the two triples and the two IP main effects.
  saturated  adds IP x China, and Decoup x Adv / Decoup x China. Decoup alone is
             absorbed by mu_kt, but Decoup x group is NOT (it varies at (i,k,t) and
             no fixed effect spans it), so the group interactions are required
             lower-order terms. Developing is the reference group, matching the
             gravity convention. In the gravity ladder, omitting these inflated both
             triples by a common amount -- the tell was the advanced coefficient
             turning spuriously significant.

China gets no triple, as in gravity: it is the source of the shock, not a recipient.

PPML on levels, two-way clustering on country and sector. No country-product trends
(pyfixest varying-slope syntax is broken -- see fit_rot.py), so this is Rotunno's
column (3), not their column (4).
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, os, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, OUT = 156, "rot_split_results.csv"
d = pd.read_pickle("rot_panel.pkl")
d['fe_ik']  = d.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32')
d['fe_kt']  = d.groupby(['ISIC4c','t'], observed=True).ngroup().astype('int32')
d['fe_ict'] = d.groupby(['i','isic2','t'], observed=True).ngroup().astype('int32')
d['cl_i']   = d['i'].astype('int32')
d['cl_k']   = d.groupby('ISIC4c', observed=True).ngroup().astype('int32')

chn = (d['i']==CHINA).astype('float64').values
adv = d['Advanced_i'].astype('float64').values
dev = (1-d['Advanced_i']).astype('float64').values * (1-chn)
T   = d['IP_share_n_policies'].astype('float64').values
dec = d['Decoup'].astype('float64').values
d['TD_dev']=T*dec*dev; d['TD_adv']=T*dec*adv
d['T_dev'] =T*dev;     d['T_adv'] =T*adv;   d['T_chn']=T*chn
d['Dec_adv']=dec*adv;  d['Dec_chn']=dec*chn
log(f"{d.shape} | rows: dev {dev.mean()*100:.1f}%  adv {adv.mean()*100:.1f}%  "
    f"chn {chn.mean()*100:.1f}%")

RHS = {
 "minimal":   "TD_dev + TD_adv + T_dev + T_adv",
 "saturated": "TD_dev + TD_adv + T_dev + T_adv + T_chn + Dec_adv + Dec_chn",
}
COLS = [("B. between (no delta_ict)", "fe_ik + fe_kt"),
        ("W. within  (+ delta_ict)",  "fe_ik + fe_kt + fe_ict")]
for vname, rhs in RHS.items():
    for cname, fes in COLS:
        for tol in [1e-6, 1e-5]:
            try:
                fit = pf.fepois(f"x_us ~ {rhs} | {fes}", data=d,
                                vcov={"CRV1":"cl_i + cl_k"},
                                demeaner=pf.LsmrDemeaner(fixef_maxiter=8000,
                                        fixef_atol=tol, fixef_btol=tol),
                                lean=True, store_data=False, copy_data=False)
                t = fit.tidy()
                print(f"\n### {vname} | {cname} | N={fit._N:,} | tol {tol:g}")
                print(t.round(5).to_string())
                pd.DataFrame([{'variant':vname,'col':cname,'term':k,'N':fit._N,
                               'coef':t.loc[k,'Estimate'],'se':t.loc[k,'Std. Error'],
                               'p':t.loc[k,'Pr(>|t|)']} for k in t.index]).to_csv(
                    OUT, mode='a', index=False, header=not os.path.exists(OUT))
                log(f"  {vname}/{cname}: dev = {t.loc['TD_dev','Estimate']:+.4f} "
                    f"(se={t.loc['TD_dev','Std. Error']:.4f}, p={t.loc['TD_dev','Pr(>|t|)']:.4f})"
                    f" | adv = {t.loc['TD_adv','Estimate']:+.4f} (p={t.loc['TD_adv','Pr(>|t|)']:.4f})")
                del fit; break
            except Exception as e:
                log(f"  {vname}/{cname} tol {tol:g} FAILED: {type(e).__name__}: {e}"); gc.collect()
        gc.collect()
log("ALL DONE")
