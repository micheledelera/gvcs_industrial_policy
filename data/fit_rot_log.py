"""Log specification: Rotunno's actual functional form, and their actual outcome.

Two things, labelled separately:
  ln(x_us)   log of US-bound exports -- the functional-form comparison to the PPML
             results, same sample frame, same split, same fixed effects.
  ln(x_tot)  log of TOTAL exports -- Rotunno's own outcome. Their finding is that
             introducing subsidies raises exports about 2 percent. This is the
             validation: does our policy measure predict exports at all?

Saturated split only (the minimal variant was shown to be contaminated by the
omitted Decoup x group terms).

The cost of logs is selection: every zero is dropped. US-bound cells are 48% zero
and total-export cells 11.3%, so the ln(x_us) column is estimated on a sample
selected on the outcome -- exactly the problem section 3i identified, where
restoring zeros mattered. N is reported for each so the loss is visible rather than
buried.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, os, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, OUT = 156, "rot_log_results.csv"
d = pd.read_pickle("rot_panel.pkl")
d['fe_ik']  = d.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32')
d['fe_kt']  = d.groupby(['ISIC4c','t'], observed=True).ngroup().astype('int32')
d['fe_ict'] = d.groupby(['i','isic2','t'], observed=True).ngroup().astype('int32')
d['cl_i']   = d['i'].astype('int32')
d['cl_k']   = d.groupby('ISIC4c', observed=True).ngroup().astype('int32')

chn=(d['i']==CHINA).astype('float64').values
adv=d['Advanced_i'].astype('float64').values
dev=(1-d['Advanced_i']).astype('float64').values*(1-chn)
T  =d['IP_share_n_policies'].astype('float64').values
dec=d['Decoup'].astype('float64').values
d['TD_dev']=T*dec*dev; d['TD_adv']=T*dec*adv
d['T_dev']=T*dev; d['T_adv']=T*adv; d['T_chn']=T*chn
d['Dec_adv']=dec*adv; d['Dec_chn']=dec*chn
d['ln_us']  = np.where(d['x_us']>0,  np.log(d['x_us'].where(d['x_us']>0)),  np.nan)
d['ln_tot'] = np.where(d['x_tot']>0, np.log(d['x_tot'].where(d['x_tot']>0)), np.nan)
log(f"panel {len(d):,} | ln_us defined {d['ln_us'].notna().sum():,} "
    f"({100*d['ln_us'].notna().mean():.1f}%) | ln_tot defined "
    f"{d['ln_tot'].notna().sum():,} ({100*d['ln_tot'].notna().mean():.1f}%)")

RHS = "TD_dev + TD_adv + T_dev + T_adv + T_chn + Dec_adv + Dec_chn"
COLS=[("B. between (no delta_ict)","fe_ik + fe_kt"),
      ("W. within  (+ delta_ict)","fe_ik + fe_kt + fe_ict")]
for out in ['ln_us','ln_tot']:
    sub = d[d[out].notna()]
    for cname, fes in COLS:
        try:
            fit = pf.feols(f"{out} ~ {RHS} | {fes}", data=sub,
                           vcov={"CRV1":"cl_i + cl_k"}, lean=True,
                           store_data=False, copy_data=False)
            t = fit.tidy()
            print(f"\n### {out} | {cname} | N={fit._N:,}")
            print(t.round(5).to_string())
            pd.DataFrame([{'outcome':out,'col':cname,'term':k,'N':fit._N,
                           'coef':t.loc[k,'Estimate'],'se':t.loc[k,'Std. Error'],
                           'p':t.loc[k,'Pr(>|t|)']} for k in t.index]).to_csv(
                OUT, mode='a', index=False, header=not os.path.exists(OUT))
            log(f"  {out}/{cname}: dev={t.loc['TD_dev','Estimate']:+.4f} "
                f"(p={t.loc['TD_dev','Pr(>|t|)']:.4f}) | adv={t.loc['TD_adv','Estimate']:+.4f} "
                f"(p={t.loc['TD_adv','Pr(>|t|)']:.4f}) | IP_dev={t.loc['T_dev','Estimate']:+.4f} "
                f"(p={t.loc['T_dev','Pr(>|t|)']:.4f})")
            del fit
        except Exception as e:
            log(f"  {out}/{cname} FAILED: {type(e).__name__}: {e}")
        gc.collect()
log("ALL DONE")
