"""Positives-only control for E1, holding the solver setting fixed.

E1 (PPML with zeros) ran at demeaning tolerance 1e-6 because 1e-8 would not
converge; the positives-only headline ran at 1e-8. So the E1-vs-headline gap
confounds two changes: restoring zeros, and the solver tolerance.

This isolates the zeros. Same rectangularised frame as E1, same tolerance, same
specification -- the only difference is that zero flows are dropped. Whatever gap
remains between this and E1 is attributable to the zeros alone.
"""
import pandas as pd
import numpy as np
import pyfixest as pf
import gc, time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, TOL = 156, 1e-6

d = pd.read_pickle("rect_panel.pkl")
pos_cell = (d.assign(p=(d['imports'] > 0).astype('int8'))
            .groupby(['i_int','ISIC4c','t_int'], observed=True)['p'].transform('max'))
d = d[pos_cell == 1]
del pos_cell
d = d[d['imports'] > 0].copy()      # <-- the only change from E1
gc.collect()
log(f"positives-only frame: {len(d):,} rows")

ip    = (d['IP_lag'] / d['IP_lag'].std()).astype('float32')
china = (d['i_int'] == CHINA).astype('float32')
adv   = d['Advanced_i'].astype('float32')
dev   = ((1 - d['Advanced_i']) * (1 - china)).astype('float32')
us    = d['US_trade'].astype('float32')
dec   = (d['target'] * (d['t_int'] >= 2018)).astype('float32')

m = pd.DataFrame({
    'imports':    d['imports'].values,
    'DDD_dev':    (dec * ip * us * dev).values,
    'DDD_adv':    (dec * ip * us * adv).values,
    'IPxUS_dev':  (ip * us * dev).values,
    'IPxUS_adv':  (ip * us * adv).values,
    'IPxUS_chn':  (ip * us * china).values,
    'Dec_US_chn': (dec * us * china).values,
    'Adv_Dec_US': (adv * dec * us).values,
    'fe_ist': d.groupby(['i_int','ISIC4c','t_int'], observed=True).ngroup().astype('int32').values,
    'fe_jst': d.groupby(['j_int','ISIC4c','t_int'], observed=True).ngroup().astype('int32').values,
    'fe_ij':  d.groupby(['i_int','j_int'], observed=True).ngroup().astype('int32').values,
    'cl_is':  d.groupby(['i_int','ISIC4c'], observed=True).ngroup().astype('int32').values,
})
del d, ip, china, adv, dev, us, dec
gc.collect()
log(f"model matrix {m.shape} | treated obs {int((m['DDD_dev']>0).sum()):,}")

fit = pf.fepois("imports ~ DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + IPxUS_chn "
                "+ Dec_US_chn + Adv_Dec_US | fe_ist + fe_jst + fe_ij",
                data=m, vcov={"CRV1": "cl_is"},
                demeaner=pf.LsmrDemeaner(fixef_maxiter=8000,
                                         fixef_atol=TOL, fixef_btol=TOL),
                lean=True, store_data=False, copy_data=False)
t = fit.tidy()
print(f"\n### E1-CONTROL. POSITIVES ONLY, tol {TOL:g}   N={fit._N:,}   cluster (i,s)")
print(t.round(5).to_string())
t.to_csv("extensive_ppml_positives_control.csv")
log(f"control: DDD_dev = {t.loc['DDD_dev','Estimate']:+.4f} "
    f"(se={t.loc['DDD_dev','Std. Error']:.4f}, p={t.loc['DDD_dev','Pr(>|t|)']:.4f})"
    f"   [E1 with zeros +0.0271, se 0.0108; headline at 1e-8 +0.0335, se 0.0138]")
log("ALL DONE")
