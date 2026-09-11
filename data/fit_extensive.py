"""Extensive margin, on the rectangularised panel.

Two runs on the same frame (rect_panel.pkl, 7.91m rows, 54.7% zeros):

E2  LPM on 1[imports > 0] -- the extensive margin proper. Does industrial policy
    raise the PROBABILITY that a developing exporter serves the US in a
    decoupling-targeted sector after 2018? alpha_ist absorbs the exporter-sector-year
    participation rate, so identification is whether the US specifically gets served
    more in treated cells.

E1  PPML with zeros restored -- a correctness check on the headline, not a new
    margin. Every estimate so far is on positives only, which is a selected sample;
    handling zeros is the main reason to use PPML at all. This is the total effect,
    intensive and extensive combined, and is the number the paper should report as
    the headline if it differs from the positives-only estimate.

Same specification and clustering as the headline throughout, so the only change is
the sample.
"""
import pandas as pd
import numpy as np
import pyfixest as pf
import gc, time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA = 156

d = pd.read_pickle("rect_panel.pkl")
log(f"frame {d.shape} | zeros {100*(d['imports']==0).mean():.1f}%")

ip    = (d['IP_lag'] / d['IP_lag'].std()).astype('float32')
china = (d['i_int'] == CHINA).astype('float32')
adv   = d['Advanced_i'].astype('float32')
dev   = ((1 - d['Advanced_i']) * (1 - china)).astype('float32')
us    = d['US_trade'].astype('float32')
dec   = (d['target'] * (d['t_int'] >= 2018)).astype('float32')

m = pd.DataFrame({
    'imports':    d['imports'].values,
    'serves':     (d['imports'] > 0).astype('float32').values,
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
log(f"model matrix {m.shape} | mean(serves) = {m['serves'].mean():.4f} | "
    f"treated obs (DDD_dev>0) = {int((m['DDD_dev']>0).sum()):,}")

RHS = ("DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + IPxUS_chn "
       "+ Dec_US_chn + Adv_Dec_US | fe_ist + fe_jst + fe_ij")

# ---- E2. extensive margin, LPM ----
try:
    fit = pf.feols(f"serves ~ {RHS}", data=m, vcov={"CRV1": "cl_is"},
                   demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                   lean=True, store_data=False, copy_data=False)
    t = fit.tidy()
    print(f"\n### E2. EXTENSIVE MARGIN -- LPM on 1[imports>0]   N={fit._N:,}   cluster (i,s)")
    print(t.round(5).to_string())
    t.to_csv("extensive_lpm.csv")
    log(f"E2 LPM: DDD_dev = {t.loc['DDD_dev','Estimate']:+.5f} "
        f"(p={t.loc['DDD_dev','Pr(>|t|)']:.4f})  [pp change in P(serves US)]")
    del fit
except Exception as e:
    log(f"E2 FAILED: {type(e).__name__}: {e}")
gc.collect()

# ---- E1. PPML with zeros ----
try:
    fit = pf.fepois(f"imports ~ {RHS}", data=m, vcov={"CRV1": "cl_is"},
                    demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                    lean=True, store_data=False, copy_data=False)
    t = fit.tidy()
    print(f"\n### E1. PPML WITH ZEROS RESTORED   N={fit._N:,}   cluster (i,s)")
    print(t.round(5).to_string())
    t.to_csv("extensive_ppml_zeros.csv")
    log(f"E1 PPML+zeros: DDD_dev = {t.loc['DDD_dev','Estimate']:+.4f} "
        f"(p={t.loc['DDD_dev','Pr(>|t|)']:.4f})  [positives-only headline was +0.0335, p=0.016]")
    del fit
except Exception as e:
    log(f"E1 FAILED: {type(e).__name__}: {e}")

log("ALL DONE")
