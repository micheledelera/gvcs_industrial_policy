"""E1 retry: PPML with zeros, with a demeaning tolerance the problem can meet.

The first attempt failed -- "Demeaning failed after 2000 iterations" -- at the
LsmrDemeaner default tolerance of 1e-8. Adding zeros makes the weighted least
squares inside PPML's IRLS much worse conditioned: zero observations receive tiny
IRLS weights and many alpha_ist groups end up near-separated, so the inner solve
converges slowly. 2000 iterations took roughly 50 minutes, so simply raising the
cap at 1e-8 would mean an 8-hour fit.

1e-6 is still far tighter than the outer IRLS loop needs. Ladder of attempts,
stopping at the first success, so the loosest tolerance actually used is recorded
rather than assumed.

Everything else matches the headline specification, so E1 vs the positives-only
+0.0335 (p=0.016) isolates the effect of restoring zeros.
"""
import pandas as pd
import numpy as np
import pyfixest as pf
import gc, time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA = 156
ATTEMPTS = [(1e-6, 8000), (1e-5, 8000), (1e-4, 12000)]

d = pd.read_pickle("rect_panel.pkl")

# drop (i,s,t) cells with no positive flow to any of the 22 destinations: the FE
# predicts zero perfectly there, so they carry no information. fepois removes them
# anyway (821,478 last run); doing it here shrinks the problem the demeaner sees.
pos_cell = (d.assign(p=(d['imports'] > 0).astype('int8'))
            .groupby(['i_int','ISIC4c','t_int'], observed=True)['p'].transform('max'))
before = len(d)
d = d[pos_cell == 1].copy()
del pos_cell
gc.collect()
log(f"frame {before:,} -> {len(d):,} rows (dropped {before-len(d):,} in all-zero cells) "
    f"| zeros {100*(d['imports']==0).mean():.1f}%")

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
log(f"model matrix {m.shape} | treated obs {int((m['DDD_dev']>0).sum()):,} | "
    f"fe_ist levels {m['fe_ist'].nunique():,}")

F = ("imports ~ DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + IPxUS_chn "
     "+ Dec_US_chn + Adv_Dec_US | fe_ist + fe_jst + fe_ij")

for tol, mx in ATTEMPTS:
    log(f"attempt: atol=btol={tol:g}, maxiter={mx}")
    try:
        fit = pf.fepois(F, data=m, vcov={"CRV1": "cl_is"},
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=mx,
                                                 fixef_atol=tol, fixef_btol=tol),
                        lean=True, store_data=False, copy_data=False)
        t = fit.tidy()
        print(f"\n### E1. PPML WITH ZEROS RESTORED   N={fit._N:,}   cluster (i,s)   "
              f"demeaning tol {tol:g}")
        print(t.round(5).to_string())
        t.to_csv("extensive_ppml_zeros.csv")
        log(f"E1 SUCCEEDED at tol {tol:g}: DDD_dev = {t.loc['DDD_dev','Estimate']:+.4f} "
            f"(se={t.loc['DDD_dev','Std. Error']:.4f}, p={t.loc['DDD_dev','Pr(>|t|)']:.4f})"
            f"   [positives-only headline +0.0335, se 0.0138, p=0.016]")
        break
    except Exception as e:
        log(f"attempt at tol {tol:g} FAILED: {type(e).__name__}: {e}")
        gc.collect()
else:
    log("E1: no tolerance in the ladder converged")

log("ALL DONE")
