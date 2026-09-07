"""Note, step 1: the bare quadruple interaction with pair-sector fixed effects.

  X_ijst = exp{ b1 Decoupling_st x IP_ist x TradeUS_j x Developing_i
              + b2 Decoupling_st x IP_ist x TradeUS_j x Advanced_i
              + a_ist + a_ijs + a_ij }

target x post as Decoupling_st, n_policies as IP, full sample, cluster (i,s),
lag 0 then lag 3. Deliberately unsaturated -- lower-order terms are added one at
a time in later steps.

a_ij is omitted because it is SPANNED by a_ijs: every (i,j) group is a union of
(i,j,s) groups, so the two fixed effects together have the same column space as
a_ijs alone. Estimates, standard errors and residuals are numerically identical
to including both; dropping it only avoids 42,684 redundant parameters.

There is no a_jst at this step, so the US sector-year shock -- including the
Section 301 tariffs themselves -- is not absorbed: Decoupling_st x TradeUS_j
varies at (j,s,t) and sits in the residual. That is what step 2 adds.

Identification: the interaction varies across j within (i,s,t) and across t
within (i,j,s), so neither fixed effect spans it.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, os, time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, MEASURE, OUT = 156, 'n_policies', "note_step1_results.csv"

raw0 = pd.read_pickle("agg_for_estimation.pkl")
raw0['i_int'] = raw0['i'].astype('int32'); raw0['t_int'] = raw0['t'].astype('int32')
raw0[MEASURE] = pd.to_numeric(raw0[MEASURE], errors='coerce').fillna(0).astype('float32')
lk0 = raw0[['i_int','ISIC4c','t_int',MEASURE]].drop_duplicates(
    subset=['i_int','ISIC4c','t_int']).copy()
raw0 = raw0[['i','j','i_int','ISIC4c','t','t_int','imports','Advanced_i','target','US_trade']]
gc.collect()

for LAG in [0, 3]:
    lk = lk0.copy(); lk['t_int'] += LAG
    lk = lk.rename(columns={MEASURE: 'IP_lag'})
    d = raw0.merge(lk, on=['i_int','ISIC4c','t_int'], how='left')
    d = d[d['IP_lag'].notna()].copy()
    del lk; gc.collect()

    china = (d['i_int'] == CHINA).astype('float32').values
    adv   = d['Advanced_i'].astype('float32').values
    dev   = (1 - d['Advanced_i']).astype('float32').values * (1 - china)
    us    = d['US_trade'].astype('float32').values
    dec   = (d['target'].astype('float32') * (d['t_int'] >= 2018)).astype('float32').values
    ip    = (d['IP_lag'] / d['IP_lag'].std()).astype('float32').values

    m = pd.DataFrame({
        'imports': d['imports'].astype('float32').values,
        'DDD_dev': dec * ip * us * dev,
        'DDD_adv': dec * ip * us * adv,
        'fe_ist': d.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
        'fe_ijs': d.groupby(['i','j','ISIC4c'], observed=True).ngroup().astype('int32').values,
        'cl_is':  d.groupby(['i','ISIC4c'],     observed=True).ngroup().astype('int32').values,
    })
    del d; gc.collect()
    log(f"lag {LAG}: {m.shape} | fe_ist {m['fe_ist'].nunique():,} | "
        f"fe_ijs {m['fe_ijs'].nunique():,} | treated {int((m['DDD_dev']>0).sum()):,}")

    for tol in [1e-6, 1e-5]:
        try:
            fit = pf.fepois("imports ~ DDD_dev + DDD_adv | fe_ist + fe_ijs",
                            data=m, vcov={"CRV1": "cl_is"},
                            demeaner=pf.LsmrDemeaner(fixef_maxiter=8000,
                                                     fixef_atol=tol, fixef_btol=tol),
                            lean=True, store_data=False, copy_data=False)
            t = fit.tidy()
            print(f"\n### STEP 1 | lag {LAG} | N={fit._N:,} | tol {tol:g}")
            print(t.round(5).to_string())
            pd.DataFrame([{'lag': LAG, 'term': k, 'N': fit._N, 'tol': tol,
                           'coef': t.loc[k,'Estimate'], 'se': t.loc[k,'Std. Error'],
                           'p': t.loc[k,'Pr(>|t|)']} for k in t.index]).to_csv(
                OUT, mode='a', index=False, header=not os.path.exists(OUT))
            log(f"lag {LAG}: b1 = {t.loc['DDD_dev','Estimate']:+.4f} "
                f"(se={t.loc['DDD_dev','Std. Error']:.4f}, p={t.loc['DDD_dev','Pr(>|t|)']:.4f})"
                f"  |  b2 = {t.loc['DDD_adv','Estimate']:+.4f} "
                f"(p={t.loc['DDD_adv','Pr(>|t|)']:.4f})")
            del fit; break
        except Exception as e:
            log(f"lag {LAG} tol {tol:g} FAILED: {type(e).__name__}: {e}"); gc.collect()
    del m; gc.collect()
log("ALL DONE")
