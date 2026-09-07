"""Note baseline: standard gravity triplet, bare quadruple interaction.

  X_ijst = exp{ b1 Decoupling_st x IP_ist x TradeUS_j x Developing_i
              + b2 Decoupling_st x IP_ist x TradeUS_j x Advanced_i
              + a_ist + a_jst + a_ij }

n_policies, Decoupling_st = target_s x 1[t>=2018], lag 0, full sample,
cluster (i,s). Developing_i EXCLUDES China.

Two variants:
  A  exactly as specified -- China in the sample with no term of its own
  B  A + Dec_US_chn + IPxUS_chn

B exists because China's US-specific decline varies at (CHN, US, s, t), which
none of alpha_ist, alpha_jst, alpha_ij spans, so it sits in the residual -- and
it is correlated with Decoupling_st by construction, since decoupling is defined
as China losing US market share. China is roughly a fifth of US imports and PPML
estimates the fixed effects jointly with beta, so that residual can move the
alpha_jst benchmark against which developing exporters are measured. The gap
between A and B measures how much that matters at the baseline.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, os, sys, time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, MEASURE, LAG, OUT = 156, 'n_policies', 0, "note_base_results.csv"

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i_int'] = raw['i'].astype('int32'); raw['t_int'] = raw['t'].astype('int32')
raw[MEASURE] = pd.to_numeric(raw[MEASURE], errors='coerce').fillna(0).astype('float32')
lk = raw[['i_int','ISIC4c','t_int',MEASURE]].drop_duplicates(
    subset=['i_int','ISIC4c','t_int']).copy()
lk['t_int'] += LAG
lk = lk.rename(columns={MEASURE: 'IP_lag'})
raw = raw[['i','j','i_int','ISIC4c','t','t_int','imports','Advanced_i','target','US_trade']]
d = raw.merge(lk, on=['i_int','ISIC4c','t_int'], how='left')
d = d[d['IP_lag'].notna()].copy()
del raw, lk; gc.collect()

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
    'Dec_US_chn': dec * us * china,
    'IPxUS_chn':  ip * us * china,
    'fe_ist': d.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
    'fe_jst': d.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
    'fe_ij':  d.groupby(['i','j'],          observed=True).ngroup().astype('int32').values,
    'cl_is':  d.groupby(['i','ISIC4c'],     observed=True).ngroup().astype('int32').values,
})
del d; gc.collect()
log(f"lag {LAG}: {m.shape} | treated {int((m['DDD_dev']>0).sum()):,} | "
    f"fe_ist {m['fe_ist'].nunique():,} fe_jst {m['fe_jst'].nunique():,} "
    f"fe_ij {m['fe_ij'].nunique():,}")

# One variant per invocation: pass the key as argv[1]. Each step of the note is
# discussed before the next is run, so nothing is queued behind anything else.
ALL = {
    "A": ("A. as specified",     "DDD_dev + DDD_adv"),
    "B": ("B. + China controls", "DDD_dev + DDD_adv + Dec_US_chn + IPxUS_chn"),
}
VARIANTS = [ALL[sys.argv[1]]]

for name, rhs in VARIANTS:
    for tol in [1e-6, 1e-5]:
        try:
            fit = pf.fepois(f"imports ~ {rhs} | fe_ist + fe_jst + fe_ij",
                            data=m, vcov={"CRV1": "cl_is"},
                            demeaner=pf.LsmrDemeaner(fixef_maxiter=8000,
                                                     fixef_atol=tol, fixef_btol=tol),
                            lean=True, store_data=False, copy_data=False)
            t = fit.tidy()
            print(f"\n### {name} | lag {LAG} | N={fit._N:,} | tol {tol:g}")
            print(t.round(5).to_string())
            pd.DataFrame([{'variant': name, 'term': k, 'N': fit._N,
                           'coef': t.loc[k,'Estimate'], 'se': t.loc[k,'Std. Error'],
                           'p': t.loc[k,'Pr(>|t|)']} for k in t.index]).to_csv(
                OUT, mode='a', index=False, header=not os.path.exists(OUT))
            log(f"{name}: b1 = {t.loc['DDD_dev','Estimate']:+.4f} "
                f"(se={t.loc['DDD_dev','Std. Error']:.4f}, p={t.loc['DDD_dev','Pr(>|t|)']:.4f})"
                f"  |  b2 = {t.loc['DDD_adv','Estimate']:+.4f} "
                f"(p={t.loc['DDD_adv','Pr(>|t|)']:.4f})")
            del fit; break
        except Exception as e:
            log(f"{name} tol {tol:g} FAILED: {type(e).__name__}: {e}"); gc.collect()
    gc.collect()
log("ALL DONE")
