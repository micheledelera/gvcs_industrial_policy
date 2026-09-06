"""lag 0 with target, plus the continuous decoupling measure at both lags.

Completes the picture from RESULTS section 3j, which covered lag 3 with the binary
target x post treatment. Three measures carried forward -- n_policies (the volume
benchmark) and the two share measures (the targeting measures that worked).

  batch 1  lag 0, dec = target_s x 1[t>=2018]              6 fits
  batch 2  lag 0, dec = Decouple_intensity_st x post       6 fits
  batch 3  lag 3, dec = Decouple_intensity_st x post       6 fits
(lag 3 with target is already in section 3j.)

On the continuous measure: Decouple_intensity_st is a DEVIATION-FROM-2017 measure --
exactly zero in 2017 and growing in both directions, with pre-period values larger
than the early post-period (mean 0.181 in 2007 against 0.021 in 2018), and identically
zero for non-target sectors. Used raw it would treat 2007 as heavily decoupled and mix
pre-period divergence into the treatment, so it is multiplied by post here: zero before
2018, rising 0.02 to 0.28 after. Left in natural units (0-1), so the coefficient is the
effect of moving from no decoupling to complete decoupling, per SD of IP -- NOT
comparable in magnitude to the binary target coefficients, which are per-sector-dummy.

Full sample, cluster (i,s), FE alpha_ist + alpha_jst + alpha_ij, demeaning tol 1e-6.
Results append per fit; completed (lag, dec, measure, spec) rows are skipped on restart.
"""
import pandas as pd
import numpy as np
import pyfixest as pf
import gc, os, time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA = 156
OUT = "lag_dec_results.csv"
TOLS = [1e-6, 1e-5]
MEASURES = ['n_policies', 'share_n_policies', 'share_frac_policies']
SPECS = {
    'with':    "DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + IPxUS_chn + Dec_US_chn + Adv_Dec_US",
    'without': "DDD_dev + DDD_adv + IPxUS_chn + Dec_US_chn + Adv_Dec_US",
}
# (lag, dec_type) batches to run
BATCHES = [(0, 'target'), (0, 'continuous'), (3, 'continuous')]

done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    done = set(zip(prev['lag'], prev['dec'], prev['measure'], prev['spec']))
    log(f"resuming: {len(done)} cells already done")

raw0 = pd.read_pickle("agg_for_estimation.pkl")
raw0['i_int'] = raw0['i'].astype('int32'); raw0['t_int'] = raw0['t'].astype('int32')
for c in MEASURES:
    raw0[c] = pd.to_numeric(raw0[c], errors='coerce').fillna(0).astype('float32')
raw0['Decouple_intensity_st'] = pd.to_numeric(
    raw0['Decouple_intensity_st'], errors='coerce').fillna(0).astype('float32')
lk0 = raw0[['i_int','ISIC4c','t_int'] + MEASURES].drop_duplicates(
    subset=['i_int','ISIC4c','t_int']).copy()
keep = ['i','j','i_int','ISIC4c','t','t_int','imports','Advanced_i','target',
        'US_trade','Decouple_intensity_st']
raw0 = raw0[keep]
gc.collect()

for LAG in sorted({b[0] for b in BATCHES}):
    batches = [b for b in BATCHES if b[0] == LAG]
    if all((LAG, dt, m, s) in done for _, dt in batches for m in MEASURES for s in SPECS):
        log(f"lag {LAG}: all cells done, skipping frame build")
        continue

    lk = lk0.copy()
    lk['t_int'] += LAG
    lk = lk.rename(columns={c: f"L_{c}" for c in MEASURES})
    d = raw0.merge(lk, on=['i_int','ISIC4c','t_int'], how='left')
    del lk
    gc.collect()
    d = d[d[f"L_{MEASURES[0]}"].notna()].copy()
    log(f"=== lag {LAG} frame: {d.shape} ===")

    china = (d['i_int'] == CHINA).astype('float32').values
    adv   = d['Advanced_i'].astype('float32').values
    dev   = (1 - d['Advanced_i']).astype('float32').values * (1 - china)
    us    = d['US_trade'].astype('float32').values
    post  = (d['t_int'] >= 2018).astype('float32').values
    DEC = {
        'target':     (d['target'].astype('float32').values * post),
        'continuous': (d['Decouple_intensity_st'].astype('float32').values * post),
    }
    base = pd.DataFrame({
        'imports': d['imports'].astype('float32').values,
        'fe_ist': d.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
        'fe_jst': d.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
        'fe_ij':  d.groupby(['i','j'], observed=True).ngroup().astype('int32').values,
        'cl_is':  d.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32').values,
    })
    mvals = {c: d[f"L_{c}"].astype('float32').values for c in MEASURES}
    del d
    gc.collect()

    for _, dec_type in batches:
        dec = DEC[dec_type]
        log(f"--- lag {LAG} | dec={dec_type} | dec>0 on {100*(dec>0).mean():.1f}% of rows, "
            f"mean|dec>0 = {dec[dec>0].mean():.4f} ---")
        for meas in MEASURES:
            if all((LAG, dec_type, meas, s) in done for s in SPECS):
                log(f"{meas}: done, skipping"); continue
            sd = float(mvals[meas].std())
            ip = (mvals[meas] / sd).astype('float32')
            m = base.copy()
            m['DDD_dev']    = dec * ip * us * dev
            m['DDD_adv']    = dec * ip * us * adv
            m['IPxUS_dev']  = ip * us * dev
            m['IPxUS_adv']  = ip * us * adv
            m['IPxUS_chn']  = ip * us * china
            m['Dec_US_chn'] = dec * us * china
            m['Adv_Dec_US'] = adv * dec * us
            treated = int((m['DDD_dev'] > 0).sum())
            log(f"  {meas}: sd={sd:.6g}, treated obs={treated:,}")

            for spec, rhs in SPECS.items():
                if (LAG, dec_type, meas, spec) in done:
                    continue
                for tol in TOLS:
                    try:
                        fit = pf.fepois(f"imports ~ {rhs} | fe_ist + fe_jst + fe_ij",
                                        data=m, vcov={"CRV1": "cl_is"},
                                        demeaner=pf.LsmrDemeaner(fixef_maxiter=8000,
                                                                 fixef_atol=tol, fixef_btol=tol),
                                        lean=True, store_data=False, copy_data=False)
                        t = fit.tidy()
                        print(f"\n### lag {LAG} | dec={dec_type} | {meas} | IPxUS {spec} "
                              f"| N={fit._N:,} | tol {tol:g}")
                        print(t.round(5).to_string())
                        pd.DataFrame([{'lag': LAG, 'dec': dec_type, 'measure': meas,
                                       'spec': spec, 'term': term, 'N': fit._N,
                                       'treated_obs': treated, 'tol': tol,
                                       'coef': t.loc[term,'Estimate'],
                                       'se': t.loc[term,'Std. Error'],
                                       'p': t.loc[term,'Pr(>|t|)']} for term in t.index]
                                     ).to_csv(OUT, mode='a', index=False,
                                              header=not os.path.exists(OUT))
                        log(f"  lag{LAG}/{dec_type}/{meas}/{spec}: DDD_dev = "
                            f"{t.loc['DDD_dev','Estimate']:+.4f} "
                            f"(se={t.loc['DDD_dev','Std. Error']:.4f}, "
                            f"p={t.loc['DDD_dev','Pr(>|t|)']:.4f})")
                        del fit
                        break
                    except Exception as e:
                        log(f"  {meas}/{spec} tol {tol:g} FAILED: {type(e).__name__}: {e}")
                        gc.collect()
            del m
            gc.collect()
    del base, mvals, DEC
    gc.collect()

log("ALL DONE")
