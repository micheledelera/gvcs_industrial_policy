"""All policy measures x with/without IPxUS, on the FULL sample.

Purpose: establish whether it is policy TARGETING (the share_* family: what fraction
of a country's policy effort goes to this sector) or policy VOLUME (the n_/frac_
family: how much policy the sector gets) that drives the result. The two families
correlate only 0.20-0.41 with each other while correlating 0.54-0.64 within family,
so they are genuinely distinct constructs rather than two labels for one thing.

Full sample (229 destinations), lag 3, cluster (i,s), FE alpha_ist + alpha_jst +
alpha_ij -- the headline specification, with only the policy variable changing.
Each measure is standardised by its own SD so coefficients are per-SD comparable.

Two specifications per measure:
  with     DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + IPxUS_chn + Dec_US_chn + Adv_Dec_US
  without  DDD_dev + DDD_adv +                         IPxUS_chn + Dec_US_chn + Adv_Dec_US
The China terms stay in both, as in RESULTS section 3c.

asinh variants of n_sub and frac_sub are included because those are monetary amounts
with SD/mean above 40 and maxima in the millions; a per-SD coefficient on the raw
variable is set by a handful of enormous observations, so a null there would be
uninterpretable -- we could not tell whether subsidies do not matter or the variable
is unusable.

Demeaning tolerance 1e-6: RESULTS section 3i verified this reproduces the 1e-8
headline to four decimals while running about 2.7x faster, which is what makes 20
full-sample fits feasible.

Results append to measures_results.csv after every fit and completed (measure, spec)
pairs are skipped on restart, so a container restart costs at most one fit.
"""
import pandas as pd
import numpy as np
import pyfixest as pf
import gc, os, time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, LAG = 156, 3
OUT = "measures_results.csv"
TOLS = [1e-6, 1e-5]

RAW_MEAS = ['n_policies','n_sub','frac_policies','frac_sub',
            'share_n_policies','share_n_sub','share_frac_policies','share_frac_sub']
ASINH = ['n_sub','frac_sub']          # monetary, heavy-tailed
MEASURES = RAW_MEAS + [f"asinh_{m}" for m in ASINH]

SPECS = {
    'with':    "DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + IPxUS_chn + Dec_US_chn + Adv_Dec_US",
    'without': "DDD_dev + DDD_adv + IPxUS_chn + Dec_US_chn + Adv_Dec_US",
}

done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    done = set(zip(prev['measure'], prev['spec']))
    log(f"resuming: {len(done)} (measure, spec) pairs already done")

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i_int'] = raw['i'].astype('int32'); raw['t_int'] = raw['t'].astype('int32')
for c in RAW_MEAS:
    raw[c] = pd.to_numeric(raw[c], errors='coerce').fillna(0).astype('float32')

lk = raw[['i_int','ISIC4c','t_int'] + RAW_MEAS].drop_duplicates(
    subset=['i_int','ISIC4c','t_int']).copy()
lk['t_int'] += LAG
lk = lk.rename(columns={c: f"L_{c}" for c in RAW_MEAS})

keep = ['i','j','i_int','ISIC4c','t','t_int','imports','Advanced_i','target','US_trade']
raw = raw[keep].merge(lk, on=['i_int','ISIC4c','t_int'], how='left')
del lk
gc.collect()
raw = raw[raw[f"L_{RAW_MEAS[0]}"].notna()].copy()
log(f"full sample after lag-{LAG} merge: {raw.shape}")

china = (raw['i_int'] == CHINA).astype('float32').values
adv   = raw['Advanced_i'].astype('float32').values
dev   = ((1 - raw['Advanced_i']).astype('float32').values) * (1 - china)
us    = raw['US_trade'].astype('float32').values
dec   = (raw['target'].astype('float32') * (raw['t_int'] >= 2018)).astype('float32').values

base = pd.DataFrame({
    'imports': raw['imports'].astype('float32').values,
    'fe_ist': raw.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
    'fe_jst': raw.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
    'fe_ij':  raw.groupby(['i','j'], observed=True).ngroup().astype('int32').values,
    'cl_is':  raw.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32').values,
})
meas_vals = {}
for c in RAW_MEAS:
    meas_vals[c] = raw[f"L_{c}"].astype('float32').values
for c in ASINH:
    meas_vals[f"asinh_{c}"] = np.arcsinh(raw[f"L_{c}"].astype('float64').values).astype('float32')
del raw
gc.collect()
log(f"base built: {base.shape}, {len(meas_vals)} measures held")

for meas in MEASURES:
    v = meas_vals[meas]
    sd = float(v.std())
    if not np.isfinite(sd) or sd == 0:
        log(f"{meas}: degenerate (sd={sd}), skipping")
        continue
    ip = (v / sd).astype('float32')

    m = base.copy()
    m['DDD_dev']    = dec * ip * us * dev
    m['DDD_adv']    = dec * ip * us * adv
    m['IPxUS_dev']  = ip * us * dev
    m['IPxUS_adv']  = ip * us * adv
    m['IPxUS_chn']  = ip * us * china
    m['Dec_US_chn'] = dec * us * china
    m['Adv_Dec_US'] = adv * dec * us
    treated = int((m['DDD_dev'] > 0).sum())
    log(f"=== {meas} === sd={sd:.6g}  treated obs={treated:,}")

    for spec, rhs in SPECS.items():
        if (meas, spec) in done:
            log(f"{meas}/{spec}: already done, skipping")
            continue
        fitted = False
        for tol in TOLS:
            try:
                fit = pf.fepois(f"imports ~ {rhs} | fe_ist + fe_jst + fe_ij",
                                data=m, vcov={"CRV1": "cl_is"},
                                demeaner=pf.LsmrDemeaner(fixef_maxiter=8000,
                                                         fixef_atol=tol, fixef_btol=tol),
                                lean=True, store_data=False, copy_data=False)
                t = fit.tidy()
                print(f"\n### {meas} | IPxUS {spec} | N={fit._N:,} | tol {tol:g}")
                print(t.round(5).to_string())
                rows = [{'measure': meas, 'spec': spec, 'term': term, 'N': fit._N,
                         'treated_obs': treated, 'sd': sd, 'tol': tol,
                         'coef': t.loc[term,'Estimate'], 'se': t.loc[term,'Std. Error'],
                         'p': t.loc[term,'Pr(>|t|)']} for term in t.index]
                pd.DataFrame(rows).to_csv(OUT, mode='a', index=False,
                                          header=not os.path.exists(OUT))
                log(f"{meas}/{spec}: DDD_dev = {t.loc['DDD_dev','Estimate']:+.4f} "
                    f"(se={t.loc['DDD_dev','Std. Error']:.4f}, "
                    f"p={t.loc['DDD_dev','Pr(>|t|)']:.4f})")
                del fit
                fitted = True
                break
            except Exception as e:
                log(f"{meas}/{spec} at tol {tol:g} FAILED: {type(e).__name__}: {e}")
                gc.collect()
        if not fitted:
            log(f"{meas}/{spec}: no tolerance converged")
    del m
    gc.collect()

if os.path.exists(OUT):
    r = pd.read_csv(OUT)
    d = r[r['term'] == 'DDD_dev']
    print("\n" + "="*80)
    print("DDD_dev BY MEASURE AND SPECIFICATION (full sample, lag 3)")
    print("="*80)
    piv = d.pivot_table(index='measure', columns='spec', values=['coef','p'])
    print(piv.round(4).to_string())
log("ALL DONE")
