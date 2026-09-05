import pandas as pd
import numpy as np
import pyfixest as pf
import time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA = 156
DEST = {842,276,251,380,826,528,724,56,392,124,410,36,752,40,208,246,372,620,300,579,757,554}

# Winning spec was: share_frac_policies + target x post + lag 3  -> DDD_dev = +0.0335
# Variant A was:    frac_policies       + Decouple_intensity + lag 0 -> DDD_dev = -0.0006
# Each config below changes exactly ONE thing from the winner.
CONFIGS = [
    ("a) MEASURE: level instead of share",  'frac_policies',       'targetpost', 3),
    ("b) TREATMENT: intensity instead of target x post", 'share_frac_policies', 'intensity', 3),
    ("c) LAG: contemporaneous instead of lag 3", 'share_frac_policies', 'targetpost', 0),
]

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i_int'] = raw['i'].astype('int32'); raw['t_int'] = raw['t'].astype('int32')
for m in ['frac_policies','share_frac_policies']:
    raw[m] = pd.to_numeric(raw[m], errors='coerce').fillna(0).astype('float32')
log(f"loaded {raw.shape}")

results = {}
for name, MEASURE, TREAT, LAG in CONFIGS:
    log(f"=== {name}  [{MEASURE} | {TREAT} | lag {LAG}] ===")
    look = (raw[['i_int','ISIC4c','t_int',MEASURE]]
            .drop_duplicates(subset=['i_int','ISIC4c','t_int']))
    look = look.assign(t_int=look['t_int'] + LAG).rename(columns={MEASURE: 'IP_lag'})

    d = raw[raw['j'].astype('int32').isin(DEST)].copy()      # China retained, own terms
    d = d.merge(look, on=['i_int','ISIC4c','t_int'], how='left')
    d = d[d['IP_lag'].notna()].copy()
    d['IP_z'] = (d['IP_lag'] / d['IP_lag'].std()).astype('float32')

    d['fe_ist'] = d.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32')
    d['fe_jst'] = d.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32')
    d['fe_ij']  = d.groupby(['i','j'], observed=True).ngroup().astype('int32')
    d['cl_is']  = d.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32')

    china = (d['i_int'] == CHINA).astype('float32')
    adv   = d['Advanced_i'].astype('float32')
    dev   = ((1 - d['Advanced_i']) * (1 - china)).astype('float32')
    us    = d['US_trade'].astype('float32')
    dec = ((d['target'] * (d['t_int'] >= 2018)).astype('float32') if TREAT == 'targetpost'
           else d['Decouple_intensity_st'].astype('float32'))

    d['DDD_dev']    = (dec * d['IP_z'] * us * dev).astype('float32')
    d['DDD_adv']    = (dec * d['IP_z'] * us * adv).astype('float32')
    d['IPxUS_dev']  = (d['IP_z'] * us * dev).astype('float32')
    d['IPxUS_adv']  = (d['IP_z'] * us * adv).astype('float32')
    d['IPxUS_chn']  = (d['IP_z'] * us * china).astype('float32')
    d['Dec_US_chn'] = (dec * us * china).astype('float32')
    d['Adv_Dec_US'] = (adv * dec * us).astype('float32')

    f = ("imports ~ DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + IPxUS_chn "
         "+ Dec_US_chn + Adv_Dec_US | fe_ist + fe_jst + fe_ij")
    try:
        fit = pf.fepois(f, data=d, vcov={"CRV1": "cl_is"},
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                        lean=True, store_data=False, copy_data=False)
        t = fit.tidy()
        print(f"\n### {name}\n    measure={MEASURE} treatment={TREAT} lag={LAG}  N={fit._N:,}")
        print(t.round(5).to_string())
        t.to_csv(f"gravity_decomp_{name[0]}.csv")
        results[name] = (t.loc['DDD_dev','Estimate'], t.loc['DDD_dev','Pr(>|t|)'])
        log(f"{name}: DDD_dev = {results[name][0]:+.4f} (p={results[name][1]:.4f})")
        del fit
    except Exception as e:
        log(f"{name} FAILED: {e}")
    del d

print("\n" + "="*78)
print("DECOMPOSITION -- each row changes ONE thing from the winning spec")
print("winning spec (share + target x post + lag3): DDD_dev = +0.0335 (p=0.016)")
print("="*78)
for k, (est, p) in results.items():
    print(f"  {k:52s} DDD_dev = {est:+.4f} (p={p:.4f})")
log("DONE")
