import pandas as pd
import pyfixest as pf
import sys
import time

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

ip_var = sys.argv[1] if len(sys.argv) > 1 else "frac_policies"
d = ip_var
CUTOFF = 2018  # first Section 301 tariff tranche; Post = t >= CUTOFF

agg = pd.read_pickle("agg_for_estimation.pkl")
log(f"loaded cached agg: {agg.shape}")

agg['fe_ist'] = agg['ist_cluster']
agg['fe_jst'] = agg.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32')
agg['fe_ij']  = agg.groupby(['i','j'], observed=True).ngroup().astype('int32')
log(f"built fe_ist ({agg['fe_ist'].nunique()}), fe_jst ({agg['fe_jst'].nunique()}), "
    f"fe_ij ({agg['fe_ij'].nunique()}) group codes")

agg['Post'] = (agg['t'].astype('int32') >= CUTOFF).astype('int8')
log(f"Post={CUTOFF}+ share of obs: {agg['Post'].mean()*100:.1f}%")

agg['Adv_x_target_x_US'] = (agg['Advanced_i'] * agg['target'] * agg['US_trade']).astype('float32')

# Full pre/post split on all three terms.
for suffix in ['dev', 'adv']:
    base = f'DDD_{d}_{suffix}'
    agg[f'{base}_pre']  = (agg[base] * (1 - agg['Post'])).astype('float32')
    agg[f'{base}_post'] = (agg[base] * agg['Post']).astype('float32')
agg['Adv_x_target_x_US_pre']  = (agg['Adv_x_target_x_US'] * (1 - agg['Post'])).astype('float32')
agg['Adv_x_target_x_US_post'] = (agg['Adv_x_target_x_US'] * agg['Post']).astype('float32')

formula = (f"imports ~ DDD_{d}_dev_pre + DDD_{d}_dev_post + DDD_{d}_adv_pre + DDD_{d}_adv_post "
           f"+ Adv_x_target_x_US_pre + Adv_x_target_x_US_post | fe_ist + fe_jst + fe_ij")
log(f"fitting DDD pre/post ({CUTOFF} cutoff) {ip_var} (dev/adv split) with LSMR demeaner ...")

fit = pf.fepois(
    formula,
    data=agg,
    vcov={"CRV1": "ist_cluster"},
    demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
    lean=True,
    store_data=False,
    copy_data=False,
)
log("fit done")
print(fit.summary())
fit.tidy().to_csv(f"ppml_{ip_var}_ddd_prepost{CUTOFF}_devadv_results.csv")
log("DONE")
