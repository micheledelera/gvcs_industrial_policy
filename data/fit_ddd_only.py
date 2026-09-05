import pandas as pd
import pyfixest as pf
import sys
import time

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

ip_var = sys.argv[1] if len(sys.argv) > 1 else "n_policies"
d = ip_var

agg = pd.read_pickle("agg_for_estimation.pkl")
log(f"loaded cached agg: {agg.shape}")

agg['fe_ist'] = agg['ist_cluster']
agg['fe_jst'] = agg.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32')
agg['fe_ij']  = agg.groupby(['i','j'], observed=True).ngroup().astype('int32')
log(f"built fe_ist ({agg['fe_ist'].nunique()}), fe_jst ({agg['fe_jst'].nunique()}), "
    f"fe_ij ({agg['fe_ij'].nunique()}) group codes")

# Adv_x_target_x_US: zero by construction for developing exporters (Advanced_i=0),
# so it only nets out the "advanced economies structurally export more to the US
# in decoupling sectors" baseline from DDD_adv -- doesn't touch DDD_dev's identification.
agg['Adv_x_target_x_US'] = (agg['Advanced_i'] * agg['target'] * agg['US_trade']).astype('float32')

# DDD-only, dev/adv split, no IPxUS control (per author's instruction), plus the
# Advanced x target x US structural-baseline control (per author's instruction).
formula = f"imports ~ DDD_{d}_dev + DDD_{d}_adv + Adv_x_target_x_US | fe_ist + fe_jst + fe_ij"
log(f"fitting DDD-only {ip_var} (dev/adv split) with LSMR demeaner ...")

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
fit.tidy().to_csv(f"ppml_{ip_var}_ddd_only_devadv_results.csv")
log("DONE")
