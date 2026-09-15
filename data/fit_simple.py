import pandas as pd
import pyfixest as pf
import sys
import time

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

ip_var = sys.argv[1] if len(sys.argv) > 1 else "n_policies"

agg = pd.read_pickle("agg_for_estimation.pkl")
log(f"loaded cached agg: {agg.shape}")

# Simplest possible spec: pooled (no dev/adv split) DDD + two-way control.
agg['DDD_pooled'] = (agg['target'] * agg[ip_var] * agg['US_trade']).astype('float32')
agg['IPxUS_pooled'] = (agg[ip_var] * agg['US_trade']).astype('float32')

agg['fe_ist'] = agg['ist_cluster']
agg['fe_jst'] = agg.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32')
agg['fe_ij']  = agg.groupby(['i','j'], observed=True).ngroup().astype('int32')
log(f"built fe_ist ({agg['fe_ist'].nunique()}), fe_jst ({agg['fe_jst'].nunique()}), "
    f"fe_ij ({agg['fe_ij'].nunique()}) group codes")

formula = f"imports ~ DDD_pooled + IPxUS_pooled | fe_ist + fe_jst + fe_ij"
log(f"fitting pooled {ip_var} spec with LSMR demeaner ...")

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
fit.tidy().to_csv(f"ppml_{ip_var}_pooled_results.csv")
log("DONE")
