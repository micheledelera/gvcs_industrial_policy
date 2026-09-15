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

# Pre-build interacted FE groups as compact int32 codes ourselves, rather than
# relying on pyfixest/formulaic's internal '^' interaction handling -- that
# appears to be what was actually blowing up memory (crash came immediately
# on formula parsing, before any demeaning iterations ran).
if 'fe_ist' not in agg.columns:
    agg['fe_ist'] = agg['ist_cluster']  # already i,ISIC4c,t group codes
    agg['fe_jst'] = agg.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32')
    agg['fe_ij']  = agg.groupby(['i','j'], observed=True).ngroup().astype('int32')
    log(f"built fe_ist ({agg['fe_ist'].nunique()}), fe_jst ({agg['fe_jst'].nunique()}), "
        f"fe_ij ({agg['fe_ij'].nunique()}) group codes")

formula = (f"imports ~ DDD_{d}_dev + DDD_{d}_adv + {d}_x_US_dev + {d}_x_US_adv "
           f"| fe_ist + fe_jst + fe_ij")
log(f"fitting {ip_var} with LSMR demeaner ...")

fit = pf.fepois(
    formula,
    data=agg,
    vcov={"CRV1": "ist_cluster"},
    demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
    lean=True,
    store_data=False,
    copy_data=False,
)
log(f"[{ip_var}] fit done")
print(fit.summary())
fit.tidy().to_csv(f"ppml_{ip_var}_results.csv")
log("DONE")
