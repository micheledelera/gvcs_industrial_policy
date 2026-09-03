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

formula = (f"imports ~ DDD_{d}_dev + DDD_{d}_adv + {d}_x_US_dev + {d}_x_US_adv "
           f"| i^ISIC4c^t + j^ISIC4c^t + i^j")
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
