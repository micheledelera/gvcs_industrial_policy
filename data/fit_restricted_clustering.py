import pandas as pd
import pyfixest as pf
import sys
import time

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

ip_var = sys.argv[1] if len(sys.argv) > 1 else "frac_policies"
d = ip_var

# Major advanced final-demand markets. Excludes entrepot/special cases
# (HK, Singapore, Macao, Taiwan) and very small economies.
DEST = {842,276,251,380,826,528,724,56,392,124,410,36,752,40,208,246,372,620,300,579,757,554}

agg = pd.read_pickle("agg_for_estimation.pkl")
log(f"loaded agg: {agg.shape}")

agg = agg[agg['j'].astype('int32').isin(DEST)].copy()
log(f"restricted to {len(DEST)} advanced destinations: {agg.shape}")

agg['fe_ist'] = agg.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32')
agg['fe_jst'] = agg.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32')
agg['fe_ij']  = agg.groupby(['i','j'], observed=True).ngroup().astype('int32')
agg['cl_is']  = agg.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32')
agg['cl_i']   = agg['i'].astype('int32')
log(f"fe_ist {agg['fe_ist'].nunique():,} | fe_jst {agg['fe_jst'].nunique():,} | "
    f"fe_ij {agg['fe_ij'].nunique():,} | cl_is {agg['cl_is'].nunique():,} | cl_i {agg['cl_i'].nunique():,}")

formula = (f"imports ~ DDD2_{d}_dev + DDD2_{d}_adv + {d}_x_US_dev + {d}_x_US_adv "
           f"+ Adv_x_Decouple_x_US | fe_ist + fe_jst + fe_ij")
log(f"fitting: {formula}")

fit = pf.fepois(
    formula,
    data=agg,
    vcov={"CRV1": "fe_ist"},
    demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
)
log("fit done")

results = {}
for label, clvar in [("cluster: i,s,t (current)", "fe_ist"),
                     ("cluster: i,s", "cl_is"),
                     ("cluster: i (exporter)", "cl_i")]:
    try:
        fit.vcov({"CRV1": clvar})
        tid = fit.tidy()
        results[label] = tid[['Estimate','Std. Error','t value','Pr(>|t|)']].copy()
        log(f"{label}: vcov recomputed OK")
    except Exception as e:
        log(f"{label}: FAILED ({e})")

print("\n" + "="*100)
for label, tid in results.items():
    print(f"\n### {label}")
    print(tid.round(5).to_string())

if results:
    combined = pd.concat(results, axis=1)
    combined.to_csv(f"ppml_{ip_var}_restricted_clustering_comparison.csv")
    log("saved comparison CSV")
log("DONE")
