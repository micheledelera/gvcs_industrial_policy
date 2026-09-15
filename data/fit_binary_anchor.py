"""Does the pooled DDD survive with a BINARY treatment?

The binary event study (fit_pretrend2.py variant C) is flat in the post period as
well as the pre period. Two readings:
  (i)  low power -- splitting the treatment across 12 year coefficients leaves each
       one estimated off roughly a seventh of the post-period mass, so a real pooled
       effect need not show up in any single year;
  (ii) a real null -- the headline depends on the continuous INTENSITY of policy, and
       a simple "any recorded policy" indicator carries no signal.

The pooled binary DDD separates them. If it is significant, (i). If it is null, (ii),
and the paper's claim is specifically about intensity rather than incidence.

Same sample and specification as the within-year anchor (2012+, lag 3, cluster (i,s)),
which gave DDD_dev = +0.0373 (p=0.005) with continuous within-year IP. Only the
treatment variable changes, so the comparison is clean.
"""
import pandas as pd
import numpy as np
import pyfixest as pf
import gc, time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, LAG, START = 156, 3, 2012
MEASURE = 'share_frac_policies'
DEST = {842,276,251,380,826,528,724,56,392,124,410,36,752,40,208,246,372,620,300,579,757,554}

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i_int'] = raw['i'].astype('int32'); raw['t_int'] = raw['t'].astype('int32')
raw[MEASURE] = pd.to_numeric(raw[MEASURE], errors='coerce').fillna(0).astype('float32')
lk = (raw[['i_int','ISIC4c','t_int',MEASURE]]
      .drop_duplicates(subset=['i_int','ISIC4c','t_int']))
lk['t_int'] = lk['t_int'] + LAG
lk = lk.rename(columns={MEASURE: 'IP_lag'})
raw = raw[raw['j'].astype('int32').isin(DEST)].copy()
raw = raw.merge(lk, on=['i_int','ISIC4c','t_int'], how='left')
d = raw[raw['IP_lag'].notna() & (raw['t_int'] >= START)].copy()
del raw, lk
gc.collect()

ip    = (d['IP_lag'] > 0).astype('float32')          # binary: any recorded policy
china = (d['i_int'] == CHINA).astype('float32')
adv   = d['Advanced_i'].astype('float32')
dev   = ((1 - d['Advanced_i']) * (1 - china)).astype('float32')
us    = d['US_trade'].astype('float32')
dec   = (d['target'] * (d['t_int'] >= 2018)).astype('float32')

m = pd.DataFrame({
    'imports':    d['imports'].astype('float32').values,
    'DDD_dev':    (dec * ip * us * dev).values,
    'DDD_adv':    (dec * ip * us * adv).values,
    'IPxUS_dev':  (ip * us * dev).values,
    'IPxUS_adv':  (ip * us * adv).values,
    'IPxUS_chn':  (ip * us * china).values,
    'Dec_US_chn': (dec * us * china).values,
    'Adv_Dec_US': (adv * dec * us).values,
    'fe_ist': d.groupby(['i','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
    'fe_jst': d.groupby(['j','ISIC4c','t'], observed=True).ngroup().astype('int32').values,
    'fe_ij':  d.groupby(['i','j'], observed=True).ngroup().astype('int32').values,
    'cl_is':  d.groupby(['i','ISIC4c'], observed=True).ngroup().astype('int32').values,
})
del d, ip, china, adv, dev, us, dec
gc.collect()
log(f"model matrix {m.shape} | treated obs (DDD_dev>0): {int((m['DDD_dev']>0).sum()):,}")

fit = pf.fepois("imports ~ DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + IPxUS_chn "
                "+ Dec_US_chn + Adv_Dec_US | fe_ist + fe_jst + fe_ij",
                data=m, vcov={"CRV1": "cl_is"},
                demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                lean=True, store_data=False, copy_data=False)
t = fit.tidy()
print(f"\n### POOLED DDD, BINARY TREATMENT 1[IP>0]   N={fit._N:,}   cluster (i,s)")
print(t.round(5).to_string())
t.to_csv("binary_anchor.csv")
log(f"binary pooled: DDD_dev = {t.loc['DDD_dev','Estimate']:+.4f} "
    f"(p={t.loc['DDD_dev','Pr(>|t|)']:.4f})   [continuous within-year was +0.0373, p=0.005]")
log("ALL DONE")
