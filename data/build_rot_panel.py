"""Country-sector panel in the Rotunno (IMF WP 2024/041) eq-(1) frame.

Used as a workhorse for the decoupling question, not as a face-value replication.
Its value over our gravity design is that it does NOT absorb country-sector-year,
so the IP main effect is estimable -- the objection raised against gravity from the
outset: a policy variable specific to a country-industry cannot be identified in the
presence of origin-industry-year fixed effects.

Panel: (i, k, t) for every (i, k) pair that exports at least once, all years
2007-2024, so the panel is balanced within pair and the country-product linear
trends are well defined. Zeros retained -- PPML on levels, since US-bound flows at
ISIC4 are 41% zeros and a log specification would drop them all.

  k = ISIC 4-digit (125)      c = ISIC 2-digit (24)
  IP_ikt     = 1[n_policies > 0]                (their dummy, deliberate: little
                                                 within-product variation in counts)
  Decoup_kt  = target_k x 1[t >= 2018]          absorbed by mu_kt, so the
                                                 interaction is what identifies
"""
import pandas as pd, numpy as np, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

USA, CHINA = 842, 156
raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
MEAS=['n_policies','share_n_policies','share_frac_policies']
for c in MEAS:
    raw[c]=pd.to_numeric(raw[c],errors='coerce').fillna(0).astype('float32')
log(f"loaded {raw.shape}")

tot = raw.groupby(['i','ISIC4c','t'], observed=True)['imports'].sum().rename('x_tot')
us  = (raw[raw['j']==USA].groupby(['i','ISIC4c','t'], observed=True)['imports']
       .sum().rename('x_us'))
pol = (raw[['i','ISIC4c','t']+MEAS].drop_duplicates(subset=['i','ISIC4c','t'])
       .set_index(['i','ISIC4c','t']))
meta = raw[['i','Advanced_i']].drop_duplicates()
tgt  = raw[['ISIC4c','target']].drop_duplicates(subset=['ISIC4c'])
years = sorted(raw['t'].unique())
del raw; gc.collect()

# balanced panel over (i,k) pairs that ever export
pairs = tot[tot>0].reset_index()[['i','ISIC4c']].drop_duplicates()
log(f"(i,k) pairs ever exporting: {len(pairs):,}  x {len(years)} years")
p = pairs.loc[pairs.index.repeat(len(years))].copy()
p['t'] = np.tile(years, len(pairs))
p = p.merge(tot, on=['i','ISIC4c','t'], how='left').merge(us, on=['i','ISIC4c','t'], how='left')
p = p.merge(pol, on=['i','ISIC4c','t'], how='left')
p[['x_tot','x_us']+MEAS] = p[['x_tot','x_us']+MEAS].fillna(0.0)
p['x_nonus'] = (p['x_tot'] - p['x_us']).clip(lower=0)
p = p.merge(meta, on='i', how='left').merge(tgt, on='ISIC4c', how='left')
p['IP']      = (p['n_policies'] > 0).astype('float64')          # dummy, baseline
for c in ['share_n_policies','share_frac_policies']:              # share extension
    p['IP_'+c] = (p[c] / p[c].std()).astype('float64')
p['isic2']   = p['ISIC4c'].astype(str).str[:2]
p['Decoup']  = (p['target'].astype('float64') * (p['t']>=2018)).astype('float64')
p['IPxDec']  = p['IP'] * p['Decoup']
for c in ['share_n_policies','share_frac_policies']:
    p['IPxDec_'+c] = p['IP_'+c] * p['Decoup']
log(f"panel {p.shape} | zeros: total {100*(p['x_tot']==0).mean():.1f}%  "
    f"US {100*(p['x_us']==0).mean():.1f}%")
log(f"IP on in {100*p['IP'].mean():.1f}% of cell-years; "
    f"{100*p.groupby(['i','ISIC4c'],observed=True)['IP'].max().mean():.1f}% of pairs ever")
sw = p.groupby(['i','ISIC4c'], observed=True)['IP'].nunique()
log(f"pairs where IP SWITCHES over time: {int((sw>1).sum()):,} of {len(sw):,} "
    f"({100*(sw>1).mean():.1f}%) -- these identify the IP main effect")
p.to_pickle("rot_panel.pkl")
log(f"saved rot_panel.pkl")
