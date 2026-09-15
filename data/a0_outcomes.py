"""A0. Candidate outcomes for the merged GSC design, characterised before choosing.

Three candidates, all on the §5b sample (developing ex-China country-sectors):

  lnX     ln US imports from i in sector k                  what §5b used; not scale-free
  lnS     ln( X_ikt / sum_i X_ikt )  = lnX minus a SECTOR-YEAR constant
          i's share of US imports in sector k -- the literal "capturing reallocated US
          demand" outcome, scale-free, and it nets out sector-year shocks by construction
  lnD     ln( X^US_ikt / X^total_ikt )
          share of the country-sector's exports going to the US -- the DESTINATION margin,
          which is where the gravity headline lives, and which nets out country-sector scale

Reports coverage, dispersion, skew, and how far the treated units sit outside the donors'
range on each -- the crude hull check that §4c found decisive for the level-vs-scale-free
choice.
"""
import pandas as pd, numpy as np, gc
USA, CHINA = 842, 156
PRE=list(range(2007,2018)); POST=list(range(2018,2025)); YRS=PRE+POST; BASE=[2015,2016,2017]
raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
us=raw[raw['j']==USA]
XUS=us.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
XTOT=raw.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
ip=(raw[raw['t'].isin(BASE)][['i','ISIC4c','t','n_policies','share_frac_policies']]
    .drop_duplicates(subset=['i','ISIC4c','t'])
    .groupby(['i','ISIC4c'],observed=True)[['n_policies','share_frac_policies']].mean())
pol9=(raw[(raw['t']>=2009)&(raw['t']<=2017)][['i','ISIC4c','t','n_policies']]
      .drop_duplicates(subset=['i','ISIC4c','t']))
pol9['n_policies']=pd.to_numeric(pol9['n_policies'],errors='coerce').fillna(0)
yrs_on=(pol9.assign(on=(pol9['n_policies']>0).astype(int))
        .groupby(['i','ISIC4c'],observed=True)['on'].sum())
del raw, us; gc.collect()
for D in (XUS,XTOT):
    D.columns=[int(c) for c in D.columns]
XUS=XUS.reindex(columns=YRS); XTOT=XTOT.reindex(columns=YRS)
keep=[(i,k) for i,k in XUS.index if adv.get(i,1)==0 and i!=CHINA]
XUS=XUS.loc[keep]; XTOT=XTOT.reindex(XUS.index)
bal=(XUS[YRS]>0).all(axis=1) & XUS[YRS].notna().all(axis=1)
XUS=XUS[bal]; XTOT=XTOT.loc[XUS.index]
print(f"balanced developing ex-China country-sectors with positive US imports 2007-24: "
      f"{len(XUS):,}")

# sector-year totals across ALL exporters (not just developing) for the share denominator
tot_k = pd.read_pickle("agg_for_estimation.pkl")
tot_k = tot_k[tot_k['j']==USA]
SK = tot_k.groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
SK.columns=[int(c) for c in SK.columns]; SK=SK.reindex(columns=YRS)
del tot_k; gc.collect()

lnX=np.log(XUS[YRS])
den=np.array([SK.loc[k,YRS].values for _,k in XUS.index])
lnS=np.log(XUS[YRS].values/den)
lnD=np.log((XUS[YRS].values/XTOT[YRS].values).clip(1e-12,1.0))
TRT=(yrs_on.reindex(XUS.index).fillna(0).values>=6)
print(f"treated (targeted 6+ of 2009-17): {int(TRT.sum()):,}   "
      f"never targeted: {int((yrs_on.reindex(XUS.index).fillna(0).values==0).sum()):,}\n")
NEV=(yrs_on.reindex(XUS.index).fillna(0).values==0)

print(f"{'outcome':6s} {'finite%':>8s} {'mean':>8s} {'sd':>7s} {'skew':>7s} "
      f"{'p5':>8s} {'p95':>8s} | {'treated mean':>12s} {'donor mean':>11s} "
      f"{'% treated outside donor range':>30s}")
for nm_, M in [('lnX',lnX.values),('lnS',lnS),('lnD',lnD)]:
    v=M[:,[YRS.index(y) for y in BASE]].mean(axis=1)
    ok=np.isfinite(v)
    from scipy import stats as st
    lo,hi=np.nanmin(v[NEV&ok]),np.nanmax(v[NEV&ok])
    out=100*np.mean((v[TRT&ok]<lo)|(v[TRT&ok]>hi))
    print(f"{nm_:6s} {100*ok.mean():7.1f}% {np.nanmean(v[ok]):8.3f} {np.nanstd(v[ok]):7.3f} "
          f"{st.skew(v[ok]):7.2f} {np.nanpercentile(v[ok],5):8.3f} "
          f"{np.nanpercentile(v[ok],95):8.3f} | {np.nanmean(v[TRT&ok]):12.3f} "
          f"{np.nanmean(v[NEV&ok]):11.3f} {out:29.1f}%")
print("\n  (hull check is crude: share of treated units whose 2015-17 mean lies outside the"
      "\n   min-max range of never-targeted units, i.e. outside their 1-D convex hull)")
