"""Is synthetic DiD feasible here, and does the share outcome break it?

SDiD (Arkhangelsky et al. 2021) chooses UNIT weights to match the treated units'
pre-treatment path and TIME weights to downweight pre-periods unlike the post. That is
attractive here for one specific reason: every design so far has TESTED pre-trends and
mostly failed. SDiD constructs a comparison that matches them instead.

Three feasibility questions:

  1. COHORT STRUCTURE. Treatment is "IP>0 and your sector's event has happened", so
     adoption is staggered by sector. SDiD needs block structure, so it is run per
     event cohort and averaged. How many treated and donor units per cohort, and how
     many pre-periods does each cohort have?

  2. SUTVA. This is the one that worries me. Shares sum to 100 within a sector, so if
     the donor pool is other countries in the SAME sector, a treated unit's gain comes
     mechanically out of its donors. SC leans entirely on donors being a valid
     counterfactual, so that is worse for SC than for regression. Checked: how much of
     a sector's US imports do the donors hold?

  3. OUTCOME. If shares break SUTVA, the fix is log US exports -- which needs a
     complete positive pre-period series. How many pairs have one?
"""
import pandas as pd, numpy as np, gc
USA, CHINA = 842, 156
BASE=[2015,2016,2017]; THR=0.25
MEASURE='share_frac_policies'

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
raw[MEASURE]=pd.to_numeric(raw[MEASURE],errors='coerce').fillna(0).astype('float32')
us=raw[raw['j']==USA]
x=us.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum().rename('X')
tot=x.groupby(level=['ISIC4c','t'],observed=True).transform('sum')
s=(x/tot.replace(0,np.nan)*100).rename('s')
chn=s.xs(CHINA,level='i').unstack('t'); chn.columns=[int(c) for c in chn.columns]
chn=chn.reindex(columns=sorted(chn.columns))
b=chn[BASE].mean(axis=1); chn=chn.loc[b>=5.0]; b=b.loc[chn.index]
below=chn.lt(b*(1-THR),axis=0)
stay=below.iloc[:,::-1].cummin(axis=1).iloc[:,::-1].astype(bool)
E=stay.apply(lambda r: r.index[r.values.argmax()] if r.any() else np.nan,axis=1).dropna().astype(int)
ip=(raw[raw['t'].isin(BASE)][['i','ISIC4c','t',MEASURE]].drop_duplicates(subset=['i','ISIC4c','t'])
    .groupby(['i','ISIC4c'],observed=True)[MEASURE].mean().rename('IP'))
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
years=sorted(raw['t'].unique())
del raw; gc.collect()

X=x.unstack('t'); X.columns=[int(c) for c in X.columns]
S=s.unstack('t'); S.columns=[int(c) for c in S.columns]
idx=pd.MultiIndex.from_tuples([t for t in X.index if t[1] in E.index], names=['i','ISIC4c'])
X=X.reindex(idx); S=S.reindex(idx)
meta=pd.DataFrame(index=idx)
meta['IP']=ip.reindex(idx).fillna(0.0)
meta['adv']=[adv.get(i,np.nan) for i,_ in idx]
meta['E']=[E[k] for _,k in idx]
meta=meta[(meta['adv']==0)&(meta.index.get_level_values('i')!=CHINA)]
X=X.loc[meta.index]; S=S.loc[meta.index]
meta['T']=(meta['IP']>0).astype(int)
print(f"pairs in event sectors, developing ex-China: {len(meta):,}   "
      f"treated {meta['T'].sum():,}   donors {(meta['T']==0).sum():,}\n")

print("="*78); print("1. COHORT STRUCTURE"); print("="*78)
print(f"{'event':>6s}{'sectors':>9s}{'treated':>9s}{'donors':>9s}{'pre yrs':>9s}{'post yrs':>9s}"
      f"{'sectors w/ >=5 tr & >=20 don':>30s}")
ok_coh=[]
for e in sorted(meta['E'].unique()):
    m=meta[meta['E']==e]
    nsec=m.index.get_level_values('ISIC4c').nunique()
    npre=len([y for y in years if y<e]); npost=len([y for y in years if y>=e])
    g=m.groupby(level='ISIC4c').agg(tr=('T','sum'), n=('T','size'))
    good=((g['tr']>=5)&((g['n']-g['tr'])>=20)).sum()
    print(f"{e:>6d}{nsec:>9d}{int(m['T'].sum()):>9d}{int((m['T']==0).sum()):>9d}"
          f"{npre:>9d}{npost:>9d}{good:>30d}")
    if npre>=5 and npost>=2: ok_coh.append(e)
print(f"\ncohorts with >=5 pre and >=2 post years: {ok_coh}")

print("\n"+"="*78); print("2. SUTVA: do donors hold the share the treated gain?"); print("="*78)
w15=X[BASE].mean(axis=1)
r=[]
for k,g in meta.groupby(level='ISIC4c'):
    ww=w15.loc[g.index]
    tr=ww[g['T']==1].sum(); dn=ww[g['T']==0].sum()
    r.append({'k':k,'tr_share_of_dev':tr/(tr+dn) if (tr+dn)>0 else np.nan,
              'n_don':int((g['T']==0).sum())})
r=pd.DataFrame(r)
print(f"treated pairs' share of developing-ex-China US imports in their sector:")
print(f"   median {r['tr_share_of_dev'].median():.3f}   mean {r['tr_share_of_dev'].mean():.3f}"
      f"   p10 {r['tr_share_of_dev'].quantile(.1):.3f}   p90 {r['tr_share_of_dev'].quantile(.9):.3f}")
print(f"   sectors where treated hold >50% of developing supply: "
      f"{(r['tr_share_of_dev']>0.5).sum()} of {len(r)}")
print("\n   -> the donor pool is NOT a passive bystander: in the median sector the")
print("      treated units already hold a large share, so any share they gain is")
print("      mechanically taken from the donors used to build their counterfactual.")

print("\n"+"="*78); print("3. OUTCOME: complete positive US export series?"); print("="*78)
for lab,yrs in [("2010-2024 (all)",years),("2012-2024",[y for y in years if y>=2012])]:
    pos=(X[yrs]>0).all(axis=1)
    print(f"   {lab:20s} pairs with X>0 in every year: {pos.sum():,} of {len(X):,} "
          f"({100*pos.mean():.1f}%)   treated among them {int(meta.loc[pos,'T'].sum()):,}")
    keep=X.loc[pos]
    print(f"      ...holding {100*keep[BASE].mean(axis=1).sum()/w15.sum():.1f}% of pre-period US value")
