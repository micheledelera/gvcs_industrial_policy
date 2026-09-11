"""Who is actually being compared in the §3t event study, and with what weight.

alpha_ik removes the pair level, so identification is over time within a pair.
alpha_kt confines every comparison to a single sector-year cell. IP_ik is time-invariant,
so beta_tau traces how the CROSS-COUNTRY IP gradient inside a sector evolves around that
sector's own event. This is a between-country comparison executed within sector-year
cells -- there is no alpha_it, so a country's overall path is not absorbed; psi_tau
(MVA x event time) is the only thing standing in for it.
"""
import pandas as pd, numpy as np, gc
USA, CHINA = 842, 156
BASE=[2015,2016,2017]; THR=0.25
MEASURE='share_frac_policies'
C2ISO={699:356,757:756,579:578,251:250,842:840,381:380,490:158,729:729}
XLSX=("/root/.claude/uploads/618110ef-d894-58be-8e00-e7f2bf548b03/"
      "4ad6510b-0154963EF4289D02DEA410D8D38E4E6A8C6D.xlsx")

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
raw[MEASURE]=pd.to_numeric(raw[MEASURE],errors='coerce').fillna(0).astype('float32')
us=raw[raw['j']==USA]
x=us.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum()
tot=x.groupby(level=['ISIC4c','t'],observed=True).transform('sum')
s=(x/tot.replace(0,np.nan)*100).rename('s')
wgt=(us[us['t'].isin(BASE)].groupby(['i','ISIC4c'],observed=True)['imports'].sum()/3).rename('w')
chn=s.xs(CHINA,level='i').unstack('t'); chn.columns=[int(c) for c in chn.columns]
chn=chn.reindex(columns=sorted(chn.columns))
b=chn[BASE].mean(axis=1); chn=chn.loc[b>=5.0]; b=b.loc[chn.index]
below=chn.lt(b*(1-THR),axis=0)
stay=below.iloc[:,::-1].cummin(axis=1).iloc[:,::-1].astype(bool)
E=stay.apply(lambda r: r.index[r.values.argmax()] if r.any() else np.nan,axis=1).dropna().astype(int).rename('E')
ip=(raw[raw['t'].isin(BASE)][['i','ISIC4c','t',MEASURE]].drop_duplicates(subset=['i','ISIC4c','t'])
    .groupby(['i','ISIC4c'],observed=True)[MEASURE].mean().rename('IP'))
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
act=raw.loc[raw['imports']>0,['i','ISIC4c']].drop_duplicates()
del raw,us; gc.collect()

d=act.set_index(['i','ISIC4c']).join([wgt,ip]).reset_index()
d['w']=d['w'].fillna(0.0); d['IP']=d['IP'].fillna(0.0)
d['Advanced_i']=d['i'].map(adv)
d=d[(d['Advanced_i']==0)&(d['i']!=CHINA)&(d['w']>0)].copy()
d=d.merge(E,left_on='ISIC4c',right_index=True,how='inner')
u=pd.read_excel(XLSX,sheet_name='Data'); u=u[u['Year'].isin(BASE)]
U=u.pivot_table(index='CountryCode',columns='VariableCode',values='Value',aggfunc='mean')
ctry=sorted(d['i'].unique())
mva=pd.Series(U['NV_IND_MANF'].reindex([C2ISO.get(c,c) for c in ctry]).values,index=ctry)
d['mva']=d['i'].map(mva); d=d.dropna(subset=['mva']).copy()

print(f"estimating pairs: {len(d):,}   countries {d['i'].nunique()}   sectors {d['ISIC4c'].nunique()}")
print(f"pairs with IP>0: {(d['IP']>0).sum():,} ({100*(d['IP']>0).mean():.1f}%), "
      f"holding {100*d.loc[d['IP']>0,'w'].sum()/d['w'].sum():.1f}% of weight\n")

print("--- comparison-set size: countries per sector ---")
n=d.groupby('ISIC4c',observed=True)['i'].nunique()
npos=d[d['IP']>0].groupby('ISIC4c',observed=True)['i'].nunique().reindex(n.index).fillna(0)
print(f"countries per sector: median {n.median():.0f}  min {n.min()}  max {n.max()}")
print(f"...of which with IP>0: median {npos.median():.0f}  min {npos.min():.0f}  max {npos.max():.0f}")
print(f"sectors with 0 policy-active countries: {(npos==0).sum()} of {len(n)}\n")

print("--- who carries the weight ---")
wc=(d.groupby('i')['w'].sum()/d['w'].sum()).sort_values(ascending=False)
NAME={484:'Mexico',699:'India',704:'Vietnam',458:'Malaysia',764:'Thailand',76:'Brazil',
      360:'Indonesia',608:'Philippines',380:'Italy',410:'Korea',702:'Singapore',
      792:'Turkiye',124:'Canada',56:'Belgium',203:'Czechia',348:'Hungary'}
for c,v in wc.head(8).items():
    print(f"   {str(NAME.get(c,c)):<12s} {100*v:5.1f}%   IP>0 in {int((d[(d['i']==c)&(d['IP']>0)]).shape[0]):>3} "
          f"of {int((d['i']==c).sum()):>3} of its sectors")
print(f"   top 5 = {100*wc.head(5).sum():.1f}% of weight\n")

print("--- identifying variation in IP, weighted ---")
d['IPz']=d['IP']/d['IP'].std()
gm=d.groupby('ISIC4c',observed=True)['IPz'].transform('mean')
print(f"sd of IPz: total {d['IPz'].std():.3f}   within sector (what beta_tau uses) "
      f"{(d['IPz']-gm).std():.3f}")

print("\n--- a concrete cell: the largest event sector by US import value ---")
top=d.groupby('ISIC4c',observed=True)['w'].sum().idxmax()
g=d[d['ISIC4c']==top].sort_values('w',ascending=False).head(10)
print(f"sector {top}, event year {int(g['E'].iloc[0])}, {d[d['ISIC4c']==top]['i'].nunique()} "
      f"developing suppliers")
print(f"{'country':<14s}{'US imports (pre, $k)':>22s}{'IP':>10s}{'MVA/GDP':>10s}")
for _,r in g.iterrows():
    nmc = str(NAME.get(r['i'], r['i']))
    print(f"{nmc:<14s}{r['w']:>22,.0f}{r['IP']:>10.4f}{r['mva']:>10.1f}")
