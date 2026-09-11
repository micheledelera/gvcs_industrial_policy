"""(1) Is a within-MVA-bracket comparison feasible? (2) What do electronics look like?

(1) The clean implementation replaces alpha_kt with alpha_{k,bracket,t}: comparisons run
only among countries in the same sector, same MVA bracket, same year. That is exactly
"within the same decile". The risk is empty support -- a cell with only policy-active or
only non-policy countries contributes nothing, so what matters is the share of WEIGHT
sitting in cells that contain both. Checked for deciles, quintiles and terciles.

(2) Electronics = ISIC Rev.4 divisions 26 (computer, electronic and optical) and
27 (electrical equipment).
"""
import pandas as pd, numpy as np, gc
USA, CHINA = 842, 156
BASE=[2015,2016,2017]; POST=[2022,2023,2024]; THR=0.25
MEASURE='share_frac_policies'
C2ISO={699:356,757:756,579:578,251:250,842:840,381:380,490:158,729:729}
XLSX=("/root/.claude/uploads/618110ef-d894-58be-8e00-e7f2bf548b03/"
      "4ad6510b-0154963EF4289D02DEA410D8D38E4E6A8C6D.xlsx")
NAME={484:'Mexico',699:'India',704:'Vietnam',458:'Malaysia',764:'Thailand',76:'Brazil',
      360:'Indonesia',608:'Philippines',50:'Bangladesh',116:'Cambodia',144:'SriLanka',
      340:'Honduras',222:'ElSalvador',400:'Jordan',792:'Turkiye',710:'SouthAfrica',
      818:'Egypt',504:'Morocco',686:'Senegal',158:'Taiwan',764:'Thailand',418:'Laos',
      104:'Myanmar',356:'India',586:'Pakistan',862:'Venezuela',170:'Colombia',
      604:'Peru',152:'Chile',32:'Argentina',188:'CostaRica',320:'Guatemala',
      214:'DomRep',388:'Jamaica',780:'TrinTob',12:'Algeria',788:'Tunisia'}

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
b=chn[BASE].mean(axis=1); pst=chn[POST].mean(axis=1)
ev=chn.loc[b>=5.0]; bb=b.loc[ev.index]
below=ev.lt(bb*(1-THR),axis=0)
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
u=pd.read_excel(XLSX,sheet_name='Data'); u=u[u['Year'].isin(BASE)]
U=u.pivot_table(index='CountryCode',columns='VariableCode',values='Value',aggfunc='mean')
ctry=sorted(d['i'].unique())
mva=pd.Series(U['NV_IND_MANF'].reindex([C2ISO.get(c,c) for c in ctry]).values,index=ctry)
d['mva']=d['i'].map(mva); d=d.dropna(subset=['mva']).copy()
dev=d.merge(E,left_on='ISIC4c',right_index=True,how='inner')

print("="*80); print("(1) SUPPORT FOR WITHIN-MVA-BRACKET COMPARISON"); print("="*80)
print(f"event-sector panel: {len(dev):,} pairs, {dev['i'].nunique()} countries, "
      f"{dev['ISIC4c'].nunique()} sectors\n")
cm=dev.groupby('i')['mva'].first()
for nb,lab in [(10,'deciles'),(5,'quintiles'),(3,'terciles')]:
    br=pd.qcut(cm,nb,labels=False,duplicates='drop')
    dev['br']=dev['i'].map(br)
    g=dev.groupby(['ISIC4c','br'],observed=True).agg(
        n=('i','size'), ntr=('IP',lambda v:(v>0).sum()), w=('w','sum'))
    both=(g['ntr']>0)&(g['ntr']<g['n'])
    print(f"--- {lab} ({nb} brackets): {len(g):,} sector-bracket cells")
    print(f"    countries per cell: median {g['n'].median():.0f}, "
          f"{100*(g['n']>=3).mean():.0f}% have >=3")
    print(f"    cells with BOTH treated and untreated: {both.sum():,} ({100*both.mean():.0f}%)")
    print(f"    ...holding {100*g.loc[both,'w'].sum()/g['w'].sum():.1f}% of the weight")
    print(f"    pairs in those cells: {dev.merge(g[both].reset_index()[['ISIC4c','br']],on=['ISIC4c','br']).shape[0]:,}\n")

print("="*80); print("(2) ELECTRONICS"); print("="*80)
d['div']=d['ISIC4c'].astype(str).str[:2]
elec=[c for c in d['ISIC4c'].unique() if str(c).startswith(('26','27'))]
E_all=pd.concat([b.rename('chn_pre'),pst.rename('chn_post')],axis=1)
E_all['E']=E; E_all=E_all.loc[[c for c in elec if c in E_all.index]]
E_all['chg']=E_all['chn_post']-E_all['chn_pre']
print(f"{'sector':>8s}{'China 15-17':>13s}{'China 22-24':>13s}{'change':>9s}{'event':>7s}")
for k,r in E_all.sort_values('chn_pre',ascending=False).iterrows():
    e='' if pd.isna(r['E']) else str(int(r['E']))
    print(f"{str(k):>8s}{r['chn_pre']:>13.1f}{r['chn_post']:>13.1f}{r['chg']:>+9.1f}{e:>7s}")

big=E_all['chn_pre'].idxmax()
print(f"\n--- top developing suppliers in {big} (largest Chinese presence) ---")
g=d[d['ISIC4c']==big].sort_values('w',ascending=False).head(12)
sp=s.unstack('t'); sp.columns=[int(c) for c in sp.columns]
print(f"{'country':<14s}{'US imp pre ($k)':>17s}{'share pre':>11s}{'share post':>11s}{'chg':>8s}{'IP':>9s}{'MVA':>7s}")
for _,r in g.iterrows():
    key=(r['i'],big)
    pre=sp.loc[key,BASE].mean() if key in sp.index else np.nan
    po =sp.loc[key,POST].mean() if key in sp.index else np.nan
    print(f"{str(NAME.get(r['i'],r['i'])):<14s}{r['w']:>17,.0f}{pre:>11.2f}{po:>11.2f}"
          f"{po-pre:>+8.2f}{r['IP']:>9.4f}{r['mva']:>7.1f}")
