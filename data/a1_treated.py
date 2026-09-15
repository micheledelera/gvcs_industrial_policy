"""A1.  NOTE: `imports` is in THOUSANDS of USD (calibrated: total US imports 2016 =
1,937,439,360 raw = $1.94tn; Mexico 264,173,984 = $264bn; Vietnam $41.8bn; China $463bn).
An earlier version of this script mislabelled the size column as $m when it was $bn.

A1. Defining the treated set, and the three questions that need settling first.

§7 audited seven treatment definitions; persistence (§8b) was the only one that survived —
no country-portfolio denominator (§5k), the best-balanced treated group in the exercise
(25 countries, largest 11%, Kish effective n 88 against §5k's 4), and a genuine binary,
which is what both canonical SC and GSC require.

Three things A1 must settle, all empirical:

  1  ARE THE CONTROLS CLEAN? Xu's framework requires controls "never exposed to the
     treatment in the observed time span". Our controls are never targeted 2009-17 -- but if
     they acquire policy after 2018 they are treated in the post-period, which contaminates
     the counterfactual and biases the ATT toward zero. Never checked.
  2  WHERE TO PUT THE THRESHOLD. §8b used 6+ of 9 years. Tighter is more clearly
     "persistent" but costs units and concentrates countries.
  3  WHICH VIETNAMESE SECTOR for the single-unit ADH illustration.
"""
import pandas as pd, numpy as np, gc
USA, CHINA = 842, 156
PRE=list(range(2007,2018)); POST=list(range(2018,2025)); YRS=PRE+POST; BASE=[2015,2016,2017]
P9=list(range(2009,2018)); PPOST=list(range(2018,2025))
NAME={484:'Mexico',699:'India',704:'Vietnam',458:'Malaysia',764:'Thailand',76:'Brazil',
 360:'Indonesia',608:'Philippines',792:'Turkiye',50:'Bangladesh',616:'Poland',348:'Hungary',
 643:'Russia',152:'Chile',710:'South Africa',604:'Peru',586:'Pakistan',32:'Argentina',
 170:'Colombia',642:'Romania',818:'Egypt',504:'Morocco',100:'Bulgaria',191:'Croatia',
 144:'Sri Lanka',116:'Cambodia',214:'Dominican Rep',188:'Costa Rica',218:'Ecuador',
 788:'Tunisia',688:'Serbia',804:'Ukraine',682:'Saudi Arabia',784:'UAE',634:'Qatar'}
def nm(c): return NAME.get(int(c),str(int(c)))
raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
us=raw[raw['j']==USA]
XUS=us.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
SK=us.groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
CHN=us[us['i']==CHINA].groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
pol=(raw[['i','ISIC4c','t','n_policies']].drop_duplicates(subset=['i','ISIC4c','t']))
pol['n_policies']=pd.to_numeric(pol['n_policies'],errors='coerce').fillna(0)
del raw,us; gc.collect()
for D in (XUS,SK,CHN): D.columns=[int(c) for c in D.columns]
XUS=XUS.reindex(columns=YRS); SK=SK.reindex(columns=YRS); CHN=CHN.reindex(columns=YRS)
keep=[(i,k) for i,k in XUS.index if adv.get(i,1)==0 and i!=CHINA]
XUS=XUS.loc[keep]
XUS=XUS[(XUS[YRS]>0).all(axis=1) & XUS[YRS].notna().all(axis=1)]
idx=XUS.index
on=pol.pivot_table(index=['i','ISIC4c'],columns='t',values='n_policies',aggfunc='sum')
on=(on.reindex(idx).fillna(0)>0)
pre_yrs=on[P9].sum(axis=1).values
post_yrs=on[[y for y in PPOST if y in on.columns]].sum(axis=1).values
CT=np.array([a for a,_ in idx]); SE=np.array([b for _,b in idx])
lnS=np.log(XUS[YRS].values/np.array([SK.loc[k,YRS].values for _,k in idx]))
dec=np.array([100*CHN.loc[k,BASE].mean()/SK.loc[k,BASE].mean() for _,k in idx])
print(f"sample: {len(idx):,} balanced developing ex-China country-sectors\n")

print(f"{'='*100}\n1. ARE THE CONTROLS CLEAN? never targeted 2009-17, but what happens after?"
      f"\n{'='*100}")
nev=(pre_yrs==0)
print(f"  never targeted 2009-17: {int(nev.sum()):,}")
vc=pd.Series(post_yrs[nev]).value_counts().sort_index()
print("  of those, years targeted 2018-24: "+"  ".join(
    f"{k}:{v:,}({100*v/nev.sum():.0f}%)" for k,v in vc.items()))
clean=nev&(post_yrs==0)
print(f"  -> STRICTLY clean controls (never targeted in ANY year 2009-24): "
      f"{int(clean.sum()):,} ({100*clean.sum()/nev.sum():.0f}% of the never-pre-targeted)")
print(f"     contaminated controls (acquire policy post-2018): {int((nev&(post_yrs>0)).sum()):,}"
      f"  -- these bias the ATT toward zero if kept")

print(f"\n{'='*100}\n2. THRESHOLD SWEEP (treated = targeted N+ of 9 pre-years; controls = "
      f"strictly clean)\n{'='*100}")
print(f"  {'thr':>4s} {'treated':>8s} {'ctries':>7s} {'sectors':>8s} {'largest ctry':>13s} "
      f"{'top3':>6s} {'dropped middle':>15s} {'pre-trend gap/yr':>17s}")
x=np.arange(len(PRE),dtype=float); x-=x.mean()
for thr in [5,6,7,8,9]:
    tr=(pre_yrs>=thr)
    cc=pd.Series([nm(c) for c in CT[tr]]).value_counts()
    sl=lambda M: ((M[:,:len(PRE)]*x).sum(axis=1)/(x**2).sum())
    g=sl(lnS[tr]).mean()-sl(lnS[clean]).mean()
    print(f"  {thr:4d}+ {int(tr.sum()):8d} {len(cc):7d} {len(set(SE[tr])):8d} "
          f"{100*cc.iloc[0]/tr.sum():12.0f}% {100*cc.head(3).sum()/tr.sum():5.0f}% "
          f"{int((~tr&~clean).sum()):15,d} {g:+17.4f}")
print("  (pre-trend gap = mean pre-2018 lnS slope of treated minus that of clean controls)")

print(f"\n{'='*100}\n3. VIETNAM: candidate sectors for the single-unit ADH illustration"
      f"\n{'='*100}")
vn=(CT==704)&(pre_yrs>=6)
sz=XUS[BASE].mean(axis=1).values
print(f"  Vietnam country-sectors in sample: {int((CT==704).sum())}, "
      f"of which targeted 6+ of 9: {int(vn.sum())}")
print(f"  {'sector':>8s} {'pre-yrs':>8s} {'US imports 2015-17, $m':>23s} "
      f"{'China share of US imports':>26s} {'clean donors in sector':>23s}")
for j in np.where(vn)[0][np.argsort(-sz[vn])][:10]:
    k=SE[j]
    nd=int(((SE==k)&clean).sum())
    print(f"  {str(k):>8s} {int(pre_yrs[j]):8d} {sz[j]/1e3:23,.0f} {dec[j]:25.1f}% {nd:23d}")
