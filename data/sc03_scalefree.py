"""ABADIE STEP 1. Which scale-free outcome brings the big policy users inside the hull?

§4a: on log US exports (a level), Mexico/India/Vietnam/Malaysia/Thailand are above the
donor maximum in all 11 pre-treatment years. §4b: disaggregating to country-sector admits
only 5% of treated trade value. Route A asks whether a genuinely SCALE-FREE outcome fixes
this without an intercept -- i.e. keeping canonical Abadie intact.

An outcome is only route A if it is scale-free IN LEVELS. Indexing exports to a base year
is NOT: that is demeaning, i.e. Ferman-Pinto, i.e. route B wearing a disguise.

Four candidates, all computable from BACI alone:

  1. us_share_own    US share of country i's own manufacturing exports.
                     "How US-oriented is this country." Bounded [0,1].
  2. us_orientation  (i's share of US imports) / (i's share of rest-of-world imports).
                     An RCA for the US destination: scale-free by construction, since a
                     country twice as large in both is unchanged. Directly measures
                     whether i disproportionately supplies America -- the decoupling
                     question.
  3. us_share_mkt    i's share of total US imports. NOT scale-free; included as the
                     benchmark that §4a already failed.
  4. us_share_dec    us_share_own restricted to sectors where China held >=25% of the US
                     market pre-period -- the decoupling-exposed basket.

For each: is every treated country inside the donor min-max across 2007-2017, and what
pre-RMSPE does canonical SC (non-negative weights, sum to one, no intercept) achieve?
"""
import pandas as pd, numpy as np, gc
USA, CHINA = 842, 156
PRE=list(range(2007,2018)); BASE=[2015,2016,2017]
NAME={484:'Mexico',699:'India',704:'Vietnam',458:'Malaysia',764:'Thailand',76:'Brazil',
 360:'Indonesia',608:'Philippines',792:'Turkiye',50:'Bangladesh',616:'Poland',348:'Hungary',
 643:'Russia',152:'Chile',710:'South Africa',604:'Peru',586:'Pakistan',32:'Argentina',
 170:'Colombia',784:'UAE',642:'Romania',218:'Ecuador',682:'Saudi Arabia',818:'Egypt',
 504:'Morocco',68:'Bolivia',634:'Qatar',804:'Ukraine',100:'Bulgaria',191:'Croatia',
 404:'Kenya',788:'Tunisia',566:'Nigeria',688:'Serbia',12:'Algeria',508:'Mozambique',
 288:'Ghana',524:'Nepal',368:'Iraq',214:'Dominican Rep'}
def nm(c): return NAME.get(int(c),str(int(c)))

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
raw['n_policies']=pd.to_numeric(raw['n_policies'],errors='coerce').fillna(0)
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
pol=raw.groupby('i')['n_policies'].max()
ALLY=sorted(raw['t'].unique())
us=raw[raw['j']==USA]
usik=us.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum()
usi =us.groupby(['i','t'],observed=True)['imports'].sum().unstack('t')
alli=raw.groupby(['i','t'],observed=True)['imports'].sum().unstack('t')
usmkt=us.groupby('t',observed=True)['imports'].sum()
wldmkt=raw.groupby('t',observed=True)['imports'].sum()
# decoupling-exposed sectors
tot_k=us.groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
chn_k=us[us['i']==CHINA].groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
chn_k.columns=[int(c) for c in chn_k.columns]; tot_k.columns=[int(c) for c in tot_k.columns]
DEC=set((chn_k[BASE].mean(axis=1)/tot_k[BASE].mean(axis=1)*100).pipe(lambda s: s[s>=25]).index)
usik_dec=us[us['ISIC4c'].isin(DEC)].groupby(['i','t'],observed=True)['imports'].sum().unstack('t')
alli_dec=raw[raw['ISIC4c'].isin(DEC)].groupby(['i','t'],observed=True)['imports'].sum().unstack('t')
del raw,us; gc.collect()
for D in (usi,alli,usik_dec,alli_dec): D.columns=[int(c) for c in D.columns]
usmkt.index=[int(c) for c in usmkt.index]; wldmkt.index=[int(c) for c in wldmkt.index]
print(f"decoupling-exposed sectors (China >=25% of US market, 2015-17): {len(DEC)} of 125\n")

O={}
O['1. us_share_own']   = (usi/alli.replace(0,np.nan)*100)
row_us=(usi.div(usmkt,axis=1)); row_rw=((alli-usi).div(wldmkt-usmkt,axis=1))
O['2. us_orientation'] = (row_us/row_rw.replace(0,np.nan))
O['3. us_share_mkt']   = (usi.div(usmkt,axis=1)*100)
O['4. us_share_dec']   = (usik_dec/alli_dec.replace(0,np.nan)*100)

def proj_simplex(v):
    u=np.sort(v)[::-1]; c=np.cumsum(u)-1.0; r=np.arange(1,len(v)+1)
    cond=u-c/r>0
    if not cond.any(): return np.ones_like(v)/len(v)
    return np.maximum(v-c[cond][-1]/r[cond][-1],0.0)
def sc(A,b,iters=4000):
    n=A.shape[0]; w=np.ones(n)/n; L=np.linalg.norm(A,2)**2/A.shape[1]+1e-12
    for _ in range(iters): w=proj_simplex(w-(A@(A.T@w-b)*2.0/A.shape[1])/L)
    return w

BIG=[484,704,699,458,764,360,608,76,792,50]
summary=[]
for lab,Y in O.items():
    Y=Y[[c for c in PRE if c in Y.columns]]
    Y=Y.loc[[i for i in Y.index if adv.get(i,1)==0 and i!=CHINA]]
    Y=Y.dropna()
    Y=Y[(Y>0).all(axis=1)]
    T=pd.Series({i: bool(pol.get(i,0)>0) for i in Y.index})
    tr,dn=Y[T.values],Y[~T.values]
    if len(tr)<5 or len(dn)<10: print(f"{lab}: too few"); continue
    lo,hi=dn.min(axis=0).values,dn.max(axis=0).values
    res=[]
    for i in tr.index:
        b=tr.loc[i].values; D=dn
        w=sc(D.values,b); fit=D.values.T@w
        res.append({'i':i,'inside':bool((b>=lo).all() and (b<=hi).all()),
                    'rmspe':float(np.sqrt(((fit-b)**2).mean())),
                    'rmspe_rel':float(np.sqrt(((fit-b)**2).mean())/np.abs(b).mean()),
                    'val':float(usi.loc[i,BASE].mean()) if i in usi.index else 0.0})
    R=pd.DataFrame(res)
    big=R[R['i'].isin(BIG)]
    print(f"{'='*94}\n{lab}   {len(tr)} treated, {len(dn)} donors\n{'='*94}")
    print(f"  inside hull: {int(R['inside'].sum())}/{len(R)} countries "
          f"({100*R.loc[R['inside'],'val'].sum()/R['val'].sum():.1f}% of treated US trade value)")
    print(f"  median relative pre-RMSPE: inside {R.loc[R['inside'],'rmspe_rel'].median():.3f}"
          f"   outside {R.loc[~R['inside'],'rmspe_rel'].median() if (~R['inside']).any() else float('nan'):.3f}")
    print(f"  {'the ten largest exporters':<28s}{'level 15-17':>13s}{'in hull':>9s}{'rel RMSPE':>11s}")
    for _,x in big.sort_values('val',ascending=False).iterrows():
        lv=Y.loc[x['i'],BASE].mean() if x['i'] in Y.index else np.nan
        print(f"  {nm(x['i']):<28s}{lv:>13.3f}{('YES' if x['inside'] else 'no'):>9s}"
              f"{x['rmspe_rel']:>11.3f}")
    summary.append({'outcome':lab,'n_tr':len(tr),'n_don':len(dn),
                    'inside_n':int(R['inside'].sum()),
                    'inside_val_pct':100*R.loc[R['inside'],'val'].sum()/R['val'].sum(),
                    'big_inside':int(big['inside'].sum()),'big_n':len(big),
                    'med_rmspe_rel':R['rmspe_rel'].median()})
    print()
S=pd.DataFrame(summary); S.to_csv("sc03_scalefree.csv",index=False)
print("="*94); print("SUMMARY"); print("="*94)
print(S.round(3).to_string(index=False))
