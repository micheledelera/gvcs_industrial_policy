"""ABADIE STEP 0. Which policy-using developing countries lie inside the convex hull of
the non-policy donor pool?

Canonical synthetic control uses non-negative weights summing to one, so the treated unit
must be reproducible as a convex combination of donors. Cunningham on Texas: "If Texas had
been extreme -- either the smallest or the largest -- then that would be impossible." This
checks who is extreme.

Unit      country (aggregate), developing excluding China
Outcome   log exports to the US, all 125 ISIC-4 manufacturing sectors, 2007-2017
Treated   any recorded GTA industrial policy in 2015-2017
Donors    no recorded industrial policy in any sector, any year
Test      (a) is the treated country's pre-period level inside the donor min-max?
          (b) solve the canonical SC on lagged outcomes alone and report pre-RMSPE,
              which is the real feasibility answer -- inside the hull means a convex
              combination fits well.
"""
import pandas as pd, numpy as np, gc
USA, CHINA = 842, 156
PRE=list(range(2007,2018))
NAME={604:'Peru',764:'Thailand',710:'South Africa',484:'Mexico',608:'Philippines',
 458:'Malaysia',214:'Dominican Rep',222:'El Salvador',188:'Costa Rica',320:'Guatemala',
 780:'Trinidad&Tob',152:'Chile',144:'Sri Lanka',704:'Vietnam',688:'Serbia',340:'Honduras',
 504:'Morocco',558:'Nicaragua',858:'Uruguay',422:'Lebanon',591:'Panama',50:'Bangladesh',
 328:'Guyana',388:'Jamaica',170:'Colombia',699:'India',360:'Indonesia',76:'Brazil',
 792:'Turkiye',818:'Egypt',586:'Pakistan',616:'Poland',642:'Romania',348:'Hungary',
 203:'Czechia',100:'Bulgaria',191:'Croatia',705:'Slovenia',703:'Slovakia',233:'Estonia',
 428:'Latvia',440:'Lithuania',804:'Ukraine',643:'Russia',398:'Kazakhstan',32:'Argentina',
 218:'Ecuador',68:'Bolivia',600:'Paraguay',862:'Venezuela',788:'Tunisia',12:'Algeria',
 566:'Nigeria',404:'Kenya',800:'Uganda',834:'Tanzania',231:'Ethiopia',854:'Burkina Faso',
 384:'Cote dIvoire',288:'Ghana',686:'Senegal',450:'Madagascar',480:'Mauritius',
 508:'Mozambique',894:'Zambia',716:'Zimbabwe',426:'Lesotho',748:'Eswatini',72:'Botswana',
 516:'Namibia',24:'Angola',120:'Cameroon',178:'Congo',266:'Gabon',682:'Saudi Arabia',
 784:'UAE',512:'Oman',634:'Qatar',414:'Kuwait',48:'Bahrain',400:'Jordan',760:'Syria',
 368:'Iraq',364:'Iran',887:'Yemen',116:'Cambodia',418:'Laos',104:'Myanmar',524:'Nepal',
 764:'Thailand',158:'Taiwan',344:'Hong Kong',702:'Singapore',410:'Korea',496:'Mongolia',
 860:'Uzbekistan',762:'Tajikistan',417:'Kyrgyzstan',795:'Turkmenistan',31:'Azerbaijan',
 51:'Armenia',268:'Georgia',112:'Belarus',498:'Moldova',8:'Albania',70:'BosniaHerz',
 807:'N Macedonia',499:'Montenegro',688:'Serbia',332:'Haiti',192:'Cuba',214:'Dominican Rep',
 780:'Trinidad&Tob',740:'Suriname',328:'Guyana',84:'Belize',222:'El Salvador',
 340:'Honduras',558:'Nicaragua',188:'Costa Rica',591:'Panama',604:'Peru',152:'Chile',
 858:'Uruguay',600:'Paraguay',242:'Fiji',598:'PapuaNG',90:'Solomon Is',548:'Vanuatu'}
def nm(c): return NAME.get(int(c), str(int(c)))

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
raw['share_frac_policies']=pd.to_numeric(raw['share_frac_policies'],errors='coerce').fillna(0)
raw['n_policies']=pd.to_numeric(raw['n_policies'],errors='coerce').fillna(0)
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
us=raw[raw['j']==USA]
X=us.groupby(['i','t'],observed=True)['imports'].sum().unstack('t')
X.columns=[int(c) for c in X.columns]; X=X[sorted(X.columns)]
usmkt=X.sum(axis=0)
pol=raw.groupby('i')[['share_frac_policies','n_policies']].max()
del raw,us; gc.collect()

dev=[i for i in X.index if adv.get(i,1)==0 and i!=CHINA]
X=X.loc[dev]
X=X[(X[PRE]>0).all(axis=1)]
T=pd.Series([bool(pol['n_policies'].get(i,0)>0) for i in X.index], index=X.index)
print(f"developing ex-China with a complete 2007-2017 US export series: {len(X)}")
print(f"  policy users : {int(T.sum())}")
print(f"  donors       : {int((~T).sum())}\n")

Y=np.log(X[PRE])
tr=Y[T]; dn=Y[~T]
lo,hi=dn.min(axis=0), dn.max(axis=0)

def proj_simplex(v):
    u=np.sort(v)[::-1]; c=np.cumsum(u)-1.0; r=np.arange(1,len(v)+1)
    cond=u-c/r>0
    if not cond.any(): return np.ones_like(v)/len(v)
    rho=r[cond][-1]; return np.maximum(v-c[cond][-1]/rho,0.0)
def sc(A,b,iters=4000):
    """canonical Abadie: non-negative weights summing to 1, NO intercept."""
    n=A.shape[0]; w=np.ones(n)/n; L=np.linalg.norm(A,2)**2/A.shape[1]+1e-12
    for _ in range(iters):
        w=proj_simplex(w-(A@(A.T@w-b)*2.0/A.shape[1])/L)
    return w

rows=[]
for i in tr.index:
    b=tr.loc[i].values
    D=dn.drop(index=[i],errors='ignore')
    w=sc(D.values,b)
    fit=D.values.T@w
    rows.append({'i':i,'name':nm(i),
                 'us_exports_2015_17':X.loc[i,[2015,2016,2017]].mean(),
                 'pct_of_US_mkt':100*X.loc[i,[2015,2016,2017]].mean()/usmkt[[2015,2016,2017]].mean(),
                 'inside_hull':bool((b>=lo.values).all() and (b<=hi.values).all()),
                 'yrs_above_max':int((b>hi.values).sum()),
                 'rmspe':float(np.sqrt(((fit-b)**2).mean())),
                 'rmspe_pct':float(np.sqrt(((fit-b)**2).mean())/np.abs(b).mean()*100),
                 'w_max':w.max(),'n_pos':int((w>0.01).sum())})
r=pd.DataFrame(rows).sort_values('us_exports_2015_17',ascending=False)
r.to_csv("sc01_feasibility.csv",index=False)
print(f"{'country':<16s}{'US exp 15-17 ($k)':>19s}{'% US mkt':>10s}{'in hull':>9s}"
      f"{'yrs>max':>9s}{'pre-RMSPE':>11s}{'(log pts)':>10s}{'donors w>.01':>14s}")
print("="*100)
for _,x in r.iterrows():
    print(f"{x['name']:<16s}{x['us_exports_2015_17']:>19,.0f}{x['pct_of_US_mkt']:>10.2f}"
          f"{('YES' if x['inside_hull'] else 'no'):>9s}{x['yrs_above_max']:>9d}"
          f"{x['rmspe']:>11.4f}{'':>10s}{x['n_pos']:>14d}")
print("\n"+"="*100)
print(f"policy users inside the donor convex hull on levels: "
      f"{int(r['inside_hull'].sum())} of {len(r)}")
print(f"  median pre-RMSPE inside : {r.loc[r['inside_hull'],'rmspe'].median():.4f} log points")
print(f"  median pre-RMSPE outside: {r.loc[~r['inside_hull'],'rmspe'].median():.4f} log points")
print(f"\nlargest donor, US exports 2015-17: {nm(X.loc[~T,[2015,2016,2017]].mean(axis=1).idxmax())}"
      f"  ${X.loc[~T,[2015,2016,2017]].mean(axis=1).max():,.0f}k")
