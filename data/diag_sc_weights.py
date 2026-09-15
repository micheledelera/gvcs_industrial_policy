"""Who is in each synthetic control, and with what weight.

omega is solved per sector (MVA_W = 2, zeta -> 0, matching the 2010-2017 log US export
path plus MVA). This dumps them: concentration, which countries carry weight across all
58 sectors, and the full donor list for a few named sectors.
"""
import pandas as pd, numpy as np
exec(open("fit_sc_mvasweep2.py").read().split("def run(mw):")[0])
MW=2.0; ZS=1e-6
NAME={604:'Peru',764:'Thailand',710:'South Africa',484:'Mexico',608:'Philippines',
 458:'Malaysia',214:'Dominican Rep',222:'El Salvador',188:'Costa Rica',320:'Guatemala',
 780:'Trinidad & Tob',152:'Chile',144:'Sri Lanka',704:'Vietnam',688:'Serbia',
 340:'Honduras',504:'Morocco',558:'Nicaragua',858:'Uruguay',422:'Lebanon',591:'Panama',
 50:'Bangladesh',328:'Guyana',388:'Jamaica',170:'Colombia',795:'Turkmenistan',
 862:'Venezuela',740:'Suriname',860:'Uzbekistan',716:'Zimbabwe',116:'Cambodia',
 600:'Paraguay',748:'Eswatini',332:'Haiti',112:'Belarus',400:'Jordan',288:'Ghana',
 686:'Senegal',417:'Kyrgyzstan',32:'Argentina',68:'Bolivia',76:'Brazil',100:'Bulgaria',
 191:'Croatia',218:'Ecuador',348:'Hungary',360:'Indonesia',566:'Nigeria',586:'Pakistan',
 616:'Poland',642:'Romania',643:'Russia',682:'Saudi Arabia',699:'India',784:'UAE',
 788:'Tunisia',792:'Turkiye',804:'Ukraine',818:'Egypt',12:'Algeria',24:'Angola',
 51:'Armenia',31:'Azerbaijan',48:'Bahrain',204:'Benin',72:'Botswana',96:'Brunei',
 108:'Burundi',120:'Cameroon',148:'Chad',178:'Congo',384:'Cote dIvoire',192:'Cuba',
 262:'Djibouti',12:'Algeria',231:'Ethiopia',242:'Fiji',268:'Georgia',270:'Gambia',
 324:'Guinea',368:'Iraq',404:'Kenya',414:'Kuwait',418:'Laos',426:'Lesotho',
 430:'Liberia',434:'Libya',450:'Madagascar',454:'Malawi',466:'Mali',478:'Mauritania',
 480:'Mauritius',496:'Mongolia',498:'Moldova',104:'Myanmar',516:'Namibia',524:'Nepal',
 562:'Niger',512:'Oman',586:'Pakistan',598:'Papua New Guinea',634:'Qatar',646:'Rwanda',
 686:'Senegal',694:'Sierra Leone',706:'Somalia',144:'Sri Lanka',729:'Sudan',
 760:'Syria',762:'Tajikistan',834:'Tanzania',768:'Togo',800:'Uganda',860:'Uzbekistan',
 704:'Vietnam',887:'Yemen',894:'Zambia',24:'Angola',132:'Cabo Verde',226:'Eq Guinea',
 266:'Gabon',624:'Guinea-Bissau',175:'Mayotte',508:'Mozambique',854:'Burkina Faso'}
def nm(c):
    return f"{NAME.get(int(c), str(int(c)))}"

W=[]
for k,g in m.groupby(level='ISIC4c', observed=True):
    t_,d_=g[g['T']==1],g[g['T']==0]
    if len(t_)==0 or len(d_)<MIN_DON: continue
    P=Yv.loc[d_.index,PATH].values; bt=Yv.loc[t_.index,PATH].values.mean(axis=0)
    A=np.hstack([P,(MW*d_['mvaz'].values)[:,None]]); b=np.append(bt,MW*t_['mvaz'].mean())
    sig=np.diff(P,axis=1).std()
    z2=(((len(t_)*len(POST))**0.5)*(sig**2))*ZS+1e-10
    w,_=solve_w2(A,b,z2,len(PATH))
    fit=P.T@w; rmse=float(np.sqrt(((fit+np.mean(bt-fit)-bt)**2).mean()))
    for (i_,_),wt in zip(d_.index,w):
        W.append({'sector':k,'donor':i_,'w':wt,'rmse':rmse,'n_don':len(d_),'n_tr':len(t_)})
    for i_,_ in t_.index:
        W.append({'sector':k,'donor':i_,'w':np.nan,'rmse':rmse,'n_don':len(d_),
                  'n_tr':len(t_),'treated':1})
W=pd.DataFrame(W); W['treated']=W.get('treated',0)
W['treated']=W['treated'].fillna(0)
W.to_csv("sc_weights.csv",index=False)
D=W[W['treated']==0]
print(f"{D['sector'].nunique()} sectors, {len(D):,} donor slots\n")

print("="*72); print("CONCENTRATION, per sector"); print("="*72)
c=D.groupby('sector').agg(n=('w','size'), wmax=('w','max'),
                          eff=('w',lambda v:1/ (v**2).sum()),
                          top5=('w',lambda v: v.nlargest(5).sum()))
print(f"  donors available   : median {c['n'].median():.0f}")
print(f"  largest weight     : median {c['wmax'].median():.3f}   p90 {c['wmax'].quantile(.9):.3f}")
print(f"  effective donors   : median {c['eff'].median():.1f}")
print(f"  top-5 weight share : median {c['top5'].median():.3f}")
print(f"  donors with w>0.01 : median {D[D['w']>0.01].groupby('sector').size().median():.0f}")

print("\n"+"="*72); print("WHICH COUNTRIES CARRY THE SYNTHETIC CONTROLS"); print("="*72)
agg=D.groupby('donor').agg(total_w=('w','sum'), sectors=('sector','nunique'),
                           mean_w=('w','mean')).sort_values('total_w',ascending=False)
agg['share_of_all_weight']=agg['total_w']/D['w'].sum()
agg2=agg.head(25).copy(); agg2.index=[nm(i) for i in agg2.index]
print(agg2.round(4).to_string())
print(f"\n  top 10 countries hold {100*agg['share_of_all_weight'].head(10).sum():.1f}% of all donor weight")
print(f"  {int((agg['share_of_all_weight']>0.01).sum())} countries hold >1% each; "
      f"{len(agg)} countries appear at all")

print("\n"+"="*72); print("THREE SECTORS IN FULL"); print("="*72)
big=D.groupby('sector').apply(lambda g: m.loc[[(i,g.name) for i in
     m.xs(g.name,level='ISIC4c',drop_level=False).index.get_level_values('i')
     if (i,g.name) in m.index],'wpre'].sum(), include_groups=False)
pick=[c['n'].idxmax()]
best=c.loc[c.index.isin(D['sector'].unique())]
pick=[big.idxmax(), c['eff'].idxmin(), c['eff'].idxmax()]
lab=['largest sector by US import value','most concentrated omega','most diffuse omega']
for k,L in zip(pick,lab):
    g=D[D['sector']==k].sort_values('w',ascending=False)
    tr=W[(W['sector']==k)&(W['treated']==1)]['donor'].tolist()
    print(f"\n--- sector {k}  ({L})   {int(g['n_don'].iloc[0])} donors, "
          f"{int(g['n_tr'].iloc[0])} treated, pre-RMSE {g['rmse'].iloc[0]:.4f}")
    print(f"    TREATED ({len(tr)}): " + ", ".join(nm(x) for x in tr))
    print(f"    {'donor':>16s}{'weight':>9s}   (w >= 0.005)")
    for _,r in g[g['w']>=0.005].iterrows():
        bar='#'*int(round(60*r['w']))
        print(f"    {nm(r['donor']):>16s}{r['w']:>9.4f}   {bar}")
    print(f"    ...{int((g['w']<0.005).sum())} further donors share {g[g['w']<0.005]['w'].sum():.4f}")
