"""Full reporting output for the between-country long difference."""
import pandas as pd, numpy as np, statsmodels.api as sm, gc
USA, CHINA = 842, 156
PRE0, PRE, POST = [2010,2011,2012],[2015,2016,2017],[2022,2023,2024]
raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
MEAS=['n_policies','frac_policies','share_n_policies','share_frac_policies']
for c in MEAS: raw[c]=pd.to_numeric(raw[c],errors='coerce').fillna(0).astype('float32')
us=raw[raw['j']==USA]
def share(years):
    x=us[us['t'].isin(years)].groupby(['i','ISIC4c'],observed=True)['imports'].sum()/len(years)
    tot=x.groupby(level='ISIC4c',observed=True).transform('sum')
    return (x/tot.replace(0,np.nan)).rename('s'), x.rename('x')
s0,_=share(PRE0); s1,x1=share(PRE); s2,_=share(POST)
ip=(raw[raw['t'].isin(PRE)][['i','ISIC4c','t']+MEAS].drop_duplicates(subset=['i','ISIC4c','t'])
    .groupby(['i','ISIC4c'],observed=True)[MEAS].mean())
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
act=raw.loc[raw['imports']>0,['i','ISIC4c']].drop_duplicates()
del raw, us; gc.collect()
d=act.set_index(['i','ISIC4c']).join([s0.rename('s0'),s1.rename('s1'),s2.rename('s2'),x1,ip]).reset_index()
for c in ['s0','s1','s2','x']+MEAS: d[c]=d[c].fillna(0.0)
d['Advanced_i']=d['i'].map(adv); d=d[(d['Advanced_i']==0)&(d['i']!=CHINA)].copy()
d['d_post']=(d['s2']-d['s1'])*100; d['d_pre']=(d['s1']-d['s0'])*100; d['s1_pp']=d['s1']*100
M='share_frac_policies'; d['IPz']=d[M]/d[M].std()

print("="*78); print("SAMPLE"); print("="*78)
print(f"unit                     country x ISIC4 sector")
print(f"observations             {len(d):,}")
print(f"countries (clusters)     {d['i'].nunique()}   developing, excluding China")
print(f"sectors                  {d['ISIC4c'].nunique()}  (ISIC rev.4, 4-digit)")
print(f"cells with any policy    {int((d[M]>0).sum()):,} ({100*(d[M]>0).mean():.1f}%)")
print("\n"+"="*78); print("VARIABLES"); print("="*78)
v=pd.DataFrame({'mean':[d['d_post'].mean(),d['d_pre'].mean(),d['s1_pp'].mean(),d[M].mean(),d['IPz'].mean()],
 'sd':[d['d_post'].std(),d['d_pre'].std(),d['s1_pp'].std(),d[M].std(),d['IPz'].std()],
 'min':[d['d_post'].min(),d['d_pre'].min(),d['s1_pp'].min(),d[M].min(),d['IPz'].min()],
 'max':[d['d_post'].max(),d['d_pre'].max(),d['s1_pp'].max(),d[M].max(),d['IPz'].max()]},
 index=['d_post (pp)','d_pre (pp)','s1_pp (pp)',M,'IPz (standardised)'])
print(v.round(6).to_string())
print(f"\ncorr(d_post, d_pre) = {d['d_post'].corr(d['d_pre']):+.4f}")
print(f"weight = mean US imports 2015-17, total {d['x'].sum():,.0f}")

sec=pd.get_dummies(d['ISIC4c'].astype(str),prefix='k',drop_first=True).astype(float)
w=d['x'].clip(lower=0).values
SPECS=[("(1) naive",['IPz'],False),("(2) + sector FE",['IPz'],True),
       ("(3) + pre-trend",['IPz','d_pre'],True),("(4) + initial share",['IPz','d_pre','s1_pp'],True)]
res={}
for name,X,fe in SPECS:
    Mx=d[X].copy()
    if fe: Mx=pd.concat([Mx,sec],axis=1)
    m=sm.WLS(d['d_post'],sm.add_constant(Mx),weights=w).fit(cov_type='cluster',
        cov_kwds={'groups':d['i']})
    res[name]=(m,X,fe)
print("\n"+"="*78); print("RESULTS  —  DV: change in share of US imports in sector k, pp")
print("="*78)
hdr=f"{'':22s}"+"".join(f"{n:>19s}" for n,_,_ in SPECS); print(hdr)
for term,label in [('IPz','IP (share_frac_pol)'),('d_pre','pre-trend d_pre'),('s1_pp','initial share')]:
    r1=f"{label:22s}"; r2=f"{'':22s}"
    for name,X,fe in SPECS:
        m=res[name][0]
        if term in X:
            st='***' if m.pvalues[term]<.01 else '**' if m.pvalues[term]<.05 else '*' if m.pvalues[term]<.10 else ''
            r1+=f"{m.params[term]:+.4f}{st:>4s}".rjust(19); r2+=f"({m.bse[term]:.4f})".rjust(19)
        else:
            r1+=f"{'':>19s}"; r2+=f"{'':>19s}"
    print(r1); print(r2)
print("-"*78)
for lab,fn in [("Sector FE (124 dummies)",lambda n:"YES" if res[n][2] else "NO"),
               ("Weighted by pre US imports",lambda n:"YES"),
               ("SE clustered by country",lambda n:"YES"),
               ("R-squared",lambda n:f"{res[n][0].rsquared:.3f}"),
               ("Clusters",lambda n:f"{d['i'].nunique()}"),
               ("N",lambda n:f"{int(res[n][0].nobs):,}")]:
    print(f"{lab:22s}"+"".join(f"{fn(n):>19s}" for n,_,_ in SPECS))
