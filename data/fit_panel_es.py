"""Panel event study in levels -- the panel form of the long difference.

  s_ikt = sum_tau b1_tau (IP_ik x Dec_k x 1[t=tau])
        + sum_tau b2_tau (IP_ik x 1[t=tau])
        + a_ik + a_kt + e_ikt                          ref tau = 2017

s_ikt   country i's share of US imports in sector k, year t, percentage points
IP_ik   pre-2018 (2015-17 mean) policy measure, standardised
Dec_k   China's 2015-17 share of US imports in sector k, standardised
a_ik    absorbs the pair level (the differencing the long difference did)
a_kt    absorbs sector-year (the sector dummies of the long difference)

Sample: developing ex-China, as in the long difference. Weighted by pre-period US
imports. Clustered by country. Run for all four policy measures in sequence.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
USA, CHINA, REF = 842, 156, 2017
PRE=[2015,2016,2017]
MEAS=['n_policies','frac_policies','share_n_policies','share_frac_policies']

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
for c in MEAS: raw[c]=pd.to_numeric(raw[c],errors='coerce').fillna(0).astype('float32')
us=raw[raw['j']==USA]
x=us.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum()
tot=x.groupby(level=['ISIC4c','t'],observed=True).transform('sum')
s=(x/tot.replace(0,np.nan)*100).rename('s')                      # pp
xp=(us[us['t'].isin(PRE)].groupby(['i','ISIC4c'],observed=True)['imports'].sum()/3).rename('w')
chn=s.xs(CHINA,level='i') if CHINA in s.index.get_level_values('i') else None
dec=(chn[chn.index.get_level_values('t').isin(PRE)].groupby('ISIC4c',observed=True).mean()
     ).rename('Dec')
ip=(raw[raw['t'].isin(PRE)][['i','ISIC4c','t']+MEAS].drop_duplicates(subset=['i','ISIC4c','t'])
    .groupby(['i','ISIC4c'],observed=True)[MEAS].mean())
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
act=raw.loc[raw['imports']>0,['i','ISIC4c']].drop_duplicates()
years=sorted(raw['t'].unique())
del raw, us; gc.collect()

p=act.loc[act.index.repeat(len(years))].copy(); p['t']=np.tile(years,len(act))
p=p.set_index(['i','ISIC4c','t']).join(s).reset_index()
p=p.merge(xp,on=['i','ISIC4c'],how='left').merge(ip,on=['i','ISIC4c'],how='left')\
   .merge(dec,on='ISIC4c',how='left')
for c in ['s','w','Dec']+MEAS: p[c]=p[c].fillna(0.0)
p['Advanced_i']=p['i'].map(adv); p=p[(p['Advanced_i']==0)&(p['i']!=CHINA)].copy()
p['Dz']=(p['Dec']-p['Dec'].mean())/p['Dec'].std()
p['fe_ik']=p.groupby(['i','ISIC4c'],observed=True).ngroup().astype('int32')
p['fe_kt']=p.groupby(['ISIC4c','t'],observed=True).ngroup().astype('int32')
p['cl_i']=p['i'].astype('int32')
log(f"panel {p.shape} | pairs {p['fe_ik'].nunique():,} | years {years[0]}-{years[-1]} "
    f"| DV s: mean {p['s'].mean():.4f}pp sd {p['s'].std():.4f}")

tv=p['t'].values
for M in MEAS:
    ipz=(p[M]/p[M].std()).values
    names_i, names_d = [], []
    for y in years:
        if y==REF: continue
        p[f"I_{y}"]=ipz*(tv==y);            names_i.append(f"I_{y}")
        p[f"D_{y}"]=ipz*p['Dz'].values*(tv==y); names_d.append(f"D_{y}")
    rhs=" + ".join(names_d+names_i)
    fit=pf.feols(f"s ~ {rhs} | fe_ik + fe_kt", data=p, weights="w",
                 vcov={"CRV1":"cl_i"}, lean=True, store_data=False, copy_data=False)
    t=fit.tidy()
    print(f"\n{'='*78}\n{M}   N={fit._N:,}   ref {REF}\n{'='*78}")
    print(f"{'year':>6s} {'IP x Dec x year':>26s} {'IP x year':>26s}")
    for y in years:
        if y==REF:
            print(f"{y:>6d} {'— reference —':>26s} {'— reference —':>26s}"); continue
        out=f"{y:>6d}"
        for nm in [f"D_{y}", f"I_{y}"]:
            if nm in t.index:
                r=t.loc[nm]
                st='*' if r['Pr(>|t|)']<0.05 else ('.' if r['Pr(>|t|)']<0.10 else ' ')
                out+=f"  {r['Estimate']:+8.4f} ({r['Std. Error']:.4f}){st}".rjust(27)
            else: out+=f"{'dropped':>27s}"
        print(out)
    for lbl,nms in [("IP x Dec",names_d),("IP",names_i)]:
        pre=[n for n in nms if int(n.split('_')[1])<2018 and n in t.index]
        if pre:
            tp=t.loc[pre]
            print(f"  {lbl:9s} pre-period: {len(pre)} coefs, mean {tp['Estimate'].mean():+.4f}, "
                  f"max|t| {tp['t value'].abs().max():.2f}, n(p<.05)={int((tp['Pr(>|t|)']<.05).sum())}")
    for c in names_i+names_d: del p[c]
    del fit; gc.collect()
log("ALL DONE")
