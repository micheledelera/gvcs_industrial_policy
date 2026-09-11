"""Does measured capability predict the PRE-period trend?

If capability were an additive time-invariant level effect, differencing would remove
it and capability would be orthogonal to delta-s^pre. If capability is instead a
LOADING on sector shocks, it predicts the change, differencing does not help, and the
"differencing removes capability" defence fails on its own terms.

  delta-s^pre_ik = s(2015-17) - s(2010-12)      the change, pre-treatment
  regressed on the same five pre-determined country controls, with alpha_k

alpha_k is in so this is not "big countries grow" -- it is, within a sector, whether
more-capable countries were on a different trajectory BEFORE anything happened.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc
USA, CHINA = 842, 156
PRE0, PRE = [2010,2011,2012], [2015,2016,2017]
C2ISO={699:356,757:756,579:578,251:250,842:840,381:380,490:158,729:729}
XLSX=("/root/.claude/uploads/618110ef-d894-58be-8e00-e7f2bf548b03/"
      "4ad6510b-0154963EF4289D02DEA410D8D38E4E6A8C6D.xlsx")
CTRL=['log_mva_pc','mva_gdp','mfg_emp','log_exp','ECI']

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
us=raw[raw['j']==USA]
def share(yrs):
    x=us[us['t'].isin(yrs)].groupby(['i','ISIC4c'],observed=True)['imports'].sum()/len(yrs)
    tot=x.groupby(level='ISIC4c',observed=True).transform('sum')
    return (x/tot.replace(0,np.nan)).rename('s'), x.rename('w')
s0,_=share(PRE0); s1,w1=share(PRE)
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
act=raw.loc[raw['imports']>0,['i','ISIC4c']].drop_duplicates()
chn=s1.xs(CHINA,level='i').rename('Dec')
prew=raw[raw['t'].isin(PRE)]
X=(prew.groupby(['i','ISIC4c'],observed=True)['imports'].sum()/len(PRE)).unstack(fill_value=0.0)
del raw,us,prew; gc.collect()
shm=X.div(X.sum(axis=1).replace(0,np.nan),axis=0); world=X.sum(axis=0)/X.sum().sum()
M=((shm.div(world,axis=1))>=1).astype(float); M=M.loc[M.sum(axis=1)>0,M.sum(axis=0)>0]
kc,kp=M.sum(axis=1),M.sum(axis=0)
vals,vecs=np.linalg.eig(((M.div(kc,axis=0))@(M.div(kp,axis=1).T)).values)
K=np.real(vecs[:,np.argsort(-np.real(vals))[1]])
ECI=pd.Series((K-K.mean())/K.std(),index=M.index)
if ECI.corr(pd.Series(kc,index=M.index))<0: ECI=-ECI
u=pd.read_excel(XLSX,sheet_name='Data'); u=u[u['Year'].isin(PRE)]
U=u.pivot_table(index='CountryCode',columns='VariableCode',values='Value',aggfunc='mean')

d=act.set_index(['i','ISIC4c']).join([s0.rename('s0'),s1.rename('s1'),w1]).reset_index()
for c in ['s0','s1','w']: d[c]=d[c].fillna(0.0)
d['Advanced_i']=d['i'].map(adv); d=d[(d['Advanced_i']==0)&(d['i']!=CHINA)].copy()
d=d.merge(chn,on='ISIC4c',how='left'); d['Dec']=d['Dec'].fillna(0.0)
d=d[d['w']>0].copy()
d['d_pre']=(d['s1']-d['s0'])*100; d['s0_pp']=d['s0']*100

ctry=sorted(d['i'].unique()); iso=[C2ISO.get(c,c) for c in ctry]
C=pd.DataFrame(index=pd.Index(ctry,name='i'))
C['log_mva_pc']=np.log(U['NV_IND_MANFPC'].reindex(iso).values)
C['mva_gdp']=U['NV_IND_MANF'].reindex(iso).values
C['mfg_emp']=U['SL_TLF_MANF'].reindex(iso).values
C['log_exp']=np.log(X.sum(axis=1).reindex(ctry).replace(0,np.nan).values)
C['ECI']=ECI.reindex(ctry).values
C=(C-C.mean())/C.std()
d=d.merge(C,left_on='i',right_index=True,how='left').dropna(subset=CTRL)
for c in CTRL: d[c+'_xD']=d[c]*d['Dec']
print(f"N={len(d):,} cells, {d['i'].nunique()} countries, {d['ISIC4c'].nunique()} sectors")
print(f"d_pre (pp): mean {d['d_pre'].mean():+.4f} sd {d['d_pre'].std():.4f}\n")

for lbl,fml in [
    ("A. capability -> pre-trend, within sector",
     "d_pre ~ " + " + ".join(CTRL) + " | ISIC4c"),
    ("B. + initial share (mean reversion)",
     "d_pre ~ " + " + ".join(CTRL) + " + s0_pp | ISIC4c"),
    ("C. capability x Dec -> pre-trend",
     "d_pre ~ " + " + ".join(CTRL) + " + " + " + ".join(c+'_xD' for c in CTRL) + " | ISIC4c"),
]:
    m=pf.feols(fml,data=d,weights='w',vcov={'CRV1':'i'})
    t=m.tidy()
    print(f"--- {lbl}   N={int(m._N):,}  R2={m._r2:.4f}")
    for k in t.index:
        r=t.loc[k]
        st='***' if r['Pr(>|t|)']<.01 else '**' if r['Pr(>|t|)']<.05 else '*' if r['Pr(>|t|)']<.10 else ''
        print(f"    {k:16s} {r['Estimate']:+9.4f} ({r['Std. Error']:.4f}){st:3s} p={r['Pr(>|t|)']:.4f}")
    ks=[k for k in t.index if k in CTRL]
    b=t.loc[ks,'Estimate'].values
    idx=[list(t.index).index(k) for k in ks]
    V=m._vcov[np.ix_(idx,idx)]
    from scipy import stats
    W=float(b@np.linalg.pinv(V)@b); F=W/len(ks)
    print(f"    JOINT capability = 0:  F({len(ks)},{d['i'].nunique()-1}) = {F:.2f}   "
          f"p = {1-stats.f.cdf(F,len(ks),d['i'].nunique()-1):.6f}\n")
