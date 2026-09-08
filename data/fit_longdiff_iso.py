"""Recommended long-difference specification, 2017 base, WITH the decoupling interaction.

  d_s_ik = b1 (IP_ik x Dec_k) + b2 IP_ik + g d_s^pre + dl s^pre
           + X_i'th + (X_i x Dec_k)'ps + a_k [+ a_i] + e_ik

b1 is the decoupling term: does the payoff to targeting sector k RISE with how much
China vacated it? Without b1 the equation answers a generic industrial-policy question,
not a decoupling one -- a_k absorbs Dec_k's main effect but IP x Dec varies across i
within k, so the sector FE is no substitute.

X_i x Dec_k must be in alongside it (SS3n): capability x Dec independently predicts
pre-trends, and policy-active countries are more capable, so omitting it loads that
confound onto b1. Including X x Dec while omitting IP x Dec -- which the previous
recommendation did -- is the worst of both.

Dec_k standardised so b1 is "per sd of Chinese share", b2 the effect at mean Dec.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc
USA, CHINA = 842, 156
PRE0, PRE, POST = [2010,2011,2012], [2015,2016,2017], [2022,2023,2024]
C2ISO={699:356,757:756,579:578,251:250,842:840,381:380,490:158,729:729}
XLSX=("/root/.claude/uploads/618110ef-d894-58be-8e00-e7f2bf548b03/"
      "4ad6510b-0154963EF4289D02DEA410D8D38E4E6A8C6D.xlsx")
MEAS=['n_policies','frac_policies','share_n_policies','share_frac_policies']
import sys
CTRL=sys.argv[1].split(',') if len(sys.argv)>1 else ['mva_gdp']
SAMP=sys.argv[2].split(',') if len(sys.argv)>2 else CTRL

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
for c in MEAS: raw[c]=pd.to_numeric(raw[c],errors='coerce').fillna(0).astype('float32')
us=raw[raw['j']==USA]
def share(yrs):
    x=us[us['t'].isin(yrs)].groupby(['i','ISIC4c'],observed=True)['imports'].sum()/len(yrs)
    tot=x.groupby(level='ISIC4c',observed=True).transform('sum')
    return (x/tot.replace(0,np.nan)).rename('s'), x.rename('w')
s0,_=share(PRE0); s1,w1=share(PRE); s2,_=share(POST)
ip=(raw[raw['t'].isin(PRE)][['i','ISIC4c','t']+MEAS].drop_duplicates(subset=['i','ISIC4c','t'])
    .groupby(['i','ISIC4c'],observed=True)[MEAS].mean())
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
act=raw.loc[raw['imports']>0,['i','ISIC4c']].drop_duplicates()
chn=(s1.xs(CHINA,level='i')*100).rename('Dec')          # pp
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

d=act.set_index(['i','ISIC4c']).join([s0.rename('s0'),s1.rename('s1'),s2.rename('s2'),w1,ip]).reset_index()
for c in ['s0','s1','s2','w']+MEAS: d[c]=d[c].fillna(0.0)
d['Advanced_i']=d['i'].map(adv); d=d[(d['Advanced_i']==0)&(d['i']!=CHINA)].copy()
d=d.merge(chn,on='ISIC4c',how='left'); d['Dec']=d['Dec'].fillna(0.0)
d=d[d['w']>0].copy()
d['d_post']=(d['s2']-d['s1'])*100; d['d_pre']=(d['s1']-d['s0'])*100; d['s_pre']=d['s1']*100

ctry=sorted(d['i'].unique()); iso=[C2ISO.get(c,c) for c in ctry]
C=pd.DataFrame(index=pd.Index(ctry,name='i'))
C['log_mva_pc']=np.log(U['NV_IND_MANFPC'].reindex(iso).values)
C['mva_gdp']=U['NV_IND_MANF'].reindex(iso).values
C['mfg_emp']=U['SL_TLF_MANF'].reindex(iso).values
C['log_exp']=np.log(X.sum(axis=1).reindex(ctry).replace(0,np.nan).values)
C['ECI']=ECI.reindex(ctry).values
C=(C-C.mean())/C.std()
w0=d['w'].sum(); d=d.merge(C,left_on='i',right_index=True,how='left').dropna(subset=SAMP)
d['Dz']=(d['Dec']-d['Dec'].mean())/d['Dec'].std()
for c in CTRL: d[c+'_xD']=d[c]*d['Dz']
print(f"N={len(d):,} cells | {d['i'].nunique()} countries | {d['ISIC4c'].nunique()} sectors "
      f"| {d['w'].sum()/w0:.3%} of pre-period US value")
print(f"Dec (pp): mean {d['Dec'].mean():.2f} sd {d['Dec'].std():.2f}")
print(f"d_post (pp): mean {d['d_post'].mean():+.4f} sd {d['d_post'].std():.4f}\n")

lev=" + ".join(CTRL); itx=" + ".join(c+'_xD' for c in CTRL)
rows=[]
for MEASURE in MEAS:
    d['IPz']=d[MEASURE]/d[MEASURE].std(); d['IPxDec']=d['IPz']*d['Dz']
    base="IPxDec + IPz + d_pre + s_pre"
    SPECS=[
      ("1. a_k only",                 f"d_post ~ {base} | ISIC4c"),
      ("2. + capability levels",      f"d_post ~ {base} + {lev} | ISIC4c"),
      ("3. + capability x Dec  [REC]",f"d_post ~ {base} + {lev} + {itx} | ISIC4c"),
      ("4. + country FE",             f"d_post ~ {base} + {itx} | ISIC4c + i"),
    ]
    print(f"{'='*88}\nMEASURE {MEASURE}\n{'='*88}")
    print(f"{'spec':32s}{'IP x Dec':>22s}{'IP':>22s}")
    for name,fml in SPECS:
        m=pf.feols(fml,data=d,weights='w',vcov={'CRV1':'i'})
        o=f"{name:32s}"; rec={'measure':MEASURE,'spec':name,'N':int(m._N)}
        for k in ['IPxDec','IPz']:
            b,se,p=m.coef()[k],m.se()[k],m.pvalue()[k]
            st='***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''
            o+=f"{b:+.4f} ({se:.4f}){st:3s}".rjust(22)
            rec[k]=b; rec[k+'_se']=se; rec[k+'_p']=p
        print(o); rows.append(rec)
    print()
pd.DataFrame(rows).to_csv("longdiff_iso.csv",index=False)
print("saved longdiff_final_"+"_".join(CTRL)+".csv")
