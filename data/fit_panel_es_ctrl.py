"""Panel event study with country-level capability controls.

The pair fixed effect a_ik absorbs every country control IN LEVELS, so the objection
that Mexico and Vietnam are not comparable has to be answered here with something that
varies over TIME. Three rungs:

  (1) a_ik + a_kt                       baseline (as fit_panel_es.py)
  (2) + capability_i x year dummies     parametric: countries with a deeper industrial
                                        base are allowed their own path through the
                                        sample, and the IP path is what remains
  (3) a_ik + a_kt + a_it                nonparametric: country-year fixed effects absorb
                                        ALL country-level time-varying heterogeneity,
                                        observed or not -- COVID exposure, exchange
                                        rates, an FTA, anything. Two countries are never
                                        compared; identification is within a country-year
                                        across its sectors, i.e. purely TARGETING.

Rung 3 is the panel analogue of the country fixed effect in the long difference, and it
is strictly stronger than rung 2: (2) removes what we measured, (3) removes what we did
not. IP_ik x 1[t=tau] still varies across k within (i,t), so nothing is collinear.

  s_ikt = sum_tau b1_tau (IP_ik x Dec_k x 1[t=tau]) + sum_tau b2_tau (IP_ik x 1[t=tau])
          + a_ik + a_kt [+ X_i x 1[t=tau]] [+ a_it] + e_ikt        ref tau = REF
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, sys, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
USA, CHINA = 842, 156
REF = int(sys.argv[1]) if len(sys.argv)>1 else 2020
PRE=[2015,2016,2017]
MEAS=['n_policies','frac_policies','share_n_policies','share_frac_policies']
C2ISO={699:356,757:756,579:578,251:250,842:840,381:380,490:158,729:729}
XLSX=("/root/.claude/uploads/618110ef-d894-58be-8e00-e7f2bf548b03/"
      "4ad6510b-0154963EF4289D02DEA410D8D38E4E6A8C6D.xlsx")
CTRL=['log_mva_pc','mva_gdp','mfg_emp','log_exp','ECI']

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
for c in MEAS: raw[c]=pd.to_numeric(raw[c],errors='coerce').fillna(0).astype('float32')
us=raw[raw['j']==USA]
x=us.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum()
tot=x.groupby(level=['ISIC4c','t'],observed=True).transform('sum')
s=(x/tot.replace(0,np.nan)*100).rename('s')
xp=(us[us['t'].isin(PRE)].groupby(['i','ISIC4c'],observed=True)['imports'].sum()/3).rename('w')
chn=s.xs(CHINA,level='i')
dec=(chn[chn.index.get_level_values('t').isin(PRE)].groupby('ISIC4c',observed=True).mean()).rename('Dec')
ip=(raw[raw['t'].isin(PRE)][['i','ISIC4c','t']+MEAS].drop_duplicates(subset=['i','ISIC4c','t'])
    .groupby(['i','ISIC4c'],observed=True)[MEAS].mean())
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
act=raw.loc[raw['imports']>0,['i','ISIC4c']].drop_duplicates()
years=sorted(raw['t'].unique())

# ---- country controls, 2015-17, pre-determined ----
prew=raw[raw['t'].isin(PRE)]
X=(prew.groupby(['i','ISIC4c'],observed=True)['imports'].sum()/len(PRE)).unstack(fill_value=0.0)
del raw, us, prew; gc.collect()
shm=X.div(X.sum(axis=1).replace(0,np.nan),axis=0); world=X.sum(axis=0)/X.sum().sum()
M=((shm.div(world,axis=1))>=1).astype(float); M=M.loc[M.sum(axis=1)>0, M.sum(axis=0)>0]
kc,kp=M.sum(axis=1),M.sum(axis=0)
vals,vecs=np.linalg.eig(((M.div(kc,axis=0))@(M.div(kp,axis=1).T)).values)
K=np.real(vecs[:,np.argsort(-np.real(vals))[1]])
ECI=pd.Series((K-K.mean())/K.std(),index=M.index)
if ECI.corr(pd.Series(kc,index=M.index))<0: ECI=-ECI
u=pd.read_excel(XLSX,sheet_name='Data'); u=u[u['Year'].isin(PRE)]
U=u.pivot_table(index='CountryCode',columns='VariableCode',values='Value',aggfunc='mean')

p=act.loc[act.index.repeat(len(years))].copy(); p['t']=np.tile(years,len(act))
p=p.set_index(['i','ISIC4c','t']).join(s).reset_index()
p=p.merge(xp,on=['i','ISIC4c'],how='left').merge(ip,on=['i','ISIC4c'],how='left')\
   .merge(dec,on='ISIC4c',how='left')
for c in ['s','w','Dec']+MEAS: p[c]=p[c].fillna(0.0)
p['Advanced_i']=p['i'].map(adv); p=p[(p['Advanced_i']==0)&(p['i']!=CHINA)].copy()
p=p[p['w']>0].copy()

ctry=sorted(p['i'].unique()); iso=[C2ISO.get(c,c) for c in ctry]
C=pd.DataFrame(index=pd.Index(ctry,name='i'))
C['log_mva_pc']=np.log(U['NV_IND_MANFPC'].reindex(iso).values)
C['mva_gdp']=U['NV_IND_MANF'].reindex(iso).values
C['mfg_emp']=U['SL_TLF_MANF'].reindex(iso).values
C['log_exp']=np.log(X.sum(axis=1).reindex(ctry).replace(0,np.nan).values)
C['ECI']=ECI.reindex(ctry).values
C=(C-C.mean())/C.std()
ok=C.notna().all(axis=1)
n0,w0=len(p),p['w'].sum()
p=p[p['i'].isin(C.index[ok])].copy()
p=p.merge(C[ok],left_on='i',right_index=True,how='left')
log(f"panel {p.shape} | pairs {p.groupby(['i','ISIC4c'],observed=True).ngroup().nunique():,} "
    f"| {p['i'].nunique()} countries | kept {p['w'].sum()/w0:.3%} of pre-period US value "
    f"({n0-len(p):,} rows dropped for missing controls)")

p['Dz']=(p['Dec']-p['Dec'].mean())/p['Dec'].std()
p['fe_ik']=p.groupby(['i','ISIC4c'],observed=True).ngroup().astype('int32')
p['fe_kt']=p.groupby(['ISIC4c','t'],observed=True).ngroup().astype('int32')
p['fe_it']=p.groupby(['i','t'],observed=True).ngroup().astype('int32')
p['cl_i']=p['i'].astype('int32')
for c in CTRL: p[c]=p[c].astype('float32')
tv=p['t'].values
post=[y for y in years if y>REF]

rows=[]
for M_ in MEAS:
    ipz=(p[M_]/p[M_].std()).values.astype('float32')
    ni,nd=[],[]
    for y in years:
        if y==REF: continue
        p[f"I_{y}"]=(ipz*(tv==y)).astype('float32'); ni.append(f"I_{y}")
        p[f"D_{y}"]=(ipz*p['Dz'].values*(tv==y)).astype('float32'); nd.append(f"D_{y}")
    nc=[]
    for c in CTRL:
        cv=p[c].values
        for y in years:
            if y==REF: continue
            p[f"{c}_{y}"]=(cv*(tv==y)).astype('float32'); nc.append(f"{c}_{y}")
    core=" + ".join(nd+ni)
    SPECS=[("1. a_ik + a_kt",        f"s ~ {core} | fe_ik + fe_kt"),
           ("2. + capability x year",f"s ~ {core} + {' + '.join(nc)} | fe_ik + fe_kt"),
           ("3. + a_it",             f"s ~ {core} | fe_ik + fe_kt + fe_it")]
    print(f"\n{'#'*90}\n# MEASURE {M_}   ref {REF}\n{'#'*90}")
    for name,fml in SPECS:
        fit=pf.feols(fml,data=p,weights="w",vcov={"CRV1":"cl_i"},
                     lean=True,store_data=False,copy_data=False)
        t=fit.tidy()
        print(f"\n--- {name}   N={fit._N:,}")
        print(f"{'year':>6s} {'IP x Dec x year':>27s} {'IP x year':>27s}")
        for y in years:
            if y==REF:
                print(f"{y:>6d} {'- reference -':>27s} {'- reference -':>27s}"); continue
            o=f"{y:>6d}"
            for nm in [f"D_{y}",f"I_{y}"]:
                if nm in t.index:
                    r=t.loc[nm]
                    st='*' if r['Pr(>|t|)']<0.05 else ('.' if r['Pr(>|t|)']<0.10 else ' ')
                    o+=f"  {r['Estimate']:+8.4f} ({r['Std. Error']:.4f}){st}".rjust(28)
                else: o+=f"{'dropped':>28s}"
            print(o)
        rec={'measure':M_,'spec':name,'N':int(fit._N)}
        for lbl,nms in [("IPxDec",nd),("IP",ni)]:
            pr=[n for n in nms if int(n.split('_')[-1])<REF and n in t.index]
            po=[n for n in nms if int(n.split('_')[-1])>REF and n in t.index]
            tp,tq=t.loc[pr],t.loc[po]
            print(f"  {lbl:7s} pre({len(pr)}): mean {tp['Estimate'].mean():+.4f} "
                  f"max|t| {tp['t value'].abs().max():.2f} n(p<.05)={int((tp['Pr(>|t|)']<.05).sum())}"
                  f"  |  post({len(po)}): mean {tq['Estimate'].mean():+.4f} "
                  f"n(p<.05)={int((tq['Pr(>|t|)']<.05).sum())}")
            rec[f'{lbl}_pre_sig']=int((tp['Pr(>|t|)']<.05).sum()); rec[f'{lbl}_pre_n']=len(pr)
            rec[f'{lbl}_post_mean']=tq['Estimate'].mean()
            rec[f'{lbl}_post_sig']=int((tq['Pr(>|t|)']<.05).sum()); rec[f'{lbl}_post_n']=len(po)
        rows.append(rec)
        del fit; gc.collect()
    for c in ni+nd+nc: del p[c]
    gc.collect()
pd.DataFrame(rows).to_csv(f"panel_es_ctrl_{REF}.csv",index=False)
log("ALL DONE")
