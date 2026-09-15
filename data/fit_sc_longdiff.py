"""Synthetic controls paired with the LONG DIFFERENCE, not an event study.

Answers two objections to §3x. (i) The synthetic controls there were built purely on the
pre-treatment outcome path -- MVA never entered. (ii) SC needs a treatment DATE, not
staggered timing, so the event study was optional; a single common pre/post split removes
the post-window-length artefact that made §3x's cohort table meaningless.

§3y (diag_overlap.py) established that common support is fine -- treated pairs sit at
mean percentile 0.499 of their sector x MVA-decile donor distribution -- so the §3x
failure was the regularisation, not the data. Hence the zeta ladder below.

  outcome   d_ik = mean log X_ik(2022-24) - mean log X_ik(2015-17)      US exports
  treated   IP_ik > 0
  donors    same sector, IP = 0
  weights   omega matching the 2010-2017 log-export path, plus MVA
  estimate  tau_ik = d_treated - d_synthetic, aggregated over sectors
  placebo   the same on d = (2015-17) - (2010-12), entirely pre-treatment

Two matching variants:
  A  soft   donors same sector; MVA enters the matching objective as an extra target row
  B  hard   donors same sector AND same MVA decile; path matching only

zeta ladder: the Arkhangelsky et al. value, then /4, /16, and effectively zero. A real
synthetic match should show the placebo shrinking toward zero as zeta falls, while the
true-event estimate holds. If both move together, there is nothing there.
"""
import pandas as pd, numpy as np, gc, sys, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("fit_sdid.py").read().split("def sdid_sector")[0]
     .replace('MEASURE = sys.argv[1] if len(sys.argv)>1 else','MEASURE ='))
C2ISO={699:356,757:756,579:578,251:250,842:840,381:380,490:158,729:729}
XLSX=("/root/.claude/uploads/618110ef-d894-58be-8e00-e7f2bf548b03/"
      "4ad6510b-0154963EF4289D02DEA410D8D38E4E6A8C6D.xlsx")
PATH=[y for y in years if 2010<=y<=2017]
PRE0=[2010,2011,2012]; PRE=[2015,2016,2017]; POST=[2022,2023,2024]
MIN_DON=5

u=pd.read_excel(XLSX,sheet_name='Data'); u=u[u['Year'].isin(PRE)]
U=u.pivot_table(index='CountryCode',columns='VariableCode',values='Value',aggfunc='mean')
ctry=sorted({i for i,_ in Y.index})
mva=pd.Series(U['NV_IND_MANF'].reindex([C2ISO.get(c,c) for c in ctry]).values,index=ctry)
m=meta.copy()
m['mva']=[mva.get(i,np.nan) for i,_ in m.index]
m=m.dropna(subset=['mva'])
Yv=Y.loc[m.index]
m['mvaz']=(m['mva']-m['mva'].mean())/m['mva'].std()
m['mva_dec']=pd.qcut(m['mva'].rank(method='first'),10,labels=False)
m['d_post']=Yv[POST].mean(axis=1)-Yv[PRE].mean(axis=1)
m['d_plac']=Yv[PRE].mean(axis=1)-Yv[PRE0].mean(axis=1)
log(f"pairs {len(m):,} | treated {int(m['T'].sum()):,} | "
    f"sectors {m.index.get_level_values('ISIC4c').nunique()}")

MVA_W = 2.0     # weight on the MVA target row, in units of the path's own scale
def taus(variant, zscale, outcome):
    out=[]
    grp = ['ISIC4c','mva_dec'] if variant=='B' else ['ISIC4c']
    for key,g in m.groupby(grp, observed=True):
        t_=g[g['T']==1]; d_=g[g['T']==0]
        if len(t_)==0 or len(d_)<MIN_DON: continue
        Ad=Yv.loc[d_.index,PATH].values
        bt=Yv.loc[t_.index,PATH].values.mean(axis=0)
        if variant=='A':
            Ad=np.hstack([Ad, (MVA_W*d_['mvaz'].values)[:,None]])
            bt=np.append(bt, MVA_W*t_['mvaz'].mean())
        sig=np.diff(Yv.loc[d_.index,PATH].values,axis=1).std()
        zeta2=(((len(t_)*len(POST))**0.5)*(sig**2))*zscale + 1e-10
        w,_=solve_w(Ad,bt,zeta2)
        tau=t_[outcome].mean()-float(w@d_[outcome].values)
        # quality of the pre-path match this omega achieves
        fit=Yv.loc[d_.index,PATH].values.T@w
        f0=np.mean(Yv.loc[t_.index,PATH].values.mean(axis=0)-fit)
        rmse=np.sqrt(((fit+f0-Yv.loc[t_.index,PATH].values.mean(axis=0))**2).mean())
        out.append({'key':key,'tau':tau,'n_tr':len(t_),'n_don':len(d_),
                    'rmse':rmse,'w_eff':1.0/(w**2).sum(),
                    'w_tr':m.loc[t_.index,'wpre'].sum()})
    return pd.DataFrame(out)

from scipy import stats
def summarise(o):
    if o.empty or len(o)<3: return None
    n=len(o); mu=o['tau'].mean()
    jk=np.array([o['tau'].drop(i).mean() for i in o.index])
    se=np.sqrt((n-1)/n*((jk-jk.mean())**2).sum())
    return mu, se, mu/se if se>0 else np.nan, n, o['rmse'].median(), o['w_eff'].median()

rows=[]
for variant,vlab in [('A','A soft: same sector, MVA in objective'),
                     ('B','B hard: same sector AND same MVA decile')]:
    print(f"\n{'='*94}\n{vlab}\n{'='*94}")
    print(f"{'zeta':>10s}{'cells':>7s}{'eff.don':>9s}{'pre RMSE':>10s}"
          f"{'TRUE tau':>22s}{'PLACEBO tau':>22s}")
    for zlab,zs in [('paper',1.0),('/4',0.25),('/16',0.0625),('~0',1e-6)]:
        a=summarise(taus(variant,zs,'d_post'))
        b=summarise(taus(variant,zs,'d_plac'))
        if a is None or b is None: print(f"{zlab:>10s}   too few cells"); continue
        f=lambda r: f"{r[0]:+.4f} ({r[1]:.4f}) t={r[2]:+.2f}"
        print(f"{zlab:>10s}{a[3]:>7d}{a[5]:>9.1f}{a[4]:>10.4f}{f(a):>22s}{f(b):>22s}")
        rows.append({'variant':variant,'zeta':zlab,'cells':a[3],'eff_don':a[5],
                     'pre_rmse':a[4],'tau':a[0],'se':a[1],'t':a[2],
                     'plac':b[0],'plac_se':b[1],'plac_t':b[2]})
pd.DataFrame(rows).to_csv("sc_longdiff.csv",index=False)
log("DONE")
