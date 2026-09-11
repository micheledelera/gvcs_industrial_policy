"""Does §3z's +0.29 survive forcing the synthetic control to match MVA?

§3z variant A left MVA_W = 2.0, an arbitrary weight on the MVA target row. If it is too
low, omega is free to build a synthetic control out of countries at a different level of
industrialisation, and +0.29 is capability rather than policy -- exactly the §3n confound.

Sweep MVA_W from 0 (no MVA matching at all, i.e. §3x's setup) upward, at zeta -> 0 where
the path match is tightest. At each rung report:

  MVA gap    treated mean MVA minus omega-weighted synthetic MVA, in sd units.
             Should fall toward zero as MVA_W rises.
  pre RMSE   quality of the pre-2018 log-export path match. Should WORSEN as MVA_W
             rises, since omega is spending its budget on MVA instead.
  TRUE tau   the estimate.
  PLACEBO    the same estimator on (2015-17) - (2010-12), entirely pre-treatment.

Verdict rule set in advance: if TRUE tau holds near +0.29 while the MVA gap goes to zero,
the effect is not capability. If it decays toward the §3z variant-B value (+0.157, ns),
it was.
"""
import pandas as pd, numpy as np, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("fit_sdid.py").read().split("def sdid_sector")[0]
     .replace('MEASURE = sys.argv[1] if len(sys.argv)>1 else','MEASURE ='))
C2ISO={699:356,757:756,579:578,251:250,842:840,381:380,490:158,729:729}
XLSX=("/root/.claude/uploads/618110ef-d894-58be-8e00-e7f2bf548b03/"
      "4ad6510b-0154963EF4289D02DEA410D8D38E4E6A8C6D.xlsx")
PATH=[y for y in years if 2010<=y<=2017]
PRE0=[2010,2011,2012]; PRE=[2015,2016,2017]; POST=[2022,2023,2024]
MIN_DON=5; ZS=1e-6

u=pd.read_excel(XLSX,sheet_name='Data'); u=u[u['Year'].isin(PRE)]
U=u.pivot_table(index='CountryCode',columns='VariableCode',values='Value',aggfunc='mean')
ctry=sorted({i for i,_ in Y.index})
mva=pd.Series(U['NV_IND_MANF'].reindex([C2ISO.get(c,c) for c in ctry]).values,index=ctry)
m=meta.copy(); m['mva']=[mva.get(i,np.nan) for i,_ in m.index]
m=m.dropna(subset=['mva']); Yv=Y.loc[m.index]
m['mvaz']=(m['mva']-m['mva'].mean())/m['mva'].std()
m['d_post']=Yv[POST].mean(axis=1)-Yv[PRE].mean(axis=1)
m['d_plac']=Yv[PRE].mean(axis=1)-Yv[PRE0].mean(axis=1)
log(f"pairs {len(m):,} | treated {int(m['T'].sum()):,}")

# benchmark: raw treated-vs-donor MVA gap, unweighted, within sector
raw_gap=[]
for k,g in m.groupby(level='ISIC4c', observed=True):
    t_=g[g['T']==1]; d_=g[g['T']==0]
    if len(t_)==0 or len(d_)<MIN_DON: continue
    raw_gap.append(t_['mvaz'].mean()-d_['mvaz'].mean())
print(f"\nBENCHMARK  raw treated-minus-donor MVA gap, unweighted, within sector: "
      f"mean {np.mean(raw_gap):+.4f} sd units (median {np.median(raw_gap):+.4f})")

def solve_w2(A, b, zeta2, npath, iters=800):
    """As solve_w, but the free intercept is fitted over the FIRST npath rows only.

    The path rows are log exports and the trailing row is MVA. A single shared intercept
    lets a level shift in log exports offset an MVA gap, which is meaningless and made
    the MVA-gap column non-monotonic in the first version of this sweep. Here the
    intercept absorbs only the log-export level, and the MVA row must be matched on its
    own terms."""
    n=A.shape[0]; w=np.ones(n)/n
    L=np.linalg.norm(A,2)**2/max(A.shape[1],1)+zeta2+1e-12
    step=1.0/L
    for _ in range(iters):
        f=A.T@w
        w0=np.mean(b[:npath]-f[:npath])
        r=f-b; r[:npath]+=w0
        g=A@r*2.0/max(A.shape[1],1)+2.0*zeta2*w
        w=proj_simplex(w-step*g)
    f=A.T@w
    return w, float(np.mean(b[:npath]-f[:npath]))

def run(mw):
    out=[]
    for k,g in m.groupby(level='ISIC4c', observed=True):
        t_=g[g['T']==1]; d_=g[g['T']==0]
        if len(t_)==0 or len(d_)<MIN_DON: continue
        P=Yv.loc[d_.index,PATH].values; bt=Yv.loc[t_.index,PATH].values.mean(axis=0)
        A=P.copy(); b=bt.copy()
        if mw>0:
            A=np.hstack([A,(mw*d_['mvaz'].values)[:,None]]); b=np.append(b, mw*t_['mvaz'].mean())
        sig=np.diff(P,axis=1).std()
        zeta2=(((len(t_)*len(POST))**0.5)*(sig**2))*ZS+1e-10
        w,_=solve_w2(A,b,zeta2,len(PATH))
        fit=P.T@w; f0=np.mean(bt-fit)
        out.append({'k':k,'tau':t_['d_post'].mean()-float(w@d_['d_post'].values),
                    'plac':t_['d_plac'].mean()-float(w@d_['d_plac'].values),
                    'mgap':t_['mvaz'].mean()-float(w@d_['mvaz'].values),
                    'rmse':np.sqrt(((fit+f0-bt)**2).mean()),
                    'w_eff':1.0/(w**2).sum()})
    return pd.DataFrame(out)

def jk(o,col):
    n=len(o); mu=o[col].mean()
    a=np.array([o[col].drop(i).mean() for i in o.index])
    se=np.sqrt((n-1)/n*((a-a.mean())**2).sum())
    return mu, se, (mu/se if se>0 else np.nan)

print(f"\n{'='*100}")
print(f"MVA_W SWEEP at zeta -> 0   ({'MVA gap' } in sd units; 0 = synthetic matches treated MVA exactly)")
print(f"{'='*100}")
print(f"{'MVA_W':>7s}{'cells':>7s}{'eff.don':>9s}{'MVA gap':>10s}{'pre RMSE':>10s}"
      f"{'TRUE tau':>24s}{'PLACEBO tau':>24s}")
rows=[]
for mw in [0.0,0.5,1.0,2.0,5.0,10.0,25.0,100.0]:
    o=run(mw)
    if len(o)<3: print(f"{mw:>7.1f}   too few cells"); continue
    a=jk(o,'tau'); b=jk(o,'plac')
    f=lambda r:f"{r[0]:+.4f} ({r[1]:.4f}) t={r[2]:+.2f}"
    print(f"{mw:>7.1f}{len(o):>7d}{o['w_eff'].median():>9.1f}{o['mgap'].mean():>+10.4f}"
          f"{o['rmse'].median():>10.4f}{f(a):>24s}{f(b):>24s}")
    rows.append({'MVA_W':mw,'cells':len(o),'eff_don':o['w_eff'].median(),
                 'mva_gap':o['mgap'].mean(),'pre_rmse':o['rmse'].median(),
                 'tau':a[0],'se':a[1],'t':a[2],'plac':b[0],'plac_se':b[1],'plac_t':b[2]})
pd.DataFrame(rows).to_csv("sc_mvasweep_fixed.csv",index=False)
log("DONE")
