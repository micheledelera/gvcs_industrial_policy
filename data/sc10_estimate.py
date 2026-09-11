"""§5b. Country-sector synthetic control, HIGH vs LOW policy, BETWEEN countries.

Design fixed in §5a, own-country donors now excluded by construction:

  unit        country-sector (i,k), developing ex-China, complete positive US series 2007-17
  treated     HIGH policy = top quartile of the positive distribution
              two measures: n_policies (VOLUME), share_frac_policies (TARGETING)
  donors      LOW policy (bottom quartile or zero), DIFFERENT COUNTRY,
              |Dec - Dec_treated| <= 10pp, |log size - log size_treated| <= 2.0
  trim        K = 20 nearest within that band on country and sector characteristics
              (MVA/GDP, log MVA per capita, ECI, the sector's share of the country's
              exports) -- §5a flagged that a median of 332 donors against 11 pre-treatment
              years reproduces §4d's exact-fit degeneracy, so trimming is mandatory
  match       eleven lagged log US exports plus those four characteristics
  estimator   canonical Abadie (w >= 0, sum w = 1, no intercept) AND ridge-augmented
              (Ben-Michael, Feller & Rothstein) so the extrapolation is visible
  date        2018;  post 2018-2024
  inference   in-space placebo over 500 randomly drawn LOW units, each given its own pool
              built by the identical rule. Aggregate SEs jackknifed over COUNTRIES, since
              a country contributes many treated sectors.
"""
import pandas as pd, numpy as np, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
rng=np.random.default_rng(7)
USA, CHINA = 842, 156
PRE=list(range(2007,2018)); POST=list(range(2018,2025)); YRS=PRE+POST; BASE=[2015,2016,2017]
DECB, SZB, K, NPLAC = 10.0, 2.0, 20, 500
LAMS=np.array([1e-3,1e-2,1e-1,1,10,100])
C2ISO={699:356,757:756,579:578,251:250,842:840,381:380,490:158,729:729}
XLSX=("/root/.claude/uploads/618110ef-d894-58be-8e00-e7f2bf548b03/"
      "4ad6510b-0154963EF4289D02DEA410D8D38E4E6A8C6D.xlsx")
NAME={484:'Mexico',699:'India',704:'Vietnam',458:'Malaysia',764:'Thailand',76:'Brazil',
 360:'Indonesia',608:'Philippines',792:'Turkiye',50:'Bangladesh',616:'Poland',348:'Hungary',
 643:'Russia',152:'Chile',710:'South Africa',604:'Peru',586:'Pakistan',32:'Argentina',
 170:'Colombia',642:'Romania',818:'Egypt',504:'Morocco',100:'Bulgaria',191:'Croatia',
 144:'Sri Lanka',116:'Cambodia',214:'Dominican Rep',188:'Costa Rica',222:'El Salvador',
 218:'Ecuador',68:'Bolivia',524:'Nepal',404:'Kenya',566:'Nigeria',788:'Tunisia',12:'Algeria',
 688:'Serbia',804:'Ukraine',508:'Mozambique',288:'Ghana',682:'Saudi Arabia',784:'UAE',
 634:'Qatar',368:'Iraq'}
def nm(c): return NAME.get(int(c),str(int(c)))

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
for c in ['n_policies','share_frac_policies']:
    raw[c]=pd.to_numeric(raw[c],errors='coerce').fillna(0)
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
us=raw[raw['j']==USA]
X=us.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
X.columns=[int(c) for c in X.columns]; X=X[sorted(X.columns)]
tot_k=us.groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
chn_k=us[us['i']==CHINA].groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
tot_k.columns=[int(c) for c in tot_k.columns]; chn_k.columns=[int(c) for c in chn_k.columns]
DEC=(chn_k[BASE].mean(axis=1)/tot_k[BASE].mean(axis=1)*100)
ip=(raw[raw['t'].isin(BASE)][['i','ISIC4c','t','n_policies','share_frac_policies']]
    .drop_duplicates(subset=['i','ISIC4c','t'])
    .groupby(['i','ISIC4c'],observed=True)[['n_policies','share_frac_policies']].mean())
wexp=(raw[raw['t'].isin(BASE)].groupby(['i','ISIC4c'],observed=True)['imports'].sum()/3)
Xall=wexp.unstack(fill_value=0.0)
del raw,us; gc.collect()
sh=Xall.div(Xall.sum(axis=1).replace(0,np.nan),axis=0); wld=Xall.sum(axis=0)/Xall.sum().sum()
Mm=((sh.div(wld,axis=1))>=1).astype(float); Mm=Mm.loc[Mm.sum(axis=1)>0,Mm.sum(axis=0)>0]
kc,kp=Mm.sum(axis=1),Mm.sum(axis=0)
vals,vecs=np.linalg.eig(((Mm.div(kc,axis=0))@(Mm.div(kp,axis=1).T)).values)
Kv=np.real(vecs[:,np.argsort(-np.real(vals))[1]])
ECI=pd.Series((Kv-Kv.mean())/Kv.std(),index=Mm.index)
if ECI.corr(pd.Series(kc,index=Mm.index))<0: ECI=-ECI
u=pd.read_excel(XLSX,sheet_name='Data'); u=u[u['Year'].isin(BASE)]
U=u.pivot_table(index='CountryCode',columns='VariableCode',values='Value',aggfunc='mean')

X=X.loc[[(i,k) for i,k in X.index if adv.get(i,1)==0 and i!=CHINA]]
# §5b BUGFIX. The first run required a complete positive series over PRE only. 356 units
# (6.4%) then carried a missing or zero POST year, giving NaN tau and NaN post/pre ratio.
# In the placebo comparison `ratio_placebo >= ratio_treated` a NaN returns False, so those
# placebos never counted as exceedances and EVERY p-value was biased downward -- the source
# of the spurious "33-37% below p<0.10". Require a complete positive series over PRE AND
# POST. Note this selects on survival: a country-sector that stopped exporting to the US is
# dropped, and that is informative. The drop is reported by treatment status below.
n_pre_only=int(((X[PRE]>0).all(axis=1)).sum())
X=X[(X[YRS]>0).all(axis=1) & X[YRS].notna().all(axis=1)]
print(f"complete PRE only: {n_pre_only:,}  ->  complete PRE and POST: {len(X):,} "
      f"({n_pre_only-len(X):,} dropped, {100*(n_pre_only-len(X))/n_pre_only:.1f}%)")
L=np.log(X[YRS])
M=pd.DataFrame(index=X.index)
M['vol']=ip['n_policies'].reindex(X.index).fillna(0.0).values
M['tgt']=ip['share_frac_policies'].reindex(X.index).fillna(0.0).values
M['Dec']=[DEC.get(k,np.nan) for _,k in X.index]
M['lsize']=L[BASE].mean(axis=1).values
iso=[C2ISO.get(i,i) for i,_ in X.index]
M['mva_gdp']=U['NV_IND_MANF'].reindex(iso).values
M['lmva']=np.log(U['NV_IND_MANFPC'].reindex(iso).values)
M['eci']=[ECI.get(i,np.nan) for i,_ in X.index]
M['spec']=[(Xall.loc[i,k]/Xall.loc[i].sum() if Xall.loc[i].sum()>0 else np.nan)
           for i,k in X.index]
M=M.dropna(); L=L.loc[M.index]
COV=['mva_gdp','lmva','eci','spec']
Z=((M[COV]-M[COV].mean())/M[COV].std()).values
idx=list(M.index); CT=np.array([a for a,_ in idx]); SE=np.array([b for _,b in idx])
Dv,Sv=M['Dec'].values,M['lsize'].values
Lp,Lq=L[PRE].values,L[POST].values
log(f"units {len(M):,} | {len(set(CT))} countries | {len(set(SE))} sectors")

def proj(v):
    w=np.sort(v)[::-1]; c=np.cumsum(w)-1.0; r=np.arange(1,len(v)+1)
    m=w-c/r>0
    if not m.any(): return np.ones_like(v)/len(v)
    return np.maximum(v-c[m][-1]/r[m][-1],0.0)
def sc(A,b,it=3000):
    n=A.shape[0]; w=np.ones(n)/n; Lc=np.linalg.norm(A,2)**2/A.shape[1]+1e-12
    for _ in range(it): w=proj(w-(A@(A.T@w-b)*2.0/A.shape[1])/Lc)
    return w
def reta(Xc,y,lam): return np.linalg.solve(Xc@Xc.T+lam*np.eye(Xc.shape[0]),Xc@y)
def plam(Xc,y):
    n=Xc.shape[1]; best=(np.inf,LAMS[0])
    for lam in LAMS:
        e=sum((y[j]-Xc[:,j]@reta(Xc[:,np.arange(n)!=j],y[np.arange(n)!=j],lam))**2
              for j in range(n))
        if e<best[0]: best=(e,lam)
    return best[1]

def one(a, lopool):
    m=(np.abs(Dv[lopool]-Dv[a])<=DECB)&(np.abs(Sv[lopool]-Sv[a])<=SZB)&(CT[lopool]!=CT[a])
    cand=lopool[m]
    if len(cand)<8: return None
    d=np.sqrt(((Z[cand]-Z[a])**2).sum(axis=1))
    pool=cand[np.argsort(d)[:K]]
    A=np.hstack([Lp[pool], Z[pool]]); b=np.concatenate([Lp[a], Z[a]])
    w=sc(A,b)
    pre=float(np.sqrt(((Lp[a]-Lp[pool].T@w)**2).mean()))
    post=float(np.sqrt(((Lq[a]-Lq[pool].T@w)**2).mean()))
    yq=Lq[pool].mean(axis=1); Xp=Lp[pool].T
    Xc=Xp-Xp.mean(axis=1,keepdims=True)
    lam=plam(Xc,yq-yq.mean()); eta=reta(Xc,yq-yq.mean(),lam)
    ts=float(Lq[a].mean()-yq@w); corr=float((Lp[a]-Xp@w)@eta)
    gpre=float((Lp[a]-Lp[pool].T@w).mean())
    return dict(pre=pre,post=post,ratio=post/max(pre,1e-8),tau_scm=ts,tau_aug=ts-corr,
                corr=corr,gap_pre=gpre,n_pool=len(pool),n_cand=len(cand))

rows=[]
for meas,lab in [('vol','VOLUME (n_policies)'),('tgt','TARGETING (share_frac_policies)')]:
    v=M[meas].values; pos=v[v>0]
    q75,q25=np.quantile(pos,.75),np.quantile(pos,.25)
    hi=np.where(v>=q75)[0]; lo=np.where(v<=q25)[0]
    print(f"  {lab}: HIGH {len(hi)}, LOW {len(lo)}")
    pl=rng.choice(lo,size=min(NPLAC,len(lo)),replace=False)
    P=[]
    for a in pl:
        r=one(a,lo[lo!=a])
        if r: P.append(r)
    pr=np.array([x['ratio'] for x in P]); pa=np.abs([x['tau_aug'] for x in P])
    ppre=np.array([x['pre'] for x in P])
    log(f"{lab}: {len(P)} placebos, median pre-RMSPE {np.median(ppre):.3f}, "
        f"median ratio {np.median(pr):.2f}")
    T=[]
    for a in hi:
        r=one(a,lo)
        if not r: continue
        r.update(i=CT[a],k=SE[a],name=nm(CT[a]),meas=meas,
                 p_scm=float((1+(pr>=r['ratio']).sum())/(1+len(pr))),
                 p_aug=float((1+(pa>=abs(r['tau_aug'])).sum())/(1+len(pa))),
                 fit_x=r['pre']/np.median(ppre))
        T.append(r)
    R=pd.DataFrame(T); R['credible']=R['fit_x']<=2.0
    rows.append(R)
    from scipy import stats
    def jkc(df,col):
        cs=df['i'].unique(); m=df[col].mean()
        a=np.array([df[df['i']!=c][col].mean() for c in cs])
        return m, np.sqrt((len(cs)-1)/len(cs)*((a-a.mean())**2).sum())
    print(f"\n{'='*104}\n{lab}   treated {len(R)}   {R['i'].nunique()} countries   "
          f"median donors available {R['n_cand'].median():.0f}\n{'='*104}")
    for slab,sub in [("ALL",R),("CREDIBLE FIT",R[R['credible']])]:
        if len(sub)<10: continue
        a=jkc(sub,'tau_scm'); b=jkc(sub,'tau_aug'); g=jkc(sub,'gap_pre')
        fs=-2*np.log(sub['p_scm'].clip(1e-6)).sum(); fa=-2*np.log(sub['p_aug'].clip(1e-6)).sum()
        n=len(sub)
        print(f"  {slab} (n={n}, {sub['i'].nunique()} countries)  median pre-RMSPE {sub['pre'].median():.3f}")
        print(f"    gap pre    {g[0]:>+8.4f} ({g[1]:.4f}) t={g[0]/g[1]:>+5.2f}")
        print(f"    tau classic{a[0]:>+8.4f} ({a[1]:.4f}) t={a[0]/a[1]:>+5.2f}   "
              f"median p {sub['p_scm'].median():.3f}  p<.10 {int((sub['p_scm']<.10).sum())}/{n}"
              f" ({100*(sub['p_scm']<.10).mean():.0f}%)  Fisher p={1-stats.chi2.cdf(fs,2*n):.4f}")
        print(f"    tau AUG    {b[0]:>+8.4f} ({b[1]:.4f}) t={b[0]/b[1]:>+5.2f}   "
              f"median p {sub['p_aug'].median():.3f}  p<.10 {int((sub['p_aug']<.10).sum())}/{n}"
              f" ({100*(sub['p_aug']<.10).mean():.0f}%)  Fisher p={1-stats.chi2.cdf(fa,2*n):.4f}")
        print(f"    extrapolation: median |correction| {sub['corr'].abs().median():.3f}")
    print()
pd.concat(rows).to_csv("sc10_estimate.csv",index=False)
log("DONE")
