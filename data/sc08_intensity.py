"""Does the §4h null survive a treatment defined by INTENSITY of policy use?

Routes A and B defined treated as "any recorded GTA intervention", which puts India
(targeting 60 of 62 sectors) in the same box as a country with one recorded measure, and
§3ac found the median treated share_frac_policies is 0.00007. A null against that
definition says little. Canonical SC needs a binary split, so the fix is to threshold on
intensity and drop the middle.

Country-level intensity, 2015-2017:
  n_pol        total interventions
  n_sec        sectors with any policy
  pol_per_bn   interventions per $bn of manufacturing exports  <- size-normalised, and
               the one that matters, since raw counts scale with the economy and size is
               exactly what breaks the convex hull (§4a)

Two comparisons:
  (i)  HIGH-vs-ZERO   treated = top tercile of intensity among users; donors = the 130
       countries with no recorded policy at all. Sharpens the §4e/§4f/§4g contrast.
  (ii) HIGH-vs-LOW    treated = top tercile; donors = bottom tercile of USERS. Holds
       "is a policy user" fixed and varies only the dose -- the cleanest test of whether
       intensity matters, and immune to the policy-user-vs-non-user confound of §3ag.

Machinery as §4e-§4g: covariate-ranked donor pool (K nearest), matching on eleven lagged
outcomes plus covariates, non-negative weights summing to one, no intercept, plus the
ridge-augmented estimate, plus Abadie in-space placebo inference.
"""
import pandas as pd, numpy as np, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc05_trimmed.py").read().split("print(f\"\\n{'='*104}\\nDONOR-POOL SIZE SWEEP")[0])
K=20; BASE=[2015,2016,2017]
LAMS=np.array([1e-4,1e-3,1e-2,1e-1,1,10,100,1000])

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
raw['n_policies']=pd.to_numeric(raw['n_policies'],errors='coerce').fillna(0)
p3=raw[raw['t'].isin(BASE)][['i','ISIC4c','t','n_policies']].drop_duplicates(subset=['i','ISIC4c','t'])
INT=p3.groupby('i').agg(n_pol=('n_policies','sum'))
INT['n_sec']=p3[p3['n_policies']>0].groupby('i')['ISIC4c'].nunique()
INT['n_sec']=INT['n_sec'].fillna(0)
us2=raw[raw['j']==USA]
tot_k=us2.groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
chn_k=us2[us2['i']==CHINA].groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
tot_k.columns=[int(c) for c in tot_k.columns]; chn_k.columns=[int(c) for c in chn_k.columns]
DEC=set((chn_k[BASE].mean(axis=1)/tot_k[BASE].mean(axis=1)*100).pipe(lambda s:s[s>=25]).index)
usdec=us2[us2['ISIC4c'].isin(DEC)].groupby(['i','t'],observed=True)['imports'].sum().unstack('t')
usdec.columns=[int(c) for c in usdec.columns]
wexp=(raw[raw['t'].isin(BASE)].groupby('i')['imports'].sum()/3)
del raw,us2,p3; gc.collect()
INT['exp_bn']=wexp/1e6
INT['pol_per_bn']=INT['n_pol']/INT['exp_bn'].replace(0,np.nan)

I2=INT.reindex(Y.index)
users=[i for i in TR if I2.loc[i,'n_pol']>0]
zeros=DN
print(f"policy users {len(users)}   zero-policy donors {len(zeros)}\n")
print("INTENSITY AMONG USERS (2015-17)")
for c in ['n_pol','n_sec','pol_per_bn']:
    q=I2.loc[users,c].quantile([0,.33,.5,.67,1])
    print(f"  {c:<12s} min {q[0]:>8.2f}  p33 {q[.33]:>8.2f}  med {q[.5]:>8.2f}"
          f"  p67 {q[.67]:>8.2f}  max {q[1]:>10.2f}")
print()
for c in ['n_pol','pol_per_bn']:
    top=I2.loc[users,c].nlargest(13)
    print(f"  top 13 by {c}: "+", ".join(f"{nm(i)}" for i in top.index))
print()

OUT={'C. log US exports, decoupling sectors':np.log(usdec.reindex(columns=YRS)),
     'D. log US orientation':Y}

def ridge_eta(Xc,y,lam):
    return np.linalg.solve(Xc@Xc.T+lam*np.eye(Xc.shape[0]), Xc@y)
def pick_lam(Xc,y):
    n=Xc.shape[1]; best=(np.inf,LAMS[0])
    for lam in LAMS:
        e=sum((y[j]-Xc[:,j]@ridge_eta(Xc[:,np.arange(n)!=j],y[np.arange(n)!=j],lam))**2
              for j in range(n))
        if e<best[0]: best=(e,lam)
    return best[1]

def run(lab, Yraw, treated, donors, tag):
    Yo=Yraw.reindex(Y.index)
    g=Yo.notna().all(axis=1)&np.isfinite(Yo).all(axis=1)
    Yo=Yo[g]
    tr=[i for i in treated if i in Yo.index]; dn=[i for i in donors if i in Yo.index]
    kk=min(K,len(dn)-1)
    if len(tr)<5 or len(dn)<8: print(f"  {tag}: too few (tr {len(tr)}, dn {len(dn)})"); return None
    def one(i,pool):
        Xp=Yo.loc[pool,PRE].values.T; x1=Yo.loc[i,PRE].values
        A=np.hstack([Yo.loc[pool,PRE].values, COV_W*Cz.loc[pool].values])
        b=np.concatenate([x1, COV_W*Cz.loc[i].values])
        w=sc(A,b); pre_=rms(x1,Xp@w)
        yp=Yo.loc[pool,POST].values.mean(axis=1)
        Xc=Xp-Xp.mean(axis=1,keepdims=True)
        lam=pick_lam(Xc,yp-yp.mean()); eta=ridge_eta(Xc,yp-yp.mean(),lam)
        ts=float(Yo.loc[i,POST].mean()-yp@w); corr=float((x1-Xp@w)@eta)
        return pre_, ts, ts-corr, corr
    pls,pla,plp=[],[],[]
    for d in dn:
        pool=nearest(d,[x for x in dn if x!=d],kk)
        pr,ts,ta,_=one(d,pool); pls.append(abs(ts)); pla.append(abs(ta)); plp.append(pr)
    pls,pla,plp=np.array(pls),np.array(pla),np.array(plp)
    rows=[]
    for i in tr:
        pool=nearest(i,dn,kk)
        pr,ts,ta,c_=one(i,pool)
        rows.append({'name':nm(i),'pre':pr,'fit_x':pr/np.median(plp),'tau_scm':ts,
                     'tau_aug':ta,'corr':c_,
                     'p_scm':float((1+(pls>=abs(ts)).sum())/(1+len(pls))),
                     'p_aug':float((1+(pla>=abs(ta)).sum())/(1+len(pla)))})
    R=pd.DataFrame(rows); R['credible']=R['fit_x']<=2.0
    def jk(v):
        v=np.asarray(v); n=len(v)
        if n<3: return np.nan,np.nan
        a=np.array([np.delete(v,j).mean() for j in range(n)])
        return v.mean(), np.sqrt((n-1)/n*((a-a.mean())**2).sum())
    from scipy import stats
    a=jk(R['tau_scm'].values); b=jk(R['tau_aug'].values)
    fa=-2*np.log(R['p_aug'].clip(1e-6)).sum()
    fs=-2*np.log(R['p_scm'].clip(1e-6)).sum()
    print(f"  {tag:<34s} tr {len(tr):>2d} dn {len(dn):>3d} | "
          f"classic {a[0]:>+7.3f}(t{a[0]/a[1]:>+5.2f}) medp {R['p_scm'].median():.3f} "
          f"Fp {1-stats.chi2.cdf(fs,2*len(R)):.3f} | "
          f"AUG {b[0]:>+7.3f}(t{b[0]/b[1]:>+5.2f}) medp {R['p_aug'].median():.3f} "
          f"Fp {1-stats.chi2.cdf(fa,2*len(R)):.3f} | "
          f"p<.10 {int((R['p_aug']<.10).sum())}/{len(R)}")
    R['spec']=f"{lab} | {tag}"
    return R

allR=[]
for lab,Yv_ in OUT.items():
    print(f"{'='*140}\n{lab}\n{'='*140}")
    for icol in ['n_pol','pol_per_bn']:
        s=I2.loc[users,icol].sort_values()
        n=len(s); lo=list(s.index[:n//3]); hi=list(s.index[-(n//3):])
        allR.append(run(lab,Yv_,hi,zeros,f"(i)  HIGH {icol} vs ZERO"))
        allR.append(run(lab,Yv_,hi,lo,   f"(ii) HIGH {icol} vs LOW users"))
    print()
allR=[r for r in allR if r is not None]
pd.concat(allR).to_csv("sc08_intensity.csv",index=False)
log("DONE")
