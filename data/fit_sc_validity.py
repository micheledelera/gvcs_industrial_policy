"""Testing Cunningham's claim: does a long, close pre-treatment fit validate the
synthetic control as a counterfactual?

"If the treated unit and its synthetic counterpart closely track each other over a long
stretch of time before treatment, it suggests that both observed and unobserved factors
are likely balanced." That is an argument, and it has testable content. Four tests.

  A. HELD-OUT PRE-PERIOD.  The claim is that tracking BEFORE treatment predicts what
     would have happened AFTER. Test the predictive half of that inside the pre-period,
     where the counterfactual is observed: fit omega on 2010-2014 only, then measure how
     well it tracks 2015-2017, which it never saw. If out-of-sample tracking collapses,
     "it tracked closely, so it is valid" does not hold in this data.

  B. FIT vs PLACEBO.  If close fit means a valid counterfactual, sectors where omega
     fits tightly should show placebo effects near zero, and badly-fitting sectors large
     ones. Correlate pre-period RMSE with |placebo tau| across sectors.

  C. RESTRICT TO GOOD FITS.  Abadie's own advice is to discard units whose pre-fit is
     poor. Re-estimate on the best-fitting tercile and half.

  D. IN-SPACE PLACEBO / RMSPE RATIO (Abadie, Diamond & Hainmueller).  The standard SC
     inference, which this project has not yet run. For each sector, form placebo
     "treated" groups of the SAME SIZE as the real one by sampling donors, build each a
     synthetic control from the remaining donors, and compute post/pre RMSPE. The
     p-value is the rank of the true group's ratio among the placebos. Sampling
     same-size groups rather than single donors keeps the comparison fair: a lone donor
     is noisier than a 16-unit mean and would inflate its own ratio.
"""
import pandas as pd, numpy as np, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("fit_sc_mvasweep2.py").read().split("def run(mw):")[0])
rng=np.random.default_rng(11)
MW=2.0; ZS=1e-6; NPLAC=50
FIT5=[y for y in PATH if y<=2014]; HOLD=[y for y in PATH if y>2014]

def omega(d_idx, t_mvaz, tgt_path, path_years):
    P=Yv.loc[d_idx,path_years].values
    A=np.hstack([P,(MW*m.loc[d_idx,'mvaz'].values)[:,None]])
    b=np.append(tgt_path, MW*t_mvaz)
    sig=np.diff(P,axis=1).std()
    z2=(((len(d_idx)*len(POST))**0.5)*(sig**2))*ZS+1e-10
    w,_=solve_w2(A,b,z2,len(path_years))
    return w

def rmse_of(w, d_idx, tgt, yrs):
    f=Yv.loc[d_idx,yrs].values.T@w
    return float(np.sqrt(((f+np.mean(tgt-f)-tgt)**2).mean()))

rows=[]
for k,g in m.groupby(level='ISIC4c', observed=True):
    t_=g[g['T']==1]; d_=g[g['T']==0]
    if len(t_)==0 or len(d_)<MIN_DON: continue
    ti,di=t_.index,d_.index
    full=Yv.loc[ti,PATH].values.mean(axis=0)
    # A. held-out
    w5=omega(di, t_['mvaz'].mean(), Yv.loc[ti,FIT5].values.mean(axis=0), FIT5)
    r_in =rmse_of(w5,di,Yv.loc[ti,FIT5].values.mean(axis=0),FIT5)
    r_out=rmse_of(w5,di,Yv.loc[ti,HOLD].values.mean(axis=0),HOLD)
    # full-path omega: the estimator actually used
    wf=omega(di, t_['mvaz'].mean(), full, PATH)
    pre_rmse=rmse_of(wf,di,full,PATH)
    dp=(Yv[POST].mean(axis=1)-Yv[PRE].mean(axis=1))
    dl=(Yv[PRE].mean(axis=1)-Yv[PRE0].mean(axis=1))
    tau =dp[ti].mean()-float(wf@dp[di].values)
    plac=dl[ti].mean()-float(wf@dl[di].values)
    post_rmspe=abs(tau); ratio=post_rmspe/max(pre_rmse,1e-6)
    # D. placebo groups of the same size
    pr=[]
    if len(di)>len(ti)+MIN_DON:
        for _ in range(NPLAC):
            pick=rng.choice(len(di), size=len(ti), replace=False)
            pi=di[pick]; qi=di[~np.isin(np.arange(len(di)),pick)]
            if len(qi)<MIN_DON: continue
            tp=Yv.loc[pi,PATH].values.mean(axis=0)
            wp=omega(qi, m.loc[pi,'mvaz'].mean(), tp, PATH)
            pre_p=rmse_of(wp,qi,tp,PATH)
            tau_p=dp[pi].mean()-float(wp@dp[qi].values)
            pr.append(abs(tau_p)/max(pre_p,1e-6))
    rows.append({'k':k,'n_tr':len(ti),'n_don':len(di),'rmse_in5':r_in,'rmse_hold':r_out,
                 'pre_rmse':pre_rmse,'tau':tau,'plac':plac,'ratio':ratio,
                 'p_rank':(np.mean([1.0]+[p>=ratio for p in pr]) if pr else np.nan),
                 'n_plac':len(pr)})
r=pd.DataFrame(rows); r.to_csv("sc_validity.csv",index=False)
log(f"{len(r)} sectors")

print(f"\n{'='*84}\nA. HELD-OUT PRE-PERIOD  (fit omega on 2010-2014, track 2015-2017 unseen)\n{'='*84}")
print(f"  RMSE in-sample  2010-2014 : median {r['rmse_in5'].median():.4f}  mean {r['rmse_in5'].mean():.4f}")
print(f"  RMSE held-out   2015-2017 : median {r['rmse_hold'].median():.4f}  mean {r['rmse_hold'].mean():.4f}")
print(f"  degradation ratio out/in  : median {(r['rmse_hold']/r['rmse_in5']).median():.2f}x")
print(f"  sectors where held-out RMSE exceeds 2x in-sample: "
      f"{int((r['rmse_hold']>2*r['rmse_in5']).sum())} of {len(r)}")
sd_out=(Yv.loc[m['T']==1,HOLD].mean(axis=1)).std()
print(f"  for scale: cross-pair sd of log exports in 2015-17 = {sd_out:.3f}")

print(f"\n{'='*84}\nB. DOES BETTER FIT MEAN A SMALLER PLACEBO?\n{'='*84}")
print(f"  corr(pre RMSE, |placebo tau|) = {r['pre_rmse'].corr(r['plac'].abs()):+.3f}")
print(f"  corr(pre RMSE, |tau|)         = {r['pre_rmse'].corr(r['tau'].abs()):+.3f}")
q=r['pre_rmse'].quantile([1/3,2/3])
for lab,sel in [("best-fitting third",r['pre_rmse']<=q.iloc[0]),
                ("middle third",(r['pre_rmse']>q.iloc[0])&(r['pre_rmse']<=q.iloc[1])),
                ("worst-fitting third",r['pre_rmse']>q.iloc[1])]:
    s=r[sel]
    print(f"  {lab:22s} n={len(s):>2}  pre RMSE {s['pre_rmse'].median():.3f}   "
          f"tau {s['tau'].mean():+.4f}   |placebo| {s['plac'].abs().mean():.4f}")

print(f"\n{'='*84}\nC. RESTRICTING TO WELL-FITTING SECTORS\n{'='*84}")
def jk(v):
    v=np.asarray(v); n=len(v); mu=v.mean()
    a=np.array([np.delete(v,i).mean() for i in range(n)])
    return mu, np.sqrt((n-1)/n*((a-a.mean())**2).sum())
for lab,sel in [("all sectors",r['pre_rmse']<=r['pre_rmse'].max()),
                ("best half",r['pre_rmse']<=r['pre_rmse'].median()),
                ("best third",r['pre_rmse']<=q.iloc[0])]:
    s=r[sel]; a=jk(s['tau'].values); b=jk(s['plac'].values)
    print(f"  {lab:14s} n={len(s):>2}   tau {a[0]:+.4f} ({a[1]:.4f}) t={a[0]/a[1]:+.2f}"
          f"    placebo {b[0]:+.4f} ({b[1]:.4f}) t={b[0]/b[1]:+.2f}")

print(f"\n{'='*84}\nD. IN-SPACE PLACEBO / RMSPE RATIO TEST\n{'='*84}")
v=r.dropna(subset=['p_rank'])
print(f"  sectors with a usable placebo distribution: {len(v)} of {len(r)} "
      f"({int(v['n_plac'].median())} placebo groups each)")
print(f"  true group's post/pre RMSPE ratio: median {v['ratio'].median():.2f}")
print(f"  rank p-value: median {v['p_rank'].median():.3f}   "
      f"share p<0.10 {100*(v['p_rank']<0.10).mean():.0f}%   "
      f"share p<0.05 {100*(v['p_rank']<0.05).mean():.0f}%")
from scipy import stats
print(f"  Fisher combined across sectors: chi2({2*len(v)}) = "
      f"{-2*np.log(v['p_rank'].clip(1e-6)).sum():.1f}, "
      f"p = {1-stats.chi2.cdf(-2*np.log(v['p_rank'].clip(1e-6)).sum(),2*len(v)):.4f}")
print(f"  (under the null the rank p-values are uniform, so ~10% should fall below 0.10)")
