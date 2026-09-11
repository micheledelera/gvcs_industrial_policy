"""§6c. Inside the top quintile — where the weighted result actually lives.

§6b: within size quintiles 1-4 the unweighted and weighted estimates agree and are null.
Within quintile 5 they still diverge (+0.065 unweighted vs +0.412** weighted) because that
quintile holds 98.2% of the trade value and is internally as skewed as the whole sample.

So split the top quintile finer -- into five bands of 4% each -- and ask where the weighted
result comes from. Same specification throughout:

    log X_ist = beta (HIGH_is x post_t) + alpha_is + alpha_it + alpha_st + e

If the effect sits in the top 4% -- a few dozen very large country-sectors -- the result is
about a handful of flows and should be named as such. If it is spread across the top
quintile, the earlier nulls were a small-sector artefact.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc15_bridge.py").read().split('FE="fe_is + fe_it + fe_st"')[0])
FE="fe_is + fe_it + fe_st"
NAME={484:'Mexico',699:'India',704:'Vietnam',458:'Malaysia',764:'Thailand',76:'Brazil',
 360:'Indonesia',608:'Philippines',792:'Turkiye',50:'Bangladesh',616:'Poland',348:'Hungary',
 643:'Russia',152:'Chile',710:'South Africa',116:'Cambodia',214:'Dominican Rep',
 188:'Costa Rica',504:'Morocco',788:'Tunisia',642:'Romania',100:'Bulgaria',191:'Croatia'}
def nm(c): return NAME.get(int(c),str(int(c)))

HL=X[((X['HIGH']==1)|(X['LOW']==1)) & X['complete']].copy()
sz=HL.groupby('pair')['wpre'].first()
q5=set(sz.index[pd.qcut(sz.rank(method='first'),5,labels=False)==4])
T=HL[HL['pair'].isin(q5)].copy()
szt=T.groupby('pair')['wpre'].first()
T['b']=T['pair'].map(pd.qcut(szt.rank(method='first'),5,labels=[1,2,3,4,5]))
tot=sz.sum()
log(f"top quintile: {len(q5):,} units, {len(T):,} rows, {100*szt.sum()/tot:.1f}% of all value")

def fit(d,kind,wt=None):
    d=d.copy()
    if kind!='ppml': d=d[d['X']>0].copy(); d['ly']=np.log(d['X'])
    try:
        if kind=='ppml':
            m=pf.fepois(f"X ~ HIGHxP | {FE}",data=d,vcov={"CRV1":"cl"},iwls_maxiter=400,
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=20000),
                        lean=True,store_data=False,copy_data=False)
        else:
            m=pf.feols(f"ly ~ HIGHxP | {FE}",data=d,weights=wt,vcov={"CRV1":"cl"},
                       lean=True,store_data=False,copy_data=False)
        t=m.tidy().loc['HIGHxP']; r=(t['Estimate'],t['Std. Error'],t['Pr(>|t|)'])
        del m; gc.collect(); return r
    except Exception: return (np.nan,np.nan,np.nan)
def cell(r):
    if not np.isfinite(r[0]): return f"{'--':>22s}"
    st='***' if r[2]<.01 else '**' if r[2]<.05 else '*' if r[2]<.10 else ''
    return f"{r[0]:>+8.3f} ({r[1]:.3f}){st:3s}"

print(f"\n{'='*118}\nINSIDE THE TOP QUINTILE: five bands of 4% each\n{'='*118}")
print(f"{'band':>16s}{'units':>7s}{'HIGH':>6s}{'median $k':>13s}{'% of ALL value':>16s}"
      f"{'OLS logs, unweighted':>24s}{'OLS logs, weighted':>23s}")
rows=[]
for b,lab in zip([1,2,3,4,5],['80-84th pct','84-88th','88-92nd','92-96th','top 4%']):
    d=T[T['b']==b]; u=d.groupby('pair').first()
    a=fit(d,'ols'); w=fit(d,'ols','wpre')
    print(f"{lab:>16s}{len(u):>7d}{int(u['HIGH'].sum()):>6d}{u['wpre'].median():>13,.0f}"
          f"{100*u['wpre'].sum()/tot:>15.1f}%{cell(a):>24s}{cell(w):>23s}")
    rows.append(dict(band=lab,units=len(u),high=int(u['HIGH'].sum()),
                     med=u['wpre'].median(),valshare=u['wpre'].sum()/tot,
                     ols=a[0],ols_se=a[1],ols_p=a[2],olsw=w[0],olsw_se=w[1],olsw_p=w[2]))
a=fit(T,'ols'); w=fit(T,'ols','wpre')
print(f"{'-'*118}\n{'top quintile':>16s}{len(q5):>7d}"
      f"{int(T.groupby('pair')['HIGH'].first().sum()):>6d}{szt.median():>13,.0f}"
      f"{100*szt.sum()/tot:>15.1f}%{cell(a):>24s}{cell(w):>23s}")
pd.DataFrame(rows).to_csv("sc17_topsplit.csv",index=False)

print(f"\n  the top 4% -- the {len(T[T['b']==5].groupby('pair'))} largest country-sectors:")
top=T[T['b']==5].groupby('pair').first().sort_values('wpre',ascending=False)
print(f"    HIGH-policy among them: "+", ".join(
    f"{nm(p[0])}-{p[1]}" for p in top.index[top['HIGH']==1][:12]))
print(f"    share of all value held by these {len(top)} units: "
      f"{100*top['wpre'].sum()/tot:.1f}%")
log("DONE")
