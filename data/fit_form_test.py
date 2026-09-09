"""Levels or logs? Isolating the functional form, holding sample and spec fixed.

§3ac showed the §3s/§3ab disagreement is not the sample: on the SC's own 2,852 pairs,
PPML still gives non-US −0.0190**. So it is the estimator. This holds the sample AND the
triple-difference structure fixed and varies only the functional form and weighting:

  P.  PPML on levels          X_ist  ~ DDD + IP | a_it + a_st + a_is     (as §3s)
  L.  OLS on logs             log X  ~ DDD + IP | a_it + a_st + a_is
  W.  OLS on logs, weighted   as L, weighted by pre-period US exports

If L flips non-US positive while P keeps it negative on identical rows, the disagreement
is levels-versus-logs -- i.e. whether large flows or proportional changes dominate --
which is the §3v weighted/unweighted tension reappearing as a functional-form choice.
W then says whether re-weighting logs by size recovers the PPML answer.

Sample: SC pairs (event sectors, complete positive US series). Logs need positive
outcomes, so rows with a zero outcome drop -- reported per outcome.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("fit_realloc_restricted.py").read().split("SAMP=[")[0])

q=p[p['pair'].isin(sc_pairs)].copy()
q['fe_it']=q.groupby(['i','t_int'],observed=True).ngroup().astype('int32')
q['fe_st']=q.groupby(['ISIC4c','t_int'],observed=True).ngroup().astype('int32')
q['fe_is']=q.groupby(['i','ISIC4c'],observed=True).ngroup().astype('int32')
q['cl']=q['fe_is']
wpre=(q[q['t_int'].isin(BASE)].groupby(['i','ISIC4c'],observed=True)['X_us'].mean()).rename('wpre')
q=q.merge(wpre,on=['i','ISIC4c'],how='left')
q=q[q['wpre']>0].copy()
log(f"SC sample: rows {len(q):,}  pairs {q['fe_is'].nunique():,}  sectors {q['ISIC4c'].nunique()}")

print(f"\n{'='*94}")
print(f"{'outcome':>12s}{'P. PPML levels':>26s}{'L. OLS logs':>26s}{'W. OLS logs, wtd':>26s}")
print(f"{'='*94}")
rows=[]
for lbl,y in [("TOTAL",'X_tot'),("US-bound",'X_us'),("NON-US",'X_non')]:
    line=f"{lbl:>12s}"; rec={'outcome':lbl}
    for tag,kind in [('P','ppml'),('L','ols'),('W','olsw')]:
        try:
            if kind=='ppml':
                m=pf.fepois(f"{y} ~ DDD + IPz | fe_it + fe_st + fe_is", data=q,
                            vcov={"CRV1":"cl"}, iwls_maxiter=500,
                            demeaner=pf.LsmrDemeaner(fixef_maxiter=20000),
                            lean=True, store_data=False, copy_data=False)
            else:
                qq=q[q[y]>0].copy(); qq['ly']=np.log(qq[y])
                m=pf.feols(f"ly ~ DDD + IPz | fe_it + fe_st + fe_is", data=qq,
                           weights=('wpre' if kind=='olsw' else None),
                           vcov={"CRV1":"cl"}, lean=True, store_data=False, copy_data=False)
            t=m.tidy()
            b_,se,pv=t.loc['DDD','Estimate'],t.loc['DDD','Std. Error'],t.loc['DDD','Pr(>|t|)']
            st='***' if pv<.01 else '**' if pv<.05 else '*' if pv<.10 else ''
            line+=f"{b_:+.4f} ({se:.4f}){st:3s} N={int(m._N)//1000}k".rjust(26)
            rec[tag]=b_; rec[tag+'_se']=se; rec[tag+'_p']=pv; rec[tag+'_N']=int(m._N)
            del m; gc.collect()
        except Exception as e:
            line+=f"{'FAILED':>26s}"; print(f"\n  {lbl} {tag}: {type(e).__name__}: {e}")
    print(line); rows.append(rec)
pd.DataFrame(rows).to_csv("form_test.csv",index=False)
log("DONE")
