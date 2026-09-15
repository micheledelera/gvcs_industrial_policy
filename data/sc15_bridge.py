"""§6. Bridging gravity (+5.2%, p=0.002) and the demeaned SC (+0.058, p=0.50).

Both are within-country-relative estimands asking nearly the same question -- does a
policy-targeted sector outperform its country's other sectors -- and they disagree. §3ac
identified the candidate channels. This walks them one at a time on a COMMON panel, so
whichever step moves the coefficient is the answer.

Common ground: the country-sector-year panel of US-bound exports, developing ex-China,
with alpha_is + alpha_it + alpha_st throughout. alpha_it IS the country-year demeaning of
§5f, so the fixed-effect structure matches the SC's estimand by construction.

The ladder, each row changing ONE thing from the row above:

  1. PPML levels, continuous IP, x target x post, all units   <- the gravity-family result
  2. PPML levels, continuous IP, x post only                  drop the decoupling contrast
  3. PPML levels, HIGH-quartile dummy, x post only            continuous -> binary
  4. OLS logs,    HIGH-quartile dummy, x post only            levels -> logs
  5. OLS logs,    HIGH dummy, x post, complete-series sample  the §5f sample restriction
  6. OLS logs,    HIGH dummy, x post, HIGH+LOW units only     drop the middle, as §5f does
  7. row 6 weighted by pre-period US exports                  the weighting channel

Row 6 is the closest regression analogue of §5f (+0.058). If the coefficient dies between
rows 1 and 4 it is functional form; between 4 and 6 it is sample and treatment definition;
if row 7 revives it, it is size weighting -- the §3ac / §3v tension again.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, time
t0=time.time()
def log(m): print(f"[{time.time()=='' and 0 or (time.time()-t0):.0f}s] {m}", flush=True)
USA, CHINA = 842, 156
PRE=list(range(2007,2018)); POST=list(range(2018,2025)); YRS=PRE+POST; BASE=[2015,2016,2017]

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
raw['share_frac_policies']=pd.to_numeric(raw['share_frac_policies'],errors='coerce').fillna(0)
raw['target']=pd.to_numeric(raw['target'],errors='coerce').fillna(0)
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
us=raw[raw['j']==USA]
X=us.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum().rename('X').reset_index()
tg=(raw[['ISIC4c','target']].drop_duplicates().groupby('ISIC4c',observed=True)['target'].max())
ipc=(raw[raw['t'].isin(BASE)][['i','ISIC4c','t','share_frac_policies']]
     .drop_duplicates(subset=['i','ISIC4c','t'])
     .groupby(['i','ISIC4c'],observed=True)['share_frac_policies'].mean().rename('ip'))
del raw,us; gc.collect()

X['Advanced_i']=X['i'].map(adv)
X=X[(X['Advanced_i']==0)&(X['i']!=CHINA)&(X['t'].isin(YRS))].copy()
X=X.merge(ipc,on=['i','ISIC4c'],how='left')
X['ip']=X['ip'].fillna(0.0)
X['target']=X['ISIC4c'].map(tg).fillna(0.0)
X['post']=(X['t']>=2018).astype(float)
X['ipz']=X['ip']/X['ip'].std()
pos=X.loc[X['ip']>0,'ip']
q75,q25=pos.quantile(.75),pos.quantile(.25)
X['HIGH']=(X['ip']>=q75).astype(float)
X['LOW']=(X['ip']<=q25).astype(float)
# §5f sample: complete positive US series over PRE and POST
piv=X.pivot_table(index=['i','ISIC4c'],columns='t',values='X',aggfunc='first')
comp=set(piv.index[(piv[YRS]>0).all(axis=1) & piv[YRS].notna().all(axis=1)])
X['pair']=list(zip(X['i'],X['ISIC4c']))
X['complete']=X['pair'].isin(comp)
wpre=(X[X['t'].isin(BASE)].groupby('pair')['X'].mean().rename('wpre'))
X=X.merge(wpre,on='pair',how='left')
X['DDD']=X['ipz']*X['target']*X['post']
X['IPxP']=X['ipz']*X['post']
X['HIGHxP']=X['HIGH']*X['post']
X['fe_is']=X.groupby(['i','ISIC4c'],observed=True).ngroup().astype('int32')
X['fe_it']=X.groupby(['i','t'],observed=True).ngroup().astype('int32')
X['fe_st']=X.groupby(['ISIC4c','t'],observed=True).ngroup().astype('int32')
X['cl']=X['fe_is']
log(f"panel {len(X):,} rows | {X['pair'].nunique():,} pairs | complete-series pairs {len(comp):,}")
print(f"  HIGH units {int(X.groupby('pair')['HIGH'].first().sum()):,}   "
      f"LOW units {int(X.groupby('pair')['LOW'].first().sum()):,}\n")

FE="fe_is + fe_it + fe_st"
def run(lab, dat, y, rhs, kind, wt=None):
    d=dat.copy()
    if kind=='ols':
        d=d[d['X']>0].copy(); d['ly']=np.log(d['X']); y='ly'
    try:
        if kind=='ppml':
            m=pf.fepois(f"{y} ~ {rhs} | {FE}", data=d, vcov={"CRV1":"cl"},
                        iwls_maxiter=400, demeaner=pf.LsmrDemeaner(fixef_maxiter=20000),
                        lean=True, store_data=False, copy_data=False)
        else:
            m=pf.feols(f"{y} ~ {rhs} | {FE}", data=d, weights=wt, vcov={"CRV1":"cl"},
                       lean=True, store_data=False, copy_data=False)
        t=m.tidy(); k=rhs.split(' + ')[0]
        b,se,p=t.loc[k,'Estimate'],t.loc[k,'Std. Error'],t.loc[k,'Pr(>|t|)']
        st='***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''
        print(f"  {lab:<58s} N={int(m._N):>9,}  {k:<7s} {b:>+8.4f} ({se:.4f}){st:3s} p={p:.4f}")
        del m; gc.collect()
        return dict(step=lab,term=k,coef=b,se=se,p=p,N=int(m._N) if False else None)
    except Exception as e:
        print(f"  {lab:<58s} FAILED: {type(e).__name__}: {str(e)[:60]}")
        return None

HL=X[(X['HIGH']==1)|(X['LOW']==1)]
print(f"{'='*116}\nBRIDGE: gravity-family -> demeaned synthetic control\n{'='*116}")
out=[]
out.append(run("1. PPML levels, continuous IP x target x post","",'X',"DDD + ipz",'ppml') if False else None)
out.append(run("1. PPML levels, continuous IP x target x post",X,'X',"DDD + IPxP",'ppml'))
out.append(run("2. PPML levels, continuous IP x post (no target)",X,'X',"IPxP",'ppml'))
out.append(run("3. PPML levels, HIGH-quartile dummy x post",X,'X',"HIGHxP",'ppml'))
out.append(run("4. OLS logs,    HIGH dummy x post",X,'X',"HIGHxP",'ols'))
out.append(run("5. OLS logs,    HIGH x post, complete-series sample",X[X['complete']],'X',"HIGHxP",'ols'))
out.append(run("6. OLS logs,    HIGH x post, HIGH+LOW only  [~ §5f]",
               HL[HL['complete']],'X',"HIGHxP",'ols'))
out.append(run("7. row 6, weighted by pre-period US exports",
               HL[HL['complete']],'X',"HIGHxP",'ols','wpre'))
print(f"\n  reference points:  gravity §2 DDD_dev +0.0524 (p=0.002)   "
      f"demeaned SC §5f tau_aug +0.058 (p=0.50)")
pd.DataFrame([o for o in out if o]).to_csv("sc15_bridge.csv",index=False)
log("DONE")
