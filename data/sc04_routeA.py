"""ROUTE A. Canonical Abadie synthetic control on US orientation, one SC per country,
with the numerator/denominator decomposition.

  outcome   log orientation_it = log(i's share of US imports) - log(i's share of
            rest-of-world imports).  Scale-free by construction. Chosen in §4c as the
            only candidate where all 39 treated countries lie inside the donor hull AND
            the fit is non-degenerate.
  treated   39 developing countries ex-China with any recorded GTA policy
  donors    130 developing countries ex-China with none
  pre       2007-2017      post 2018-2024      treatment date 2018
  weights   canonical Abadie: w >= 0, sum w = 1, NO intercept, matched on the eleven
            lagged outcomes (Ben-Michael, Feller & Rothstein note that lagged outcomes
            alone are increasingly standard)

  inference Abadie's in-space placebo: assign treatment to each of the 130 donors, build
            its own SC from the remaining 129, compute post/pre RMSPE, and rank each
            treated country's ratio against that distribution.

  DECOMPOSITION. Because log orientation = log(US share) - log(RoW share) identically,
  applying the SAME omega to each component splits the estimated gap exactly:
      gap_orientation = gap_US_share - gap_RoW_share
  A gap driven by the US numerator with RoW flat is expansion tilted at America; one
  driven by the RoW denominator falling is diversion. This is what recovers the
  substantive question from a compositional outcome.

  OVERFITTING CHECK. 130 donors fitting 11 observations is the Ferman-Pinto concern.
  Also fit omega on 2007-2013 alone and report tracking on 2014-2017 unseen.
"""
import pandas as pd, numpy as np, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
USA, CHINA = 842, 156
PRE=list(range(2007,2018)); POST=list(range(2018,2025))
FIT5=list(range(2007,2014)); HOLD=list(range(2014,2018))
NAME={484:'Mexico',699:'India',704:'Vietnam',458:'Malaysia',764:'Thailand',76:'Brazil',
 360:'Indonesia',608:'Philippines',792:'Turkiye',50:'Bangladesh',616:'Poland',348:'Hungary',
 643:'Russia',152:'Chile',710:'South Africa',604:'Peru',586:'Pakistan',32:'Argentina',
 170:'Colombia',784:'UAE',642:'Romania',218:'Ecuador',682:'Saudi Arabia',818:'Egypt',
 504:'Morocco',68:'Bolivia',634:'Qatar',804:'Ukraine',100:'Bulgaria',191:'Croatia',
 404:'Kenya',788:'Tunisia',566:'Nigeria',688:'Serbia',12:'Algeria',508:'Mozambique',
 288:'Ghana',524:'Nepal',368:'Iraq',214:'Dominican Rep',222:'El Salvador',188:'Costa Rica',
 320:'Guatemala',340:'Honduras',558:'Nicaragua',591:'Panama',858:'Uruguay',116:'Cambodia',
 144:'Sri Lanka',780:'Trinidad&Tob',328:'Guyana',388:'Jamaica',862:'Venezuela',600:'Paraguay'}
def nm(c): return NAME.get(int(c),str(int(c)))

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
raw['n_policies']=pd.to_numeric(raw['n_policies'],errors='coerce').fillna(0)
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
pol=raw.groupby('i')['n_policies'].max()
us=raw[raw['j']==USA]
usi=us.groupby(['i','t'],observed=True)['imports'].sum().unstack('t')
alli=raw.groupby(['i','t'],observed=True)['imports'].sum().unstack('t')
usmkt=us.groupby('t',observed=True)['imports'].sum()
wldmkt=raw.groupby('t',observed=True)['imports'].sum()
del raw,us; gc.collect()
for D in (usi,alli): D.columns=[int(c) for c in D.columns]
usmkt.index=[int(c) for c in usmkt.index]; wldmkt.index=[int(c) for c in wldmkt.index]

YRS=[y for y in PRE+POST]
S_us =(usi.div(usmkt,axis=1))[YRS]
S_row=((alli-usi).div(wldmkt-usmkt,axis=1))[YRS]
keep=[i for i in S_us.index if adv.get(i,1)==0 and i!=CHINA]
S_us,S_row=S_us.loc[keep],S_row.loc[keep]
ok=((S_us>0).all(axis=1))&((S_row>0).all(axis=1))
S_us,S_row=S_us[ok],S_row[ok]
LU,LR=np.log(S_us),np.log(S_row)
Y=LU-LR                                       # log orientation
T=pd.Series({i: bool(pol.get(i,0)>0) for i in Y.index})
tr,dn=Y[T.values],Y[~T.values]
log(f"treated {len(tr)}  donors {len(dn)}  pre {PRE[0]}-{PRE[-1]}  post {POST[0]}-{POST[-1]}")

def proj_simplex(v):
    u=np.sort(v)[::-1]; c=np.cumsum(u)-1.0; r=np.arange(1,len(v)+1)
    cond=u-c/r>0
    if not cond.any(): return np.ones_like(v)/len(v)
    return np.maximum(v-c[cond][-1]/r[cond][-1],0.0)
def sc(A,b,iters=6000):
    n=A.shape[0]; w=np.ones(n)/n; L=np.linalg.norm(A,2)**2/A.shape[1]+1e-12
    for _ in range(iters): w=proj_simplex(w-(A@(A.T@w-b)*2.0/A.shape[1])/L)
    return w
def rmspe(b,f): return float(np.sqrt(((b-f)**2).mean()))

# ---- placebo distribution: each donor as pseudo-treated ----
plac=[]
for i in dn.index:
    D=dn.drop(index=i); b=dn.loc[i]
    w=sc(D[PRE].values,b[PRE].values)
    pre_=rmspe(b[PRE].values, D[PRE].values.T@w)
    post_=rmspe(b[POST].values, D[POST].values.T@w)
    plac.append(post_/max(pre_,1e-8))
plac=np.array(plac)
log(f"placebo distribution built from {len(plac)} donors; median ratio {np.median(plac):.2f}")

rows=[]; paths={}
for i in tr.index:
    b=tr.loc[i]
    w=sc(dn[PRE].values,b[PRE].values)
    fit=pd.Series(dn.values.T@w, index=YRS)
    pre_=rmspe(b[PRE].values, fit[PRE].values); post_=rmspe(b[POST].values, fit[POST].values)
    ratio=post_/max(pre_,1e-8)
    # held-out
    w5=sc(dn[FIT5].values, b[FIT5].values)
    r_in =rmspe(b[FIT5].values, dn[FIT5].values.T@w5)
    r_out=rmspe(b[HOLD].values, dn[HOLD].values.T@w5)
    # decomposition with the SAME omega
    gU=(LU.loc[i,POST]-(LU.loc[dn.index,POST].values.T@w)).mean()
    gR=(LR.loc[i,POST]-(LR.loc[dn.index,POST].values.T@w)).mean()
    gU_pre=(LU.loc[i,PRE]-(LU.loc[dn.index,PRE].values.T@w)).mean()
    gR_pre=(LR.loc[i,PRE]-(LR.loc[dn.index,PRE].values.T@w)).mean()
    rows.append({'i':i,'name':nm(i),'pre_rmspe':pre_,'post_rmspe':post_,'ratio':ratio,
                 'p':float((1+ (plac>=ratio).sum())/(1+len(plac))),
                 'gap_post':float((b[POST]-fit[POST]).mean()),
                 'gap_pre':float((b[PRE]-fit[PRE]).mean()),
                 'g_US':float(gU),'g_RoW':float(gR),
                 'g_US_pre':float(gU_pre),'g_RoW_pre':float(gR_pre),
                 'held_in':r_in,'held_out':r_out,
                 'w_max':w.max(),'n_pos':int((w>0.01).sum()),
                 'us_val':float(usi.loc[i,[2015,2016,2017]].mean())})
    paths[i]=(b,fit)
R=pd.DataFrame(rows).sort_values('us_val',ascending=False)
R.to_csv("sc04_routeA.csv",index=False)

print(f"\n{'='*112}\nCANONICAL ABADIE ON LOG US ORIENTATION — per country\n{'='*112}")
print(f"{'country':<15s}{'pre-RMSPE':>10s}{'post/pre':>10s}{'p':>7s}"
      f"{'gap pre':>9s}{'gap post':>10s}{'  =':>4s}{'US share':>10s}{'  -':>4s}{'RoW share':>11s}"
      f"{'donors':>8s}")
for _,x in R.head(20).iterrows():
    print(f"{x['name']:<15s}{x['pre_rmspe']:>10.4f}{x['ratio']:>10.2f}{x['p']:>7.3f}"
          f"{x['gap_pre']:>+9.3f}{x['gap_post']:>+10.3f}{'':>4s}{x['g_US']:>+10.3f}{'':>4s}"
          f"{x['g_RoW']:>+11.3f}{x['n_pos']:>8d}")
print(f"\n{'='*112}\nAGGREGATE\n{'='*112}")
def jk(v):
    v=np.asarray(v); n=len(v); a=np.array([np.delete(v,i).mean() for i in range(n)])
    return v.mean(), np.sqrt((n-1)/n*((a-a.mean())**2).sum())
for lab,col in [("gap, pre-period (should be ~0)",'gap_pre'),
                ("gap, post-period  [the estimate]",'gap_post'),
                ("  ...from US share (numerator)",'g_US'),
                ("  ...from RoW share (denominator, enters negatively)",'g_RoW')]:
    m_,s_=jk(R[col].values)
    print(f"  {lab:<52s}{m_:>+9.4f} ({s_:.4f})  t={m_/s_:>+6.2f}")
print(f"\n  check: {R['g_US'].mean():+.4f} - ({R['g_RoW'].mean():+.4f}) = "
      f"{R['g_US'].mean()-R['g_RoW'].mean():+.4f}  vs gap_post {R['gap_post'].mean():+.4f}")
print(f"\n  placebo rank p-values: median {R['p'].median():.3f}   "
      f"p<0.10 {100*(R['p']<0.10).mean():.0f}% (null 10%)   p<0.05 {100*(R['p']<0.05).mean():.0f}% (null 5%)")
print(f"\n  OVERFITTING CHECK (omega on 2007-13, tracking 2014-17 unseen)")
print(f"    in-sample RMSPE  median {R['held_in'].median():.4f}")
print(f"    held-out RMSPE   median {R['held_out'].median():.4f}"
      f"   degradation {R['held_out'].median()/max(R['held_in'].median(),1e-9):.1f}x")
print(f"    donors with w>0.01: median {R['n_pos'].median():.0f} of {len(dn)}"
      f"   median max weight {R['w_max'].median():.3f}")
log("DONE")
