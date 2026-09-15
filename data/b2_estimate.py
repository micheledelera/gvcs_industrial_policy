"""B2. GSC estimation under the merged protocol.

Design frozen in A0-A4:
  outcome     lnS primary (i's share of US imports in sector k), lnD secondary
              (US share of the country-sector's exports), lnX as a check
  event       2018
  treated     persistently targeted, threshold SWEPT 5+..9+ of 2009-17 (§10b: the
              threshold sits inside a smooth distribution, so the sweep is part of the
              primary result, not a robustness check)
  controls    strictly clean -- zero interventions in every year 2009-2024
  X           Dec_k x Post
  r           swept 0-4; CV picks 1 with 2 within 2% (§10); §9c says the CV errs downward
  inference   parametric bootstrap, BLOCKED AT COUNTRY (§9b), max_loo capped

Plus the two checks §10b demanded: size strata, and excluding units that overlap the
share_frac_policies definition §5k voided.
"""
import pandas as pd, numpy as np, sys, gc, time
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
import gsc as G
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
USA, CHINA = 842, 156
PRE=list(range(2007,2018)); POST=list(range(2018,2025)); YRS=PRE+POST; BASE=[2015,2016,2017]
P9=list(range(2009,2018))
NAME={484:'Mexico',699:'India',704:'Vietnam',458:'Malaysia',764:'Thailand',76:'Brazil',
 360:'Indonesia',608:'Philippines',792:'Turkiye',50:'Bangladesh',616:'Poland',348:'Hungary',
 643:'Russia',152:'Chile',710:'South Africa',604:'Peru',586:'Pakistan',32:'Argentina',
 170:'Colombia',642:'Romania',818:'Egypt',504:'Morocco',100:'Bulgaria',191:'Croatia',
 144:'Sri Lanka',116:'Cambodia',214:'Dominican Rep',188:'Costa Rica',688:'Serbia',
 804:'Ukraine',682:'Saudi Arabia',784:'UAE',634:'Qatar',218:'Ecuador',788:'Tunisia'}
def nm(c): return NAME.get(int(c),str(int(c)))

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
us=raw[raw['j']==USA]
XUS=us.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
XTOT=raw.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
SK=us.groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
CHN=us[us['i']==CHINA].groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
pol=raw[['i','ISIC4c','t','n_policies','share_frac_policies']].drop_duplicates(
    subset=['i','ISIC4c','t'])
for c in ['n_policies','share_frac_policies']:
    pol[c]=pd.to_numeric(pol[c],errors='coerce').fillna(0)
del raw, us; gc.collect()
for D in (XUS,XTOT,SK,CHN): D.columns=[int(c) for c in D.columns]
XUS=XUS.reindex(columns=YRS); XTOT=XTOT.reindex(columns=YRS)
SK=SK.reindex(columns=YRS); CHN=CHN.reindex(columns=YRS)
XUS=XUS.loc[[(i,k) for i,k in XUS.index if adv.get(i,1)==0 and i!=CHINA]]
XUS=XUS[(XUS[YRS]>0).all(axis=1) & XUS[YRS].notna().all(axis=1)]
idx=XUS.index; XTOT=XTOT.reindex(idx)
CT=np.array([a for a,_ in idx]); SE=np.array([b for _,b in idx])
onp=pol.pivot_table(index=['i','ISIC4c'],columns='t',values='n_policies',
                    aggfunc='sum').reindex(idx).fillna(0)
on=(onp>0)
pre_yrs=on[P9].sum(axis=1).values
post_yrs=on[[y for y in POST if y in on.columns]].sum(axis=1).values
SF=(pol[pol['t'].isin(BASE)].groupby(['i','ISIC4c'],observed=True)['share_frac_policies']
    .mean().reindex(idx).fillna(0).values)
den=np.array([SK.loc[k,YRS].values for _,k in idx])
OUT={'lnS':np.log(XUS[YRS].values/den),
     'lnD':np.log((XUS[YRS].values/XTOT[YRS].values).clip(1e-12,1.0)),
     'lnX':np.log(XUS[YRS].values)}
dec=np.array([100*CHN.loc[k,BASE].mean()/SK.loc[k,BASE].mean() for _,k in idx])
sz=XUS[BASE].mean(axis=1).values
clean=(pre_yrs==0)&(post_yrs==0)
T0=len(PRE); postv=np.array([1.0 if y>=2018 else 0.0 for y in YRS])
log(f"panel built: {len(idx):,} units, clean controls {int(clean.sum()):,}")

def build(thr, key, mask_extra=None):
    tr=(pre_yrs>=thr)
    if mask_extra is not None: tr=tr&mask_extra
    use=tr|clean
    Y=OUT[key][use].T
    X=(dec[use][None,:,None]*postv[:,None,None])
    return Y, X, tr[use], CT[use], SE[use], sz[use]

def att_post(Y,X,W,r):
    g=G.gsc(Y,X,W,T0,r)
    return g['att'][T0:].mean(), g['att']

print(f"\n{'='*104}\nB2.4  POINT ESTIMATES: threshold x outcome x r   (post-2018 mean ATT, "
      f"log points)\n{'='*104}")
rows=[]
for key in ['lnS','lnD','lnX']:
    print(f"\n  {key}")
    print(f"  {'thr':>4s} {'treated':>8s} "+"".join(f"{'r='+str(r):>10s}" for r in range(5)))
    for thr in [5,6,7,8,9]:
        Y,X,W,ct,se_,s_=build(thr,key)
        line=f"  {thr:3d}+ {int(W.sum()):8d} "
        for r in range(5):
            a,_=att_post(Y,X,W,r); line+=f"{a:+10.4f}"
            rows.append(dict(outcome=key,thr=thr,r=r,att=a,n_tr=int(W.sum())))
        print(line)
    log(f"{key} grid done")
pd.DataFrame(rows).to_csv("b2_grid.csv",index=False)

print(f"\n{'='*104}\nB2.5  PRIMARY with country-blocked bootstrap: lnS, threshold 6+, "
      f"r = 1 and 2\n{'='*104}")
BOOT=[]
for key in ['lnS','lnD','lnX']:
    for r in ([1,2] if key=='lnS' else [2]):
        Y,X,W,ct,se_,s_=build(6,key)
        att,sd,A=G.bootstrap(Y,X,W,T0,r,B=200,rng=np.random.default_rng(31),
                             blocks=ct,max_loo=60)
        pm=att[T0:].mean(); ps=A[:,T0:].mean(axis=1).std(ddof=1)
        pre_m=att[:T0].mean(); pre_s=A[:,:T0].mean(axis=1).std(ddof=1)
        print(f"  {key} r={r}:  post ATT {pm:+.4f} (se {ps:.4f}, t={pm/ps:+.2f})   "
              f"pre-period mean gap {pre_m:+.4f} (se {pre_s:.4f}, t={pre_m/pre_s:+.2f})")
        print(f"      by year: "+" ".join(f"{y}:{att[i]:+.3f}" for i,y in enumerate(YRS)))
        BOOT.append(dict(outcome=key,r=r,post=pm,post_se=ps,pre=pre_m,pre_se=pre_s))
        np.save(f"b2_att_{key}_r{r}.npy",np.vstack([att,A.std(axis=0,ddof=1)]))
        log(f"bootstrap {key} r={r} done")
pd.DataFrame(BOOT).to_csv("b2_boot.csv",index=False)
