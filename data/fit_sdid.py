"""Synthetic difference-in-differences (Arkhangelsky, Athey, Hirshberg, Imbens & Wager
2021), run per sector on sector-specific decoupling dates.

WHY SDiD HERE. Every design so far has TESTED the parallel-trends assumption and mostly
failed it. SDiD does not assume it: unit weights omega are chosen so the donor pool's
pre-treatment path MATCHES the treated units', and time weights lambda downweight
pre-periods that look unlike the post. It is doubly robust -- consistent if either set of
weights does its job -- and its unit weights are a principled answer to the
weighted-vs-unweighted tension of §3v, which turned on trade size rather than on fit.

OUTCOME. log US exports, not share of US imports. §3w-adjacent diagnostic
(diag_sdid_feasible.py) found treated pairs hold a MEDIAN 50.1% of developing-ex-China
supply in their own sector. With a share outcome, anything the treated gain is
mechanically taken from the donors used to build their counterfactual, which is
disqualifying for SC specifically. Log exports needs a complete positive series: 2,567
of 8,833 pairs qualify but hold 99.5% of pre-period US value.

DESIGN. Donors are restricted to the SAME sector, so exposure to Chinese withdrawal is
held fixed exactly as in every other design here. Treated = IP>0; E_k from the 25%
threshold definition. One SDiD per sector, then aggregated.

ESTIMATOR, per sector k with pre-periods P and post-periods Q:
  omega : min_w  || Y[don,P]' w + w0 - mean_tr Y[tr,P] ||^2 + zeta^2 |P| ||w||^2
          w >= 0, sum w = 1                      (projected gradient on the simplex)
  lambda: min_l  || Y[don,P] l + l0 - mean_post Y[don,Q] ||^2
          l >= 0, sum l = 1
  tau   : weighted two-way regression with unit weights omega and time weights lambda,
          which for the 2x2 block reduces to
          (Ybar_tr,Q - Ybar_tr,P^lambda) - (Ybar_don,Q^omega - Ybar_don,P^omega,lambda)

  zeta = (n_tr * |Q|)^(1/4) * sd(first differences of donor outcomes over P)

INFERENCE. Jackknife over sectors for the aggregate (Arkhangelsky et al. §5, appropriate
when there are many treated units). Plus an in-time PLACEBO: re-run with a fake event 3
years before the true one, using only pre-event data. A placebo ATT near zero is the SC
analogue of a flat pre-trend.
"""
import pandas as pd, numpy as np, gc, sys, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
USA, CHINA = 842, 156
BASE=[2015,2016,2017]; THR=0.25
MEASURE = sys.argv[1] if len(sys.argv)>1 else 'share_frac_policies'
MIN_TR, MIN_DON = 3, 8

# ---------- simplex projection and projected-gradient solver ----------
def proj_simplex(v):
    u=np.sort(v)[::-1]; c=np.cumsum(u)-1.0
    r=np.arange(1,len(v)+1)
    cond=u-c/r>0
    if not cond.any(): return np.ones_like(v)/len(v)
    rho=r[cond][-1]; theta=c[cond][-1]/rho
    return np.maximum(v-theta,0.0)

def solve_w(A, b, zeta2, iters=800):
    """min ||A'w + w0 - b||^2 + zeta2*||w||^2 over the simplex, w0 free."""
    n=A.shape[0]; w=np.ones(n)/n
    L=np.linalg.norm(A,2)**2/max(len(b),1)+zeta2+1e-12
    step=1.0/L
    for _ in range(iters):
        f=A.T@w; w0=np.mean(b-f)
        g=A@(f+w0-b)*2.0/max(len(b),1)+2.0*zeta2*w
        w=proj_simplex(w-step*g)
    f=A.T@w
    return w, float(np.mean(b-f))

# ---------- data ----------
raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
raw[MEASURE]=pd.to_numeric(raw[MEASURE],errors='coerce').fillna(0).astype('float32')
us=raw[raw['j']==USA]
x=us.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum().rename('X')
tot=x.groupby(level=['ISIC4c','t'],observed=True).transform('sum')
s=(x/tot.replace(0,np.nan)*100)
chn=s.unstack('t').xs(CHINA,level='i') if False else s.xs(CHINA,level='i').unstack('t')
chn.columns=[int(c) for c in chn.columns]; chn=chn.reindex(columns=sorted(chn.columns))
b=chn[BASE].mean(axis=1); chn=chn.loc[b>=5.0]; b=b.loc[chn.index]
below=chn.lt(b*(1-THR),axis=0)
stay=below.iloc[:,::-1].cummin(axis=1).iloc[:,::-1].astype(bool)
E=stay.apply(lambda r: r.index[r.values.argmax()] if r.any() else np.nan,axis=1).dropna().astype(int)
ip=(raw[raw['t'].isin(BASE)][['i','ISIC4c','t',MEASURE]].drop_duplicates(subset=['i','ISIC4c','t'])
    .groupby(['i','ISIC4c'],observed=True)[MEASURE].mean().rename('IP'))
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
years=sorted(raw['t'].unique())
del raw,us; gc.collect()

X=x.unstack('t'); X.columns=[int(c) for c in X.columns]; X=X[sorted(X.columns)]
idx=[t for t in X.index if t[1] in E.index and adv.get(t[0],1)==0 and t[0]!=CHINA]
X=X.loc[idx]
X=X[(X>0).all(axis=1)]                         # complete positive series
Y=np.log(X)
meta=pd.DataFrame(index=Y.index)
meta['IP']=ip.reindex(Y.index).fillna(0.0)
meta['T']=(meta['IP']>0).astype(int)
meta['E']=[E[k] for _,k in Y.index]
meta['wpre']=X[BASE].mean(axis=1)
log(f"balanced pairs {len(Y):,} | treated {int(meta['T'].sum()):,} | "
    f"sectors {Y.index.get_level_values('ISIC4c').nunique()}")

def sdid_sector(Yk, Tk, P, Q):
    tr=Yk[Tk==1]; dn=Yk[Tk==0]
    if len(tr)<MIN_TR or len(dn)<MIN_DON or len(P)<3 or len(Q)<1: return None
    Ap=dn[P].values; bt=tr[P].values.mean(axis=0)
    sig=np.diff(dn[P].values,axis=1).std()
    zeta2=((len(tr)*len(Q))**0.5)*(sig**2)+1e-8
    w,_=solve_w(Ap,bt,zeta2)
    l,_=solve_w(Ap.T, dn[Q].values.mean(axis=1), 1e-8)
    tr_post=tr[Q].values.mean(); tr_pre=float(tr[P].values.mean(axis=0)@l)
    dn_post=float(w@dn[Q].values.mean(axis=1)); dn_pre=float(w@(dn[P].values@l))
    return (tr_post-tr_pre)-(dn_post-dn_pre), len(tr), len(dn), w, l

def run(placebo_shift=0, label="SDiD"):
    out=[]
    for k,g in meta.groupby(level='ISIC4c', observed=True):
        if len(g)==0: continue
        e=int(g['E'].iloc[0])-placebo_shift
        P=[y for y in years if y<e]; Q=[y for y in years if y>=e]
        if placebo_shift>0: Q=[y for y in Q if y< e+placebo_shift]   # placebo post stays pre-event
        Yk=Y.loc[g.index]; Tk=g['T'].values
        r=sdid_sector(Yk,Tk,P,Q)
        if r is None: continue
        out.append({'k':k,'E':int(g['E'].iloc[0]),'tau':r[0],'n_tr':r[1],'n_don':r[2],
                    'w_tr':g.loc[g['T']==1,'wpre'].sum()})
    o=pd.DataFrame(out)
    if o.empty: print(f"{label}: no estimable sectors"); return o
    simple=o['tau'].mean()
    wtd=np.average(o['tau'],weights=o['w_tr'])
    n=len(o); jk=np.array([o['tau'].drop(i).mean() for i in o.index])
    se=np.sqrt((n-1)/n*((jk-jk.mean())**2).sum())
    jkw=np.array([np.average(o['tau'].drop(i),weights=o['w_tr'].drop(i)) for i in o.index])
    sew=np.sqrt((n-1)/n*((jkw-jkw.mean())**2).sum())
    from scipy import stats
    print(f"\n{'='*72}\n{label}   {n} sectors, {int(o['n_tr'].sum())} treated, "
          f"{int(o['n_don'].sum())} donor slots\n{'='*72}")
    for lab,v,e_ in [("simple mean ATT",simple,se),("trade-weighted ATT",wtd,sew)]:
        t_=v/e_ if e_>0 else np.nan
        print(f"  {lab:22s} {v:+.4f}  (jackknife se {e_:.4f})  t={t_:+.2f}  "
              f"p={2*(1-stats.norm.cdf(abs(t_))):.4f}")
    print(f"  sector taus: median {o['tau'].median():+.4f}  "
          f"IQR [{o['tau'].quantile(.25):+.4f}, {o['tau'].quantile(.75):+.4f}]  "
          f"share>0 {100*(o['tau']>0).mean():.0f}%")
    return o

main=run(0,"SDiD -- true event")
plac=run(3,"PLACEBO -- fake event 3 years early, post window entirely pre-event")
if not main.empty:
    main.to_csv(f"sdid_{MEASURE}.csv",index=False)
    print("\nby event cohort:")
    print(main.groupby('E').agg(sectors=('k','size'),mean_tau=('tau','mean'),
                                median=('tau','median')).round(4).to_string())
log("DONE")
