"""B2.0 part 3. Re-validation after the two §9 fixes.

FIX 1  cv_r's r = 0 option is now a genuine two-way fixed effects fit (ife_fit accepts
       r = 0 and adds no factors), not an r = 1 fit with the factors zeroed afterwards.
FIX 2  the leave-one-control-out prediction errors feeding the parametric bootstrap are now
       computed against the OTHER CONTROLS only. Previously the donor pool evaluated to
       every unit, so each fake-treated control was predicted from a pool still containing
       the real treated units, whose post-period outcomes carry delta != 0.

Checks: the size of the prediction errors before and after the fix; parametric-bootstrap
coverage against the REALISED sample ATT (Xu's stated estimand); nonparametric block
bootstrap coverage at large N_tr, for both the sample ATT (controls resampled only) and the
population ATT (treated resampled too); and the CV's r distribution with the fixed r = 0.
"""
import numpy as np, sys, time
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
import gsc as G
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
S3=np.sqrt(3.0); W=0.8; Q=10; DBAR=np.arange(1,11)

def design(T0,Nco,seed,Ntr=5):
    rng=np.random.default_rng(seed); T=T0+Q; N=Ntr+Nco
    treated=np.zeros(N,bool); treated[:Ntr]=True
    lam=np.empty((N,2)); al=np.empty(N)
    lo,hi=S3-2*W*S3, 3*S3-2*W*S3
    lam[treated]=rng.uniform(lo,hi,(Ntr,2)); al[treated]=rng.uniform(lo,hi,Ntr)
    lam[~treated]=rng.uniform(-S3,S3,(Nco,2)); al[~treated]=rng.uniform(-S3,S3,Nco)
    F=rng.normal(size=(T,2)); xi=rng.normal(size=T); lf=F@lam.T
    X=np.empty((T,N,2))
    for kk in range(2):
        X[:,:,kk]=(1+lf+lam[:,0][None,:]+lam[:,1][None,:]
                   +F[:,0][:,None]+F[:,1][:,None]+rng.normal(size=(T,N)))
    D=np.zeros((T,N)); D[T0:,treated]=1.0
    delta=np.zeros((T,N)); delta[T0:,treated]=DBAR[:,None]+rng.normal(size=(Q,Ntr))
    mu=X[:,:,0]*1.0+X[:,:,1]*3.0+lf+al[None,:]+xi[:,None]+5.0
    return dict(T=T,T0=T0,N=N,Ntr=Ntr,treated=treated,X=X,mu=mu,D=D,delta=delta,
                target=delta[T0+4,treated].mean(), pop_target=float(DBAR[4]))
def draw(d,rng): return d['mu']+d['delta']*d['D']+rng.normal(size=(d['T'],d['N']))

# ---- did the fix change the prediction errors? ----
print(f"\n{'='*100}\nFIX 2 diagnostic: size of the leave-one-control-out prediction errors"
      f"\n{'='*100}")
d=design(20,45,seed=999); rng=np.random.default_rng(5); Y=draw(d,rng)
co=np.where(~d['treated'])[0]
def eps_p_of(pool_includes_treated):
    out=[]
    for u in co[:25]:
        if pool_includes_treated:
            sub=np.ones(d['N'],bool)                      # the old, buggy pool
        else:
            sub=(~d['treated']).copy()                    # the fixed pool
        fake=np.zeros(d['N'],bool); fake[u]=True
        g=G.gsc(Y[:,sub],d['X'][:,sub,:],fake[sub],20,2)
        out.append(g['gap'][:,0])
    return np.array(out).T
old=eps_p_of(True); new=eps_p_of(False)
print(f"  post-period sd of prediction errors: OLD pool (incl. treated) {old[20:].std():.3f}"
      f"   FIXED pool (controls only) {new[20:].std():.3f}"
      f"   ratio {old[20:].std()/new[20:].std():.2f}x")
print(f"  pre-period  sd: OLD {old[:20].std():.3f}   FIXED {new[:20].std():.3f}")
log("diagnostic done")

# ---- CV with the fixed r = 0 branch ----
print(f"\n{'='*100}\nFIX 1: CV choice of r, r = 0 now a true two-way FE fit (truth r = 2)"
      f"\n{'='*100}")
for T0,Nco in [(15,40),(20,45),(20,80),(30,80)]:
    dd=design(T0,Nco,seed=2000+T0*10+Nco); rg=np.random.default_rng(11); picks=[]
    for _ in range(200): picks.append(G.cv_r(draw(dd,rg),dd['X'],dd['treated'],T0,rmax=4)[1])
    picks=np.array(picks)
    print(f"  T0={T0:3d} Nco={Nco:3d}   "+"  ".join(f"r={r}:{100*(picks==r).mean():.0f}%"
          for r in range(5))+f"   correct {100*(picks==2).mean():.0f}%")
log("CV done")

# ---- parametric bootstrap coverage, small N_tr, sample-ATT estimand ----
print(f"\n{'='*100}\nAlgorithm 2 (parametric) coverage after FIX 2 -- target = realised "
      f"sample ATT at T0+5\n{'='*100}")
for T0,Nco,NS,B in [(15,80,80,120),(20,45,80,120)]:
    dd=design(T0,Nco,seed=5000+T0); rg=np.random.default_rng(17); hit=[]; ses=[]; ests=[]
    for _ in range(NS):
        Y=draw(dd,rg)
        att,se,_=G.bootstrap(Y,dd['X'],dd['treated'],T0,2,B=B,rng=rg,max_loo=30)
        ests.append(att[T0+4]); ses.append(se[T0+4])
        hit.append(abs(att[T0+4]-dd['target'])<=1.96*se[T0+4])
    print(f"  T0={T0:3d} Nco={Nco:3d} N_tr=5   coverage {100*np.mean(hit):.1f}%  "
          f"mean boot se {np.mean(ses):.3f}   actual sd of estimate {np.std(ests,ddof=1):.3f}"
          f"   se/sd {np.mean(ses)/np.std(ests,ddof=1):.2f}")
    log(f"parametric coverage T0={T0} done")

# ---- nonparametric block bootstrap, large N_tr ----
print(f"\n{'='*100}\nNonparametric block bootstrap, LARGE N_tr (Xu's prescription)"
      f"\n{'='*100}")
for T0,Nco,Ntr,NS,B in [(20,80,50,60,150)]:
    dd=design(T0,Nco,seed=7777,Ntr=Ntr); rg=np.random.default_rng(23)
    for lab,rt,tgt in [("sample ATT  (controls resampled only)",False,dd['target']),
                       ("population ATT (treated resampled too)",True,dd['pop_target'])]:
        hit=[]; ses=[]; ests=[]
        for _ in range(NS):
            Y=draw(dd,rg)
            att,se,_=G.bootstrap_np(Y,dd['X'],dd['treated'],T0,2,B=B,rng=rg,
                                    resample_treated=rt)
            ests.append(att[T0+4]); ses.append(se[T0+4])
            hit.append(abs(att[T0+4]-tgt)<=1.96*se[T0+4])
        print(f"  N_tr={Ntr} Nco={Nco} T0={T0}  {lab:40s} target {tgt:.3f}  "
              f"coverage {100*np.mean(hit):.1f}%  boot se {np.mean(ses):.3f}  "
              f"actual sd {np.std(ests,ddof=1):.3f}  se/sd {np.mean(ses)/np.std(ests,ddof=1):.2f}")
        log(f"nonparametric {lab[:12]} done")
