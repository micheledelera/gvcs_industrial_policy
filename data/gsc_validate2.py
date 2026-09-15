"""B2.0 continued. Xu (2017) properties (iv) coverage, (v) CV choosing r, (vi) head-to-head."""
import numpy as np, sys, time
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
from gsc import gsc, cv_r, ife_fit, bootstrap
exec(open("gsc_validate.py").read().split("# ---- comparison estimators")[0]
     .split('"""',2)[2].replace("from gsc import gsc, cv_r, ife_fit, bootstrap",""))
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

def twoway_att(Y, X, treated, T0):
    """literal DiD: two-way FE fitted on controls, treated intercept from its pre-period"""
    pre=np.arange(T0); co=~treated; k=X.shape[2]
    Yc=Y[:,co]; Xc=X[:,co,:]; beta=np.zeros(k); a=np.zeros(co.sum()); xi=np.zeros(Y.shape[0])
    for _ in range(300):
        R=Yc-Xc@beta
        a=R.mean(axis=0); xi=(R-a[None,:]).mean(axis=1)
        beta=np.linalg.lstsq(Xc.reshape(-1,k),(Yc-a[None,:]-xi[:,None]).reshape(-1),rcond=None)[0]
    net=Y[:,treated]-X[:,treated,:]@beta-xi[:,None]
    Y0=X[:,treated,:]@beta+net[pre].mean(axis=0)[None,:]+xi[:,None]
    return (Y[:,treated]-Y0).mean(axis=1)

def ife_att(Y, X, treated, T0, r=2):
    T,N=Y.shape; k=X.shape[2]
    D=np.zeros((T,N)); D[T0:,treated]=1.0
    Z=np.concatenate([X,D[:,:,None]],axis=2)
    beta=np.zeros(k+1); F=np.zeros((T,r)); L=np.zeros((N,r))
    for _ in range(300):
        M=Y-Z@beta; Rf=M-F@L.T
        a=Rf.mean(axis=0); xi=(Rf-a[None,:]).mean(axis=1)
        R=M-a[None,:]-xi[:,None]
        w,V=np.linalg.eigh(R@R.T); U=V[:,np.argsort(w)[::-1][:r]]
        F=U*np.sqrt(T); L=R.T@F/T
        fit=a[None,:]+xi[:,None]+F@L.T
        beta=np.linalg.lstsq(Z.reshape(-1,k+1),(Y-fit).reshape(-1),rcond=None)[0]
    return beta[-1]

def _proj(v):
    u=np.sort(v)[::-1]; c=np.cumsum(u)-1.0; rr=np.arange(1,len(v)+1); m=u-c/rr>0
    return np.maximum(v-c[m][-1]/rr[m][-1],0.0)
def sc_att(Y, treated, T0, it=1500):
    co=np.where(~treated)[0]; A=Y[:T0][:,co]; Lc=np.linalg.norm(A,2)**2+1e-12
    gaps=[]
    for a_ in np.where(treated)[0]:
        b=Y[:T0,a_]; w=np.ones(len(co))/len(co)
        for _ in range(it): w=_proj(w-(A.T@(A@w-b))/Lc)
        gaps.append(Y[:,a_]-Y[:,co]@w)
    return np.array(gaps).mean(axis=0)

# ---- (v) does the CV pick r = 2? ----
print(f"\n{'='*100}\n(v) Algorithm 1: leave-one-pre-period-out CV choice of r "
      f"(truth = 2 factors, additive FE imposed)\n{'='*100}")
for T0,Nco in [(15,40),(20,45),(20,80),(30,80)]:
    d=design(T0,Nco,seed=2000+T0*10+Nco); rng=np.random.default_rng(11); picks=[]
    for _ in range(200):
        Y=draw(d,rng); picks.append(cv_r(Y,d['X'],d['treated'],T0,rmax=4)[1])
    picks=np.array(picks)
    dist="  ".join(f"r={r}:{100*(picks==r).mean():.0f}%" for r in range(5))
    print(f"  T0={T0:3d} Nco={Nco:3d}   {dist}   correct {100*(picks==2).mean():.0f}%")
    log(f"CV T0={T0} Nco={Nco} done")

# ---- (vi) head-to-head at Xu's illustration cell ----
print(f"\n{'='*100}\n(vi) Head-to-head, T0 = 20, Nco = 45, N_tr = 5, ATT at T0+5, "
      f"300 replications\n{'='*100}")
d=design(20,45,seed=4242); rng=np.random.default_rng(3); R={k:[] for k in
    ['GSC','DiD (two-way FE)','IFE (constant effect)','SC (Abadie)']}
for _ in range(300):
    Y=draw(d,rng)
    R['GSC'].append(gsc(Y,d['X'],d['treated'],20,2)['att'][24])
    R['DiD (two-way FE)'].append(twoway_att(Y,d['X'],d['treated'],20)[24])
    R['IFE (constant effect)'].append(ife_att(Y,d['X'],d['treated'],20,2))
    R['SC (Abadie)'].append(sc_att(Y,d['treated'],20)[24])
tgt=d['target']
print(f"  target (realised ATT at T0+5) = {tgt:.3f}\n")
print(f"  {'estimator':24s} {'mean':>8s} {'bias':>8s} {'SD':>7s} {'RMSE':>7s}")
for k,v in R.items():
    v=np.array(v)
    print(f"  {k:24s} {v.mean():8.3f} {v.mean()-tgt:+8.3f} {v.std(ddof=1):7.3f} "
          f"{np.sqrt(np.mean((v-tgt)**2)):7.3f}")
log("head-to-head done")

# ---- (iv) bootstrap coverage ----
print(f"\n{'='*100}\n(iv) Algorithm 2 parametric bootstrap: 95% CI coverage for ATT at T0+5"
      f"\n{'='*100}")
for T0,Nco,NS,B in [(15,80,100,150),(20,45,100,150)]:
    d=design(T0,Nco,seed=5000+T0); rng=np.random.default_rng(17); hit=[]; ses=[]
    for _ in range(NS):
        Y=draw(d,rng)
        att,se,_=bootstrap(Y,d['X'],d['treated'],T0,2,B=B,rng=rng,max_loo=40)
        lo,hi=att[T0+4]-1.96*se[T0+4], att[T0+4]+1.96*se[T0+4]
        hit.append(lo<=d['target']<=hi); ses.append(se[T0+4])
    print(f"  T0={T0:3d} Nco={Nco:3d} N_tr=5   coverage {100*np.mean(hit):.1f}%  "
          f"(nominal 95%)   mean bootstrap se {np.mean(ses):.3f}")
    log(f"coverage T0={T0} done")
