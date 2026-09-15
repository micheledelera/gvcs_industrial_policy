"""B2.0b. Validate GSC at T0 = 11 -- this project's actual pre-period length.

T0 = 11 is a data constraint: the Zenodo source panel is 2007-2024, so extending the
pre-period would mean rebuilding from raw BACI. Xu cautions against T0 < 10, and §9b found
the CV selects r = 0 more often as T0 falls -- and T0 = 11 is shorter than every cell
validated so far (coverage at 15 and 20, the CV at 15 and above). So this checks bias,
CV behaviour and bootstrap coverage at T0 = 11 exactly, with N_co pushed toward our 2,497
and N_tr toward our 500-1,083.

Xu's DGP throughout (his eq. 3), q = 10 post-periods, two true factors, w = 0.8.
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
                target=delta[T0+4,treated].mean())
def draw(d,rng): return d['mu']+d['delta']*d['D']+rng.normal(size=(d['T'],d['N']))

print(f"\n{'='*104}\n(1) Bias, SD, RMSE at T0 = 11, ATT at T0+5, r fixed at 2, 400 reps"
      f"\n{'='*104}")
print(f"  {'T0':>3s} {'Nco':>6s} {'Ntr':>4s} {'target':>8s} {'bias':>8s} {'SD':>7s} "
      f"{'RMSE':>7s} {'floor 1/sqrt(Ntr)':>18s}")
for Nco,Ntr in [(80,5),(400,5),(1000,5),(1000,50)]:
    d=design(11,Nco,seed=300+Nco+Ntr); rng=np.random.default_rng(7); e=[]
    for _ in range(400): e.append(G.gsc(draw(d,rng),d['X'],d['treated'],11,2)['att'][15])
    e=np.array(e)
    print(f"  {11:3d} {Nco:6d} {Ntr:4d} {d['target']:8.3f} {e.mean()-d['target']:+8.3f} "
          f"{e.std(ddof=1):7.3f} {np.sqrt(np.mean((e-d['target'])**2)):7.3f} "
          f"{1/np.sqrt(Ntr):18.3f}")
log("bias/SD done")

print(f"\n{'='*104}\n(2) CV choice of r at T0 = 11 (truth r = 2), 200 draws per cell"
      f"\n{'='*104}")
for Nco in [80,400,1000,2000]:
    d=design(11,Nco,seed=600+Nco); rng=np.random.default_rng(11); p=[]
    for _ in range(200): p.append(G.cv_r(draw(d,rng),d['X'],d['treated'],11,rmax=4)[1])
    p=np.array(p)
    print(f"  Nco={Nco:5d}   "+"  ".join(f"r={r}:{100*(p==r).mean():.0f}%" for r in range(5))
          +f"   correct {100*(p==2).mean():.0f}%")
    log(f"CV Nco={Nco} done")

print(f"\n{'='*104}\n(3) Parametric bootstrap coverage at T0 = 11 -- target = realised "
      f"sample ATT at T0+5\n{'='*104}")
for Nco,Ntr,NS,B in [(80,5,150,200),(400,5,150,200),(1000,5,120,200),(1000,50,100,150)]:
    d=design(11,Nco,seed=900+Nco+Ntr,Ntr=Ntr); rng=np.random.default_rng(17)
    hit=[]; ses=[]; est=[]
    for _ in range(NS):
        Y=draw(d,rng)
        att,se,_=G.bootstrap(Y,d['X'],d['treated'],11,2,B=B,rng=rng,max_loo=50)
        est.append(att[15]); ses.append(se[15])
        hit.append(abs(att[15]-d['target'])<=1.96*se[15])
    sd=np.std(est,ddof=1)
    print(f"  Nco={Nco:5d} Ntr={Ntr:3d}   coverage {100*np.mean(hit):5.1f}%   "
          f"boot se {np.mean(ses):.3f}   actual sd {sd:.3f}   se/sd {np.mean(ses)/sd:.2f}")
    log(f"coverage Nco={Nco} Ntr={Ntr} done")
