"""B2.0. Validate the from-scratch GSC against Xu (2017)'s own Monte Carlo design.

Xu's DGP, his equation (3):
  Y_it = delta_it D_it + x_it1*1 + x_it2*3 + lambda_i'f_t + alpha_i + xi_t + 5 + eps_it
  x_itk = 1 + lambda_i'f_t + lambda_i1 + lambda_i2 + f_1t + f_2t + eta_itk,  k = 1,2
  eps, eta, f_1t, f_2t, xi_t ~ iid N(0,1)
  lambda_i1, lambda_i2, alpha_i ~ U[-sqrt3, sqrt3]              for controls
                                ~ U[sqrt3-2w*sqrt3, 3sqrt3-2w*sqrt3] for treated, w = 0.8
  delta_it (t > T0) = deltabar_t + e_it,  e_it ~ N(0,1),  deltabar = [1,2,...,10]
  N_tr = 5, q = 10 post-treatment periods.

Per his section 4.2, observables, factors and loadings are drawn ONCE per design cell and
only the error term is redrawn; treatment effects are taken as given once the sample is
drawn. So bias is measured against the REALISED ATT at T0+5, not against 5.

NOTE ON WHAT IS BEING VALIDATED. The provided text of the paper references Table 1 but does
not reproduce its digits, so this is not a digit-for-digit replication. It checks the
properties Xu states in the text:
  (i)   limited bias even at small T0 and N_co, shrinking as both grow
  (ii)  SD and RMSE shrinking as T0 and N_co grow
  (iii) RMSE close to SD (his remark, implying bias is the small component)
  (iv)  parametric-bootstrap 95% CIs attaining nominal coverage, "even when T0 = 15,
        N_tr = 5, N_co = 80" -- his named cell
  (v)   the CV scheme picking r = 2 most of the time
  (vi)  GSC less biased than DiD given decomposable time-varying confounders, less biased
        than IFE under heterogeneous effects, and more efficient than canonical SC
"""
import numpy as np, sys, time
sys.path.insert(0, "/home/user/gvcs_industrial_policy/data")
from gsc import gsc, cv_r, ife_fit, bootstrap
t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
S3 = np.sqrt(3.0); W = 0.8; NTR = 5; Q = 10; DBAR = np.arange(1, 11)

def design(T0, Nco, seed):
    """draw the fixed part of a design cell once: X, F, Lambda, alpha, xi, delta"""
    rng = np.random.default_rng(seed)
    T = T0 + Q; N = NTR + Nco
    treated = np.zeros(N, bool); treated[:NTR] = True
    lam = np.empty((N, 2)); al = np.empty(N)
    lo, hi = S3 - 2*W*S3, 3*S3 - 2*W*S3
    lam[treated] = rng.uniform(lo, hi, (NTR, 2)); al[treated] = rng.uniform(lo, hi, NTR)
    lam[~treated] = rng.uniform(-S3, S3, (Nco, 2)); al[~treated] = rng.uniform(-S3, S3, Nco)
    F = rng.normal(size=(T, 2)); xi = rng.normal(size=T)
    lf = F @ lam.T                                              # T x N
    eta = rng.normal(size=(T, N, 2))
    X = np.empty((T, N, 2))
    for kk in range(2):
        X[:, :, kk] = (1 + lf + lam[:, 0][None, :] + lam[:, 1][None, :]
                       + F[:, 0][:, None] + F[:, 1][:, None] + eta[:, :, kk])
    D = np.zeros((T, N)); D[T0:, treated] = 1.0
    delta = np.zeros((T, N))
    delta[T0:, treated] = DBAR[:, None] + rng.normal(size=(Q, NTR))
    mu = X[:, :, 0]*1.0 + X[:, :, 1]*3.0 + lf + al[None, :] + xi[:, None] + 5.0
    target = delta[T0+4, treated].mean()                        # realised ATT at T0+5
    return dict(T=T, T0=T0, N=N, treated=treated, X=X, mu=mu, D=D, delta=delta,
                target=target, lam=lam)

def draw(d, rng):
    return d['mu'] + d['delta']*d['D'] + rng.normal(size=(d['T'], d['N']))

# ---- comparison estimators ----------------------------------------------------------
def did_att(Y, X, treated, T0):
    """literal DiD: unit and time intercepts from controls/pre, then the gap"""
    T = Y.shape[0]; pre = np.arange(T0); co = ~treated
    b, F, L, a, xi = ife_fit(Y[:, co], X[:, co, :], 1)
    F[:] = 0; L[:] = 0                                          # r = 0 -> two-way FE
    b, F2, L2, a, xi = ife_fit(Y[:, co], X[:, co, :], 1)
    fit = X[:, co, :] @ b + a[None, :] + xi[:, None]
    # refit with no factors
    Yc = Y[:, co]; k = X.shape[2]
    beta = b
    for _ in range(200):
        R = Yc - X[:, co, :] @ beta
        a = R.mean(axis=0); xi = (R - a[None, :]).mean(axis=1)
        Z = X[:, co, :].reshape(-1, k); y = (Yc - a[None, :] - xi[:, None]).reshape(-1)
        beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    net = Y[:, treated] - X[:, treated, :] @ beta - xi[:, None]
    a_t = net[pre].mean(axis=0)
    Y0 = X[:, treated, :] @ beta + a_t[None, :] + xi[:, None]
    return (Y[:, treated] - Y0).mean(axis=1)

def ife_att(Y, X, treated, T0, r=2):
    """Bai (2009) IFE with a constant treatment effect imposed"""
    T, N = Y.shape; k = X.shape[2]
    D = np.zeros((T, N)); D[T0:, treated] = 1.0
    Z = np.concatenate([X, D[:, :, None]], axis=2)
    beta = np.zeros(k+1); a = np.zeros(N); xi = np.zeros(T)
    F = np.zeros((T, r)); L = np.zeros((N, r))
    for _ in range(300):
        M = Y - Z @ beta
        Rf = M - F @ L.T
        a = Rf.mean(axis=0); xi = (Rf - a[None, :]).mean(axis=1)
        R = M - a[None, :] - xi[:, None]
        w, V = np.linalg.eigh(R @ R.T)
        U = V[:, np.argsort(w)[::-1][:r]]; F = U*np.sqrt(T); L = R.T @ F / T
        fit = a[None, :] + xi[:, None] + F @ L.T
        beta = np.linalg.lstsq(Z.reshape(-1, k+1), (Y - fit).reshape(-1), rcond=None)[0]
    return beta[-1]

def _proj(v):
    u = np.sort(v)[::-1]; c = np.cumsum(u) - 1.0; rr = np.arange(1, len(v)+1)
    m = u - c/rr > 0
    return np.maximum(v - c[m][-1]/rr[m][-1], 0.0)

def sc_att(Y, treated, T0, it=2000):
    """canonical Abadie on pre-period outcomes, per treated unit, simplex, no intercept"""
    co = np.where(~treated)[0]; tr = np.where(treated)[0]
    A = Y[:T0][:, co]
    gaps = []
    for a_ in tr:
        b = Y[:T0, a_]
        w = np.ones(len(co))/len(co); Lc = np.linalg.norm(A, 2)**2 + 1e-12
        for _ in range(it):
            w = _proj(w - (A.T @ (A @ w - b))/Lc)
        gaps.append(Y[:, a_] - Y[:, co] @ w)
    return np.array(gaps).mean(axis=0)

# ---- (i)-(iii) bias, SD, RMSE ---------------------------------------------------------
REPS = 500
print(f"\n{'='*104}\nXu Table 1 properties: ATT at T0+5, N_tr = 5, r fixed at 2, "
      f"{REPS} replications per cell\n{'='*104}")
print(f"  {'T0':>4s} {'Nco':>5s} {'target':>8s} {'mean est':>9s} {'bias':>8s} "
      f"{'SD':>7s} {'RMSE':>7s}")
for T0 in [10, 15, 20, 30]:
    for Nco in [40, 80]:
        d = design(T0, Nco, seed=1000+T0*10+Nco)
        rng = np.random.default_rng(7)
        est = []
        for _ in range(REPS):
            Y = draw(d, rng)
            est.append(gsc(Y, d['X'], d['treated'], T0, 2)['att'][T0+4])
        est = np.array(est); bias = est.mean() - d['target']
        print(f"  {T0:4d} {Nco:5d} {d['target']:8.3f} {est.mean():9.3f} {bias:+8.3f} "
              f"{est.std(ddof=1):7.3f} {np.sqrt(np.mean((est-d['target'])**2)):7.3f}")
    log(f"T0={T0} done")
