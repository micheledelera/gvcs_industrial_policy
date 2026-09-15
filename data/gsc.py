"""Generalized synthetic control (Xu 2017, Political Analysis 25:57-76), from scratch.

No R in this container, so `gsynth` is unavailable. This implements the estimator directly:

  Step 1  interactive fixed effects on CONTROLS ONLY (Bai 2009):
          min over beta, F, Lambda_co  sum_{i in C} ||Y_i - X_i beta - a_i - xi - F lambda_i||^2
          s.t. F'F/T = I_r.  Additive unit and time effects imposed (Xu's Remark 3 /
          extended model), which is what makes the CV target r = 2 in his Monte Carlo
          rather than r = 4.
  Step 2  each treated unit's loadings by projecting its PRE-treatment outcomes onto the
          estimated factor space, with an intercept so the treated unit's own additive
          effect is estimated from the pre-period too.
  Step 3  impute Y_it(0) = x_it'beta + a_i + xi_t + lambda_i'f_t, and ATT_t is the mean
          gap over treated units.

Also: Algorithm 1 (leave-one-pre-period-out CV for r) and Algorithm 2 (parametric
bootstrap; block resampling of whole residual time series, with treated units drawn from
leave-one-control-out PREDICTION errors rather than in-sample residuals).

The alpha/lambda split is not separately identified once both are present; only the
prediction a_i + xi_t + lambda_i'f_t is, and that is all the estimator uses.
"""
import numpy as np

def _factors(R, r):
    """top-r factors of a T x N residual matrix, normalised so F'F/T = I_r"""
    T = R.shape[0]
    w, V = np.linalg.eigh(R @ R.T)
    U = V[:, np.argsort(w)[::-1][:r]]
    F = U * np.sqrt(T)
    Lam = R.T @ F / T
    return F, Lam

def ife_fit(Yc, Xc, r, tol=1e-6, maxit=200):
    """Step 1. Yc: T x Nco. Xc: T x Nco x k (or None). Returns beta, F, Lam, a, xi."""
    T, N = Yc.shape
    k = 0 if Xc is None else Xc.shape[2]
    beta = np.zeros(k)
    a = np.zeros(N); xi = np.zeros(T)
    F = np.zeros((T, r)); Lam = np.zeros((N, r))
    prev = None
    for _ in range(maxit):
        M = Yc - (Xc @ beta if k else 0.0)
        # additive effects given factors
        Rf = M - F @ Lam.T
        a = Rf.mean(axis=0)
        xi = (Rf - a[None, :]).mean(axis=1)
        # factors given additive effects
        R = M - a[None, :] - xi[:, None]
        F, Lam = _factors(R, r)
        # beta given everything
        if k:
            fit = a[None, :] + xi[:, None] + F @ Lam.T
            Z = Xc.reshape(-1, k); y = (Yc - fit).reshape(-1)
            beta = np.linalg.lstsq(Z, y, rcond=None)[0]
        cur = a[None, :] + xi[:, None] + F @ Lam.T + (Xc @ beta if k else 0.0)
        if prev is not None and np.max(np.abs(cur - prev)) < tol: break
        prev = cur
    return beta, F, Lam, a, xi

def _treated_loadings(Yt, Xt, beta, F, xi, pre):
    """Step 2. Yt: T x Ntr. Regress pre-period net outcome on [1, F_pre]."""
    k = 0 if Xt is None else Xt.shape[2]
    net = Yt - (Xt @ beta if k else 0.0) - xi[:, None]
    D = np.hstack([np.ones((len(pre), 1)), F[pre]])
    coef = np.linalg.lstsq(D, net[pre], rcond=None)[0]        # (1+r) x Ntr
    return coef[0], coef[1:].T                                # a_tr, Lam_tr (Ntr x r)

def gsc(Y, X, treated, T0, r):
    """Y: T x N. X: T x N x k or None. treated: bool mask over units. Returns dict."""
    T = Y.shape[0]; pre = np.arange(T0)
    co = ~treated
    Xc = None if X is None else X[:, co, :]
    Xt = None if X is None else X[:, treated, :]
    beta, F, Lam, a_c, xi = ife_fit(Y[:, co], Xc, r)
    a_t, Lam_t = _treated_loadings(Y[:, treated], Xt, beta, F, xi, pre)
    Y0 = ((Xt @ beta if X is not None else 0.0)
          + a_t[None, :] + xi[:, None] + F @ Lam_t.T)
    gap = Y[:, treated] - Y0
    return dict(att=gap.mean(axis=1), gap=gap, Y0=Y0, beta=beta, F=F, Lam=Lam,
                Lam_t=Lam_t, a_c=a_c, a_t=a_t, xi=xi)

def cv_r(Y, X, treated, T0, rmax=5):
    """Algorithm 1. Leave-one-pre-period-out MSPE on the treated units."""
    pre = np.arange(T0); out = {}
    co = ~treated
    Xc = None if X is None else X[:, co, :]
    Xt = None if X is None else X[:, treated, :]
    for r in range(0, rmax + 1):
        if r == 0:
            beta, F, Lam, a_c, xi = ife_fit(Y[:, co], Xc, 1)
            F = np.zeros_like(F); Lam = np.zeros_like(Lam)
        else:
            beta, F, Lam, a_c, xi = ife_fit(Y[:, co], Xc, r)
        k = 0 if X is None else X.shape[2]
        net = Y[:, treated] - (Xt @ beta if k else 0.0) - xi[:, None]
        errs = []
        for s in pre:
            keep = pre[pre != s]
            D = np.hstack([np.ones((len(keep), 1)), F[keep]])
            cf = np.linalg.lstsq(D, net[keep], rcond=None)[0]
            pred = cf[0] + (F[s] @ cf[1:] if r else 0.0)
            errs.append(net[s] - pred)
        out[r] = float(np.mean(np.array(errs) ** 2))
    return out, min(out, key=out.get)

def bootstrap(Y, X, treated, T0, r, B=200, rng=None, blocks=None, max_loo=None):
    """Algorithm 2. Parametric bootstrap. `blocks` gives a group id per unit so whole
    groups (e.g. countries) are resampled together; default is unit-level as in Xu."""
    rng = rng or np.random.default_rng(0)
    T, N = Y.shape
    co = np.where(~treated)[0]; tr = np.where(treated)[0]
    base = gsc(Y, X, treated, T0, r)
    # in-sample control residuals
    Xc = None if X is None else X[:, co, :]
    beta, F, Lam, a_c, xi = ife_fit(Y[:, co], Xc, r)
    fitc = ((Xc @ beta if X is not None else 0.0) + a_c[None, :] + xi[:, None] + F @ Lam.T)
    eps_c = Y[:, co] - fitc                                   # T x Nco
    # leave-one-control-out prediction errors, for the treated units' error distribution
    eps_p = []
    loo = co if max_loo is None or len(co) <= max_loo else rng.choice(co, max_loo, replace=False)
    for j0 in range(len(loo)):
        j = int(np.where(co == loo[j0])[0][0])
        m = np.ones(N, bool); m[co[j]] = False
        fake = np.zeros(N, bool); fake[co[j]] = True
        sub = m | fake
        g = gsc(Y[:, sub], None if X is None else X[:, sub, :],
                fake[sub], T0, r)
        eps_p.append(g['gap'][:, 0])
    eps_p = np.array(eps_p).T                                  # T x Nco
    fit_all = np.zeros((T, N))
    fit_all[:, co] = fitc
    fit_all[:, tr] = base['Y0']
    atts = []
    for _ in range(B):
        Yb = fit_all.copy()
        if blocks is None:
            Yb[:, co] += eps_c[:, rng.integers(0, eps_c.shape[1], len(co))]
            Yb[:, tr] += eps_p[:, rng.integers(0, eps_p.shape[1], len(tr))]
        else:
            for grp in np.unique(blocks):
                idx = np.where(blocks == grp)[0]
                src = rng.integers(0, eps_c.shape[1])
                pick = np.array([src] * len(idx))
                for jj, u in enumerate(idx):
                    Yb[:, u] += (eps_p[:, pick[jj] % eps_p.shape[1]] if treated[u]
                                 else eps_c[:, pick[jj] % eps_c.shape[1]])
        atts.append(gsc(Yb, X, treated, T0, r)['att'])
    A = np.array(atts)
    return base['att'], A.std(axis=0, ddof=1), A
