"""Xu's NONPARAMETRIC bootstrap applied to the project data. Never run before: SS9b validated
it in Xu's Monte Carlo, found it worse than the parametric version, and chose the parametric
route -- so on the actual panel it has no reading at all. It is also Xu's own recommendation
for large N_tr ("when N_tr is large in particular, a simple nonparametric bootstrap procedure
can provide valid uncertainty estimates"), and N_tr here is 1,083.

Resample whole blocks of units with replacement and refit. Two variants, and the difference
is the ESTIMAND rather than the method:
  resample_treated=False  only control blocks are redrawn -> the ATT of the treated units we
                          actually observe (Xu's stated target, "the ATT in the sample we
                          draw"). SS9b: runs ~12% light because the treated units' own errors
                          never vary.
  resample_treated=True   treated blocks redrawn too -> the POPULATION ATT, the effect for a
                          policy-using country-sector drawn at random from the population the
                          25 observed countries represent. Wider, and arguably the estimand
                          the research question actually asks about.
Crossed with the blocking level, which SS10c/the clustering sweep showed is the binding choice.

RESULT NOTE, recorded here so the CSV is not misread later. At SECTOR level the
resample_treated=False cell is DEGENERATE and must be discarded: all 123 ISIC 4-digit sectors
contain at least one treated unit, so there are no control-only blocks to resample and the
function returns the original panel every draw (se exactly 0, CI collapsed on the point
estimate). The country cut has 114 control-only blocks of 139 and is fine. The sector
resample_treated=True cell is valid but mislabelled: with no control-only blocks it is a plain
sector-level cluster bootstrap, and the sample-versus-population distinction does not exist
there.
"""
import numpy as np, pandas as pd, sys, time
from scipy import stats
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
import gsc as G
exec(open("b2_estimate.py").read().split('print(f"\\n{\'=\'*104}\\nB2.4')[0])
t0=time.time()
KEY,R,THR,B='lnX',2,6,400
Y,X,W,ct,se_,s_=build(THR,KEY)
div=np.array([s[:2] for s in se_])
base=G.gsc(Y,X,W,T0,R)['att'][T0:].mean()
print(f"\nlnX  r={R}  ATT {base:+.4f}\n")
ROWS=[]
for blab,bl in [('country',ct),('ISIC 4-digit sector',se_)]:
    for tlab,rt in [('sample ATT (controls only)',False),('population ATT (treated too)',True)]:
        b0,sd,A=G.bootstrap_np(Y,X,W,T0,R,B=B,rng=np.random.default_rng(7),blocks=bl,
                               resample_treated=rt)
        d=A[:,T0:].mean(axis=1)
        s=d.std(ddof=1); ng=len(np.unique(bl[W]))
        p_n=2*(1-stats.norm.cdf(abs(base/s))); p_t=2*(1-stats.t.cdf(abs(base/s),df=ng-1))
        lo,hi=np.percentile(d,[2.5,97.5])
        print(f"{blab:20s} | {tlab:28s} draws {len(A):4d}  se {s:.4f}  t {base/s:+5.2f}  "
              f"p_t {p_t:.3f}  percentile CI [{lo:+.3f}, {hi:+.3f}]",flush=True)
        ROWS.append(dict(outcome=KEY,block=blab,estimand=tlab,n_draws=len(A),se=s,
                         att=base,p_norm=p_n,p_t=p_t,ci_lo=lo,ci_hi=hi))
        print(f"  [{time.time()-t0:.0f}s]",flush=True)
pd.DataFrame(ROWS).to_csv("c_npboot.csv",index=False)
print(f"[{time.time()-t0:.0f}s] saved c_npboot.csv")
