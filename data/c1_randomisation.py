"""C1. The ADH randomisation test for the merged design, as the protocol's C4 note demands:
an AGGREGATE test -- the group statistic ranked against placebo groups of equal size -- rather
than 1,083 individual exact p-values.

Placebo universe is the 2,223 strictly clean controls ONLY. The real treated units never enter
a placebo fit, as either placebo-treated or donor. For each draw a placebo-treated set is taken
from the clean controls and the REMAINING clean controls are the donor pool, then the identical
GSC estimator is refitted and the post-2018 mean ATT recorded.

Three draw schemes, because no scheme can match both the cluster count and the size profile:
the nine largest treated countries have 119..75 units and no control country has more than 112.

  block-weighted   countries drawn without replacement with probability proportional to their
                   unit count, all their units taken, until the treated total is reached; the
                   last country truncated. Matches total size and the country lumpiness; the
                   number of clusters floats.
  profile-matched  25 distinct control countries, one per treated country in descending order,
                   taking min(n_c, available) units. Matches the cluster count (25) and
                   approximates the size profile; the achieved total falls short.
  unit-wise        1,083 clean controls drawn uniformly, ignoring country. The anti-conservative
                   benchmark -- SS8d found this understates the null by about 2.6x.

Statistic is the ATT itself, not ADH's post/pre RMSPE ratio: SS5c found the ratio behaves
backwards in this panel (well-fitting placebos have tiny denominators and explode), and the GSC
pre-period gap is zero per unit by construction, so the ratio has no usable denominator here.

Two-sided exact p = (1 + #{|placebo| >= |estimate|}) / (1 + B).
"""
import numpy as np, pandas as pd, sys, time
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
import gsc as G
exec(open("b2_estimate.py").read().split('print(f"\\n{\'=\'*104}\\nB2.4')[0])
t0=time.time()
B=1000; R=2; THR=6
def log(m): print(f"[{time.time()-t0:.0f}s] {m}",flush=True)

def draw_block_weighted(cc, nn, target, rng):
    """Countries without replacement, prob proportional to size, until target units."""
    order=rng.choice(len(cc),size=len(cc),replace=False,p=nn/nn.sum())
    picked=[]; tot=0
    for j in order:
        picked.append((cc[j],nn[j])); tot+=nn[j]
        if tot>=target: break
    return picked, tot

def draw_profile(cc, nn, prof, rng):
    """One distinct control country per treated country, descending, taking min(n_c, avail)."""
    avail=dict(zip(cc,nn)); picked=[]
    for n_c in sorted(prof)[::-1]:
        cand=[c for c in avail if avail[c]>0]
        if not cand: break
        # prefer countries that can supply n_c, else the largest remaining
        ok=[c for c in cand if avail[c]>=n_c]
        c=rng.choice(ok) if ok else max(cand,key=lambda c: avail[c])
        picked.append((c,min(n_c,avail[c]))); del avail[c]
    return picked, sum(k for _,k in picked)

ROWS=[]; DIST={}
for KEY in ['lnX','lnS']:
    Y,X,W,ct,se_,s_=build(THR,KEY)
    co=np.where(~W)[0]; ctc=ct[co]
    cc,nn=np.unique(ctc,return_counts=True)
    prof=np.unique(ct[W],return_counts=True)[1]
    ntr=int(W.sum())
    g=G.gsc(Y,X,W,T0,R); tau=g['att'][T0:].mean()
    log(f"{KEY}: actual ATT {tau:+.4f} on {ntr:,} treated in {len(prof)} countries")
    Yc=Y[:,co]; Xc=None if X is None else X[:,co,:]
    for scheme in ['block-weighted','profile-matched','unit-wise']:
        rng=np.random.default_rng(2024); stats=[]; sizes=[]; nclus=[]
        for b in range(B):
            if scheme=='unit-wise':
                sel=rng.choice(len(co),size=ntr,replace=False)
            else:
                picked,tot=(draw_block_weighted(cc,nn,ntr,rng) if scheme=='block-weighted'
                            else draw_profile(cc,nn,prof,rng))
                sel=[]
                for c,k in picked:
                    ix=np.where(ctc==c)[0]
                    sel.extend(rng.choice(ix,size=min(k,len(ix)),replace=False))
                sel=np.array(sel); nclus.append(len(picked))
            fake=np.zeros(len(co),bool); fake[sel]=True
            sizes.append(int(fake.sum()))
            gp=G.gsc(Yc,Xc,fake,T0,R)
            stats.append(gp['att'][T0:].mean())
            if (b+1)%250==0: log(f"  {KEY} {scheme}: {b+1}/{B}")
        a=np.array(stats)
        p2=(1+int((np.abs(a)>=abs(tau)).sum()))/(1+B)
        p1=(1+int((a>=tau).sum()))/(1+B)
        print(f"\n  {KEY}  {scheme}")
        print(f"    placebo groups: mean size {np.mean(sizes):.0f} (actual {ntr:,})"
              +(f", mean clusters {np.mean(nclus):.1f}" if nclus else ", clusters n/a"))
        print(f"    placebo ATT distribution: mean {a.mean():+.4f} sd {a.std(ddof=1):.4f}  "
              f"p05 {np.percentile(a,5):+.4f}  p95 {np.percentile(a,95):+.4f}")
        print(f"    actual {tau:+.4f} sits at the {100*(a<tau).mean():.1f}th percentile")
        print(f"    EXACT p, two-sided {p2:.4f}   one-sided {p1:.4f}   (B = {B})",flush=True)
        ROWS.append(dict(outcome=KEY,scheme=scheme,att=tau,placebo_sd=a.std(ddof=1),
                         placebo_mean=a.mean(),mean_size=np.mean(sizes),
                         mean_clusters=np.mean(nclus) if nclus else np.nan,
                         pct=100*(a<tau).mean(),p_two=p2,p_one=p1))
        DIST[f"{KEY}_{scheme}"]=a
pd.DataFrame(ROWS).to_csv("c1_randomisation.csv",index=False)
np.savez("c1_placebo_dist.npz",**DIST)
log("saved c1_randomisation.csv and c1_placebo_dist.npz")
