"""D1. PLACEBO IN TIME -- the test this project has never run, and the only one that can
invalidate the headline.

The flat pre-period in the main figure is an arithmetic identity, not evidence: GSC Step 2
regresses each treated unit's pre-period on [1, F_pre], so that unit's MEAN pre-period gap is
exactly zero by construction. So we have no evidence that this design does not manufacture
effects. C1 gave a concrete reason to worry -- its country-block placebo distribution is
centred on +0.053 rather than zero.

The test: truncate the panel at 2017 so the real 2018 event is invisible, move the pseudo-event
back, and ask whether a gap opens where none should.

Three variants:
  (a) headline treated set (6+ of 2009-17), pseudo-event 2014. Tests whether the exact units
      behind the result show a spurious gap. Caveat: the treatment window overlaps the
      pseudo-post period, so policy activity in 2014-17 is inside the definition.
  (b) re-defined treated set (3+ of 2009-13, matching the 6-of-9 rate), pseudo-event 2014.
      Treatment measured strictly BEFORE the pseudo-event, mirroring the real design's
      structure. The cleaner placebo.
  (c) headline treated set, pseudo-event 2013, to check sensitivity to the date.

The covariate is rebuilt as China's share in the PRE-pseudo-event window rather than 2015-17,
which lies in the pseudo-post period.

Controls are the same strictly-clean pool as the headline. Inference is the sector-blocked
parametric bootstrap plus the block-weighted randomisation test, so the placebo is judged by
the same instruments as the result.

A clean null here is the strongest single piece of support the note could carry. A positive,
significant placebo would mean the headline is machinery.
"""
import numpy as np, pandas as pd, sys, time
from scipy import stats
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
import gsc as G
exec(open("b2_estimate.py").read().split('print(f"\\n{\'=\'*104}\\nB2.4')[0])
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}",flush=True)
R=2; BB=400; BR=500
XUSv=XUS[YRS].values
pre13=on[[y for y in range(2009,2014)]].sum(axis=1).values   # persistence 2009-13 only

def make(ev, tr_mask, base_win):
    """Panel truncated at 2017 with pseudo-event `ev`; covariate from `base_win`."""
    yrs=[y for y in YRS if y<=2017]
    ii=[YRS.index(y) for y in yrs]
    use=tr_mask|clean
    Y=np.log(XUSv[use][:,ii]).T
    d=np.array([100*CHN.loc[k,base_win].mean()/SK.loc[k,base_win].mean() for _,k in idx])[use]
    pv=np.array([1.0 if y>=ev else 0.0 for y in yrs])
    X=(d[None,:,None]*pv[:,None,None])
    return Y, X, tr_mask[use], CT[use], SE[use], yrs, yrs.index(ev)

def draw_bw(cc, nn, target, rng):
    order=rng.choice(len(cc),size=len(cc),replace=False,p=nn/nn.sum())
    picked=[]; tot=0
    for j in order:
        picked.append((cc[j],nn[j])); tot+=nn[j]
        if tot>=target: break
    return picked

ROWS=[]
SPECS=[('(a) headline set, event 2014', 2014, (pre_yrs>=6), [2011,2012,2013]),
       ('(b) 3+ of 2009-13, event 2014', 2014, (pre13>=3),   [2011,2012,2013]),
       ('(c) headline set, event 2013', 2013, (pre_yrs>=6),  [2010,2011,2012])]
for lab, ev, msk, bw in SPECS:
    Y,X,W,ct_,se_,yrs,t0p=make(ev,msk,bw)
    g=G.gsc(Y,X,W,t0p,R); att=g['att']; a=att[t0p:].mean()
    print(f"\n{'='*96}\nD1 {lab}\n{'='*96}")
    print(f"  panel {Y.shape[1]:,} units ({int(W.sum()):,} treated in "
          f"{len(np.unique(ct_[W]))} countries), {len(yrs)} years, "
          f"pre {yrs[0]}-{yrs[t0p-1]} ({t0p}y), post {yrs[t0p]}-{yrs[-1]} ({len(yrs)-t0p}y)")
    print(f"  PLACEBO ATT {a:+.4f}   (headline, real event, 7 post-years: +0.2337)")
    print(f"  by year: "+" ".join(f"{y}:{att[i]:+.3f}" for i,y in enumerate(yrs)))
    _,sd,A=G.bootstrap(Y,X,W,t0p,R,B=BB,rng=np.random.default_rng(101),max_loo=60,blocks=se_)
    s=A[:,t0p:].mean(axis=1).std(ddof=1); ng=len(np.unique(se_[W]))
    p=2*(1-stats.t.cdf(abs(a/s),df=ng-1))
    print(f"  sector-blocked se {s:.4f}  t {a/s:+.2f}  p {p:.3f}  "
          f"CI [{a-1.96*s:+.3f}, {a+1.96*s:+.3f}]",flush=True)
    # randomisation, block-weighted, on the same truncated panel
    co=np.where(~W)[0]; ctc=ct_[co]; cc,nn=np.unique(ctc,return_counts=True)
    Yc=Y[:,co]; Xc=X[:,co,:]; rng=np.random.default_rng(2024); ps=[]
    for b in range(BR):
        sel=[]
        for c,k in draw_bw(cc,nn,int(W.sum()),rng):
            ix=np.where(ctc==c)[0]; sel.extend(rng.choice(ix,size=min(k,len(ix)),replace=False))
        fk=np.zeros(len(co),bool); fk[np.array(sel)]=True
        ps.append(G.gsc(Yc,Xc,fk,t0p,R)['att'][t0p:].mean())
    ps=np.array(ps)
    pr=(1+int((np.abs(ps)>=abs(a)).sum()))/(1+BR)
    print(f"  randomisation: placebo mean {ps.mean():+.4f} sd {ps.std(ddof=1):.4f}; "
          f"actual at the {100*(ps<a).mean():.1f}th pct; exact p {pr:.4f}",flush=True)
    ROWS.append(dict(spec=lab,event=ev,n_tr=int(W.sum()),att=a,se_sector=s,p_sector=p,
                     rand_mean=ps.mean(),rand_sd=ps.std(ddof=1),p_rand=pr))
    pd.DataFrame(ROWS).to_csv("d1_placebo_time.csv",index=False)
    log(f"  wrote {lab}")
log("done")
