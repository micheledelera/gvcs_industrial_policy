"""C1 + C2 + C3 + C6 + C7 for the merged design, in one pass.

Relaunch of c1_randomisation.py with two changes:
  - the full 18-year MEAN GAP PATH of each placebo group is retained, not just the scalar ATT,
    which is what C2 (pre/post RMSPE per placebo) and C3 (their ratio) need;
  - the profile-matched scheme is DROPPED. It could not randomise: the treated size profile
    needs nine country-slots of 75+ units and only eight clean-control countries are that
    large, so nine countries appeared in 100% of draws and ~785 of ~996 placebo-treated units
    were identical every draw. Its p = 0.0080 was an artefact of that, not a result. The
    failure is itself recorded as a finding: the treated countries have no size-matched
    counterpart in the donor universe.

Results are written after EACH scheme so a killed run keeps what it has finished.

On the ratio statistic: SS5c found post/pre RMSPE behaves backwards in this panel, and the GSC
pre-period gap is zero per unit by construction, so at the group level the denominator is
small and unstable. The ratio is therefore reported as a DIAGNOSTIC and as a basis for ADH's
fit filter (C6), not as the inference statistic. C1's ATT comparison remains primary.
"""
import numpy as np, pandas as pd, sys, time, os
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
import gsc as G
exec(open("b2_estimate.py").read().split('print(f"\\n{\'=\'*104}\\nB2.4')[0])
t0=time.time()
B=1000; R=2; THR=6
def log(m): print(f"[{time.time()-t0:.0f}s] {m}",flush=True)
def rmspe(path,sl): return float(np.sqrt((path[sl]**2).mean()))

def draw_block_weighted(cc, nn, target, rng):
    order=rng.choice(len(cc),size=len(cc),replace=False,p=nn/nn.sum())
    picked=[]; tot=0
    for j in order:
        picked.append((cc[j],nn[j])); tot+=nn[j]
        if tot>=target: break
    return picked

ROWS=[]; STORE={}
for KEY in ['lnX','lnS']:
    Y,X,W,ct,se_,s_=build(THR,KEY)
    co=np.where(~W)[0]; ctc=ct[co]
    cc,nn=np.unique(ctc,return_counts=True)
    ntr=int(W.sum())
    g=G.gsc(Y,X,W,T0,R)
    tpath=g['gap'].mean(axis=1)                       # the real group's mean gap path
    tau=tpath[T0:].mean(); t_pre=rmspe(tpath,slice(0,T0)); t_post=rmspe(tpath,slice(T0,None))
    log(f"{KEY}: ATT {tau:+.4f}; treated group pre-RMSPE {t_pre:.4f}, post {t_post:.4f}, "
        f"ratio {t_post/t_pre:.2f}")
    Yc=Y[:,co]; Xc=None if X is None else X[:,co,:]
    for scheme in ['block-weighted','unit-wise']:
        rng=np.random.default_rng(2024); paths=np.zeros((B,len(YRS))); sizes=[]
        for b in range(B):
            if scheme=='unit-wise':
                sel=rng.choice(len(co),size=ntr,replace=False)
            else:
                sel=[]
                for c,k in draw_block_weighted(cc,nn,ntr,rng):
                    ix=np.where(ctc==c)[0]
                    sel.extend(rng.choice(ix,size=min(k,len(ix)),replace=False))
                sel=np.array(sel)
            fake=np.zeros(len(co),bool); fake[sel]=True; sizes.append(int(fake.sum()))
            paths[b]=G.gsc(Yc,Xc,fake,T0,R)['gap'].mean(axis=1)
            if (b+1)%250==0: log(f"  {KEY} {scheme}: {b+1}/{B}")
        a=paths[:,T0:].mean(axis=1)
        pre=np.sqrt((paths[:,:T0]**2).mean(axis=1)); post=np.sqrt((paths[:,T0:]**2).mean(axis=1))
        ratio=post/np.maximum(pre,1e-12)
        p2=(1+int((np.abs(a)>=abs(tau)).sum()))/(1+B)
        ac=a-a.mean(); p2c=(1+int((np.abs(ac)>=abs(tau-a.mean())).sum()))/(1+B)
        print(f"\n{'='*92}\n{KEY}  {scheme}  (B = {B})\n{'='*92}")
        print(f"  C1  placebo ATT: mean {a.mean():+.4f} sd {a.std(ddof=1):.4f}  "
              f"p05 {np.percentile(a,5):+.4f} p95 {np.percentile(a,95):+.4f}; "
              f"mean group size {np.mean(sizes):.0f}")
        print(f"      actual {tau:+.4f} at the {100*(a<tau).mean():.1f}th percentile")
        print(f"      exact p two-sided {p2:.4f};  recentred on the placebo mean {p2c:.4f}")
        print(f"  C2  placebo PRE-RMSPE : median {np.median(pre):.4f}  "
              f"p10 {np.percentile(pre,10):.4f}  p90 {np.percentile(pre,90):.4f}   "
              f"(treated {t_pre:.4f}, at the {100*(pre<t_pre).mean():.0f}th pct)")
        print(f"      placebo POST-RMSPE: median {np.median(post):.4f}  "
              f"p10 {np.percentile(post,10):.4f} p90 {np.percentile(post,90):.4f}   "
              f"(treated {t_post:.4f}, at the {100*(post<t_post).mean():.0f}th pct)")
        print(f"  C3  placebo post/pre ratio: median {np.median(ratio):.2f}  "
              f"p90 {np.percentile(ratio,90):.2f}   (treated {t_post/t_pre:.2f}, at the "
              f"{100*(ratio<t_post/t_pre).mean():.0f}th pct)")
        print(f"      exact p on the RATIO statistic: "
              f"{(1+int((ratio>=t_post/t_pre).sum()))/(1+B):.4f}")
        print(f"  C6  ADH fit filter -- keep only placebos whose own pre-fit is comparable:")
        for mult in [1.0,1.5,2.0,3.0,None]:
            keep=np.ones(B,bool) if mult is None else (pre<=mult*t_pre)
            if keep.sum()<20:
                print(f"        pre <= {mult}x treated: only {keep.sum()} placebos, skipped"); continue
            pk=(1+int((np.abs(a[keep])>=abs(tau)).sum()))/(1+int(keep.sum()))
            lab='no filter' if mult is None else f'pre <= {mult:g}x treated'
            print(f"        {lab:22s} n {int(keep.sum()):5d}  placebo sd {a[keep].std(ddof=1):.4f}"
                  f"  exact p {pk:.4f}")
            ROWS.append(dict(outcome=KEY,scheme=scheme,filter=lab,n=int(keep.sum()),
                             att=tau,placebo_sd=a[keep].std(ddof=1),p=pk))
        print(flush=True)
        STORE[f"{KEY}_{scheme}_paths"]=paths; STORE[f"{KEY}_treated_path"]=tpath
        np.savez("c1c2_placebos.npz",**STORE)
        pd.DataFrame(ROWS).to_csv("c1c2_randomisation.csv",index=False)
        log(f"  wrote {KEY} {scheme}")
log("done")
