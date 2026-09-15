"""§5e. The decisive check: do Vietnam's LOW-policy sectors also beat their synthetics?

§5d found Vietnam's ten high-targeting sectors average tau_aug +1.058 against synthetic
controls built from low-policy sectors in other countries. Vietnam grew faster than almost
everything in this period, so that number is only about TARGETING if Vietnam's UNTARGETED
sectors do not show the same thing. If they do, the design is measuring Vietnam.

Run Vietnam's low-policy sectors through the identical machinery -- same donor rule (LOW
policy, DIFFERENT country, Chinese US share within 10pp, log size within 2.0, 20 nearest on
MVA/GDP, log MVA per capita, ECI, export share), same weights, same placebo distribution.

Then a within-Vietnam permutation test: shuffle the high/low labels across Vietnam's own
sectors 5,000 times and ask how often the reshuffled difference in mean tau_aug exceeds the
real one. That tests targeting against a Vietnam-wide effect directly, using only Vietnam's
sectors, so a common country-level shock cancels.

The same high/low split is reported for the other large exporters as context.
"""
import pandas as pd, numpy as np, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])
VN=704
v=M['tgt'].values; pos=v[v>0]
q75,q25=np.quantile(pos,.75),np.quantile(pos,.25)
hi=np.where(v>=q75)[0]; lo=np.where(v<=q25)[0]

pl=rng.choice(lo,size=min(NPLAC,len(lo)),replace=False)
P=[]
for a in pl:
    r=one(a,lo[lo!=a])
    if r and np.isfinite(r['ratio']) and np.isfinite(r['tau_aug']): P.append(r)
ppre=np.array([x['pre'] for x in P]); ptau=np.abs([x['tau_aug'] for x in P])
log(f"{len(P)} placebos, median pre-RMSPE {np.median(ppre):.3f}")

def est(a):
    """identical to §5d, donors always drawn from the LOW pool of OTHER countries."""
    r=one(a, lo[lo!=a])
    if not r or not np.isfinite(r['tau_aug']): return None
    m=ppre<=2.0*r['pre']
    r['p_aug']=(1+(ptau[m]>=abs(r['tau_aug'])).sum())/(1+int(m.sum())) if m.sum()>=5 else np.nan
    return r

rows=[]
for a in range(len(M)):
    if CT[a]!=VN: continue
    grp = 'HIGH' if a in set(hi) else ('LOW' if a in set(lo) else 'mid')
    if grp=='mid': continue
    r=est(a)
    if r: rows.append(dict(k=str(SE[a]),grp=grp,ip=v[a],Dec=Dv[a],pre=r['pre'],
                           gap_pre=r['gap_pre'],tau=r['tau_scm'],tau_aug=r['tau_aug'],
                           p_aug=r['p_aug']))
V=pd.DataFrame(rows); V.to_csv("sc13_vnplacebo.csv",index=False)

print(f"\n{'='*92}\nVIETNAM: high-targeting vs low-targeting sectors, identical machinery\n{'='*92}")
print(f"{'group':>8s}{'n':>5s}{'median pre-RMSPE':>19s}{'mean gap pre':>14s}"
      f"{'mean tau_aug':>14s}{'median tau_aug':>16s}{'p<.10':>9s}")
for g in ['HIGH','LOW']:
    s=V[V['grp']==g]; ok=s['p_aug'].notna()
    print(f"{g:>8s}{len(s):>5d}{s['pre'].median():>19.3f}{s['gap_pre'].mean():>+14.3f}"
          f"{s['tau_aug'].mean():>+14.3f}{s['tau_aug'].median():>+16.3f}"
          f"{f'{int((s[ok].p_aug<.10).sum())}/{int(ok.sum())}':>9s}")
H=V[V['grp']=='HIGH']['tau_aug'].values; Lw=V[V['grp']=='LOW']['tau_aug'].values
diff=H.mean()-Lw.mean()
print(f"\n  HIGH - LOW difference in mean tau_aug: {diff:+.3f}")

# within-Vietnam permutation test
allv=np.concatenate([H,Lw]); nH=len(H)
perm=np.array([ (lambda s: s[:nH].mean()-s[nH:].mean())(rng.permutation(allv))
                for _ in range(5000)])
p_perm=float((np.abs(perm)>=abs(diff)).mean())
print(f"  within-Vietnam permutation test (5,000 draws, labels shuffled across Vietnam's")
print(f"  own sectors so any country-wide shock cancels):  p = {p_perm:.4f}")
print(f"    permutation distribution: mean {perm.mean():+.3f}, sd {perm.std():.3f}, "
      f"95% range [{np.quantile(perm,.025):+.3f}, {np.quantile(perm,.975):+.3f}]")

print(f"\n{'='*92}\nCONTEXT: the same split for other large exporters\n{'='*92}")
print(f"{'country':<15s}{'n HIGH':>8s}{'n LOW':>7s}{'mean tau HIGH':>15s}{'mean tau LOW':>14s}"
      f"{'difference':>12s}")
for c in [484,699,704,458,764,76,360,608,792,50]:
    rs=[]
    for a in range(len(M)):
        if CT[a]!=c: continue
        g='HIGH' if a in set(hi) else ('LOW' if a in set(lo) else None)
        if not g: continue
        r=est(a)
        if r: rs.append((g,r['tau_aug']))
    if len(rs)<12: continue
    h=[t for g,t in rs if g=='HIGH']; l=[t for g,t in rs if g=='LOW']
    if len(h)<3 or len(l)<5: continue
    print(f"{nm(c):<15s}{len(h):>8d}{len(l):>7d}{np.mean(h):>+15.3f}{np.mean(l):>+14.3f}"
          f"{np.mean(h)-np.mean(l):>+12.3f}")
log("DONE")
