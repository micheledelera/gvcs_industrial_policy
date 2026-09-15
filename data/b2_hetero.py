"""B2.7 heterogeneity, plus the checks §10b demanded and protocol D2.

GSC returns a per-unit effect in one run, so all of this comes from a single fit:
  ATT by country, by sector, by SIZE quintile (§10b's size worry)
  ATT excluding the 299 units that overlap the share_frac_policies definition §5k voided
  concentration of the estimate -- §5k's lesson, is it a handful of units again?
  leave-one-country-out (protocol D2; with 2,223 controls, leave-one-DONOR-out is meaningless)
"""
import numpy as np, pandas as pd, sys, time
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
import gsc as G
exec(open("b2_estimate.py").read().split('print(f"\\n{\'=\'*104}\\nB2.4')[0])
t0=time.time()
KEY,R,THR='lnS',2,6
Y,X,W,ct,se_,s_=build(THR,KEY)
g=G.gsc(Y,X,W,T0,R)
gap=g['gap']                      # T x N_tr
per=gap[T0:].mean(axis=0)         # post-2018 mean effect per treated unit
ctT=ct[W]; seT=se_[W]; szT=s_[W]
print(f"treated {len(per):,} units, {len(set(ctT))} countries, {len(set(seT))} sectors")
print(f"pooled post-2018 ATT = {per.mean():+.4f}\n")

def jk_country(v,c):
    cs=np.unique(c); m=v.mean()
    a=np.array([v[c!=x].mean() for x in cs])
    return m, np.sqrt((len(cs)-1)/len(cs)*((a-a.mean())**2).sum())

print(f"{'='*100}\nCONCENTRATION -- is the estimate a handful of units again (§5k's lesson)?"
      f"\n{'='*100}")
o=np.argsort(-np.abs(per))
for f in [0.01,0.05,0.10,0.25]:
    n=max(1,int(f*len(per)))
    keep=np.ones(len(per),bool); keep[o[:n]]=False
    print(f"  drop the {100*f:4.0f}% most influential treated units (n={n:4d}): "
          f"ATT {per[keep].mean():+.4f}")
sh=np.abs(per)/np.abs(per).sum()
print(f"  Kish effective n of the treated units: {1/(sh**2).sum():.0f} of {len(per):,}")
print(f"  top 15 units hold {100*np.sort(sh)[::-1][:15].sum():.1f}% of the absolute effect")

print(f"\n{'='*100}\nBY SIZE QUINTILE (§10b: persistence correlates +0.331 with log size)"
      f"\n{'='*100}")
ln=np.log(szT)
for lo,hi,lab in [(0,20,'smallest'),(20,40,'2nd'),(40,60,'3rd'),(60,80,'4th'),(80,100,'largest')]:
    a,b=np.percentile(ln,[lo,hi]); m=(ln>=a)&(ln<=b)
    mu,s=jk_country(per[m],ctT[m])
    print(f"  {lab:10s} n={int(m.sum()):4d}  ATT {mu:+.4f} (jk se {s:.4f}, t={mu/s:+5.2f})")

print(f"\n{'='*100}\nEXCLUDING the §5k-voided-measure overlap\n{'='*100}")
pos=SF[SF>0]; q=np.quantile(pos,.75)
use=(pre_yrs>=THR)|clean
hi_sf=(SF[use]>=q)[W]
for lab,m in [("all treated",np.ones(len(per),bool)),
              ("excluding share_frac top quartile",~hi_sf),
              ("only share_frac top quartile",hi_sf)]:
    if m.sum()<20: continue
    mu,s=jk_country(per[m],ctT[m])
    print(f"  {lab:34s} n={int(m.sum()):4d}  ATT {mu:+.4f} (jk se {s:.4f}, t={mu/s:+5.2f})")

print(f"\n{'='*100}\nBY COUNTRY (countries with >= 15 treated units)\n{'='*100}")
rows=[]
for c in np.unique(ctT):
    m=(ctT==c)
    if m.sum()>=15: rows.append((nm(c),int(m.sum()),per[m].mean()))
rows.sort(key=lambda z:-z[2])
print(f"  {'country':16s} {'n':>5s} {'ATT':>9s}      {'country':16s} {'n':>5s} {'ATT':>9s}")
half=(len(rows)+1)//2
for a in range(half):
    L=f"  {rows[a][0]:16s} {rows[a][1]:5d} {rows[a][2]:+9.4f}"
    Rr=(f"      {rows[a+half][0]:16s} {rows[a+half][1]:5d} {rows[a+half][2]:+9.4f}"
        if a+half<len(rows) else "")
    print(L+Rr)

print(f"\n{'='*100}\nLEAVE-ONE-COUNTRY-OUT (protocol D2), countries with >= 15 treated\n{'='*100}")
big=[c for c in np.unique(ctT) if (ctT==c).sum()>=15]
loo=[(nm(c),per[ctT!=c].mean()) for c in big]
loo.sort(key=lambda z:z[1])
print(f"  pooled {per.mean():+.4f};  range across leave-one-out: "
      f"{loo[0][1]:+.4f} (drop {loo[0][0]}) to {loo[-1][1]:+.4f} (drop {loo[-1][0]})")
print("   "+"   ".join(f"{k}:{v:+.3f}" for k,v in loo[:6]))
pd.DataFrame(dict(country=[nm(c) for c in ctT],sector=[str(x) for x in seT],
                  att=per,lnsize=ln)).to_csv("b2_hetero.csv",index=False)
print(f"\n[{time.time()-t0:.0f}s] done")
