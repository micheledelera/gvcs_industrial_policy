"""The clustering gap as a RESULT, not a nuisance. If the ATT's variance is dominated by
between-country variation, then the unit of analysis for the research question is the country,
not the country-sector. Three things:
  1. variance decomposition of the unit-level gaps, between vs within country, and the same
     for sector -- run separately on the PRE and POST periods, because a between-country share
     that is already high before 2018 is baseline heterogeneity, not heterogeneous effects;
  2. the per-country ATT distribution;
  3. the Moulton factor implied by the residual ICC against the factor the bootstrap actually
     delivers, to see how much of the country-block SE is error correlation and how much is
     genuine effect dispersion.
"""
import numpy as np, pandas as pd, sys, time
sys.path.insert(0,"/home/user/gvcs_industrial_policy/data")
import gsc as G
exec(open("b2_estimate.py").read().split('print(f"\\n{\'=\'*104}\\nB2.4')[0])
t0=time.time()
Y,X,W,ct,se_,s_=build(6,'lnS')
g=G.gsc(Y,X,W,T0,2)
gap=g['gap']                      # T x N_treated
ctw=ct[W]; sew=se_[W]; divw=np.array([s[:2] for s in se_[W]])
def decomp(v, grp):
    """Share of the cross-unit variance of v that lies BETWEEN groups (one-way ANOVA)."""
    d=pd.DataFrame(dict(v=v,g=pd.factorize(grp)[0]))
    gm=d.groupby('g')['v'].transform('mean')
    sb=((gm-d.v.mean())**2).sum(); sw=((d.v-gm)**2).sum()
    return sb/(sb+sw)
print(f"\n{'='*100}\n1. VARIANCE DECOMPOSITION OF THE UNIT-LEVEL GAPS\n{'='*100}")
print(f"  {'window':14s} {'sd of unit gaps':>16s} {'between COUNTRY':>17s} {'between SECTOR':>16s}"
      f" {'between DIVISION':>18s}")
for lab,sl in [('pre 2007-17',slice(0,T0)),('post 2018-24',slice(T0,None)),
               ('2018-19 only',slice(T0,T0+2)),('2020-24 only',slice(T0+2,None))]:
    v=gap[sl].mean(axis=0)
    print(f"  {lab:14s} {v.std():16.3f} {decomp(v,ctw):16.1%} {decomp(v,sew):15.1%}"
          f" {decomp(v,divw):17.1%}")
print("\n  Read: if the between-country share JUMPS from pre to post, the country effect is in")
print("  the TREATMENT RESPONSE. If it is already high pre-2018, it is baseline heterogeneity")
print("  that the design has failed to balance at country level.")

print(f"\n{'='*100}\n2. PER-COUNTRY ATT, post 2018-24\n{'='*100}")
R=[]
for c in np.unique(ctw):
    m=ctw==c
    R.append(dict(country=nm(c),n=int(m.sum()),att=gap[T0:][:,m].mean(),
                  pre=gap[:T0][:,m].mean(),
                  val=float(s_[W][m].sum())/1e6))
D=pd.DataFrame(R).sort_values('att',ascending=False)
print(f"  {'country':16s} {'n':>5s} {'pre gap':>9s} {'ATT':>8s} {'2015-17 $bn':>12s}")
for _,r in D.iterrows():
    print(f"  {r.country:16s} {r.n:5d} {r.pre:+9.3f} {r.att:+8.3f} {r.val:12.2f}")
w=D.n/D.n.sum()
print(f"\n  unweighted mean of per-country ATTs {D.att.mean():+.4f}  sd {D.att.std():.4f}")
print(f"  unit-weighted mean                  {(D.att*w).sum():+.4f}   (= the pooled ATT)")
vw=D.val/D.val.sum()
print(f"  TRADE-VALUE-weighted mean           {(D.att*vw).sum():+.4f}   <- the dollar estimand")
print(f"  countries with ATT > 0: {int((D.att>0).sum())} of {len(D)}   "
      f"sign test p = {2*min(sum(D.att>0),sum(D.att<0))/len(D):.3f} (binomial, crude)")
print(f"\n  implied se of the mean from between-country dispersion alone: "
      f"{D.att.std()/np.sqrt(len(D)):.4f}")
print(f"  the country-blocked bootstrap se was 0.2383 (§10f); unit-level 0.0972")
D.to_csv("b2_countryeffect.csv",index=False)
print(f"[{time.time()-t0:.0f}s] saved b2_countryeffect.csv")
