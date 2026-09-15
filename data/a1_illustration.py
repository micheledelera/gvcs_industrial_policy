"""A1 continued. The persistence treatment does not select Vietnam's significant sectors
(only 5 Vietnamese country-sectors are targeted 6+ of 9, the largest just $5m of US
imports), so the single-unit ADH illustration needs a different candidate. Requirements:
persistently targeted, economically meaningful, genuinely exposed to decoupling, and with
enough strictly-clean donors inside its own sector to build a credible synthetic control.
"""
exec(open("a1_treated.py").read().split('print(f"sample:')[0])
import numpy as np, pandas as pd
print(f"sample: {len(idx):,} units\n")
nev=(pre_yrs==0); clean=nev&(post_yrs==0)
tr=(pre_yrs>=6)
sz=XUS[BASE].mean(axis=1).values
share=np.exp(lnS[:,[YRS.index(y) for y in BASE]].mean(axis=1))
nd=np.array([int(((SE==k)&clean).sum()) for _,k in idx])
cand=tr&(dec>=25)&(nd>=20)&(sz>=200e3)
print(f"{'='*104}\nPersistently-targeted units that are large, decoupling-exposed and have "
      f"donors\n  (targeted 6+ of 9 | China >= 25% of US imports in the sector | "
      f">= 20 clean donors in sector | US imports >= $200m)\n{'='*104}")
print(f"  {'country':16s} {'sector':>7s} {'yrs':>4s} {'US imp $m':>10s} {'own share':>10s} "
      f"{'China share':>12s} {'clean donors':>13s}")
for j in np.where(cand)[0][np.argsort(-sz[cand])][:18]:
    print(f"  {nm(CT[j]):16s} {str(SE[j]):>7s} {int(pre_yrs[j]):4d} {sz[j]/1e3:10,.0f} "
          f"{100*share[j]:9.2f}% {dec[j]:11.1f}% {nd[j]:13d}")
print(f"\n  units meeting all four criteria: {int(cand.sum())} of {int(tr.sum())} treated")
print(f"  treated units that are large AND exposed but donor-poor (<20): "
      f"{int((tr&(dec>=25)&(sz>=200e3)&(nd<20)).sum())}")
# how much of treated trade value do Vietnam-style winners represent?
WIN=[704,484,764,458,50]
print(f"\n  for reference, the five 'decoupling winners' (Vietnam, Mexico, Thailand, "
      f"Malaysia, Bangladesh):")
for c in WIN:
    m=(CT==c)
    print(f"    {nm(c):14s} {int(m.sum()):4d} units in sample, {int((m&tr).sum()):3d} "
          f"targeted 6+ of 9, their US imports ${sz[m&tr].sum()/1e6:7.1f}bn of "
          f"${sz[m].sum()/1e6:7.1f}bn total ({100*sz[m&tr].sum()/max(sz[m].sum(),1):.0f}%)")
