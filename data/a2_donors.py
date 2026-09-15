"""A2. The donor pool. Under the merge, GSC uses every control unit, so this step is not
"which donors per treated unit" but "which units are legitimate controls at all", plus the
requirement carried forward from A0: donors must face the SAME decoupling shock, or the
counterfactual becomes "no policy AND no decoupling".

Settled in A1: controls = strictly clean (never targeted 2009-2024) -> 2,223.

Open questions this quantifies:
  1  EXPOSURE OVERLAP. Do treated and control units sit in sectors with comparable Chinese
     import shares? If controls are systematically low-exposure, pooling them is the A0 trap.
  2  SECTOR-BY-SECTOR FEASIBILITY. Because the outcome is lnS (share within a sector),
     donors in the SAME sector automatically share the treated unit's exposure. Running GSC
     per sector would hold exposure exactly fixed -- but Xu cautions below N_co = 40.
  3  ADVANCED ECONOMIES as donors -- excluded so far. They also compete for US demand.
"""
exec(open("a1_treated.py").read().split('print(f"sample:')[0])
import numpy as np, pandas as pd
nev=(pre_yrs==0); clean=nev&(post_yrs==0); tr=(pre_yrs>=6)
sz=XUS[BASE].mean(axis=1).values
print(f"sample {len(idx):,} | treated {int(tr.sum()):,} | clean controls {int(clean.sum()):,}\n")

print(f"{'='*100}\n1. EXPOSURE OVERLAP: China's share of US imports in the unit's sector, "
      f"2015-17\n{'='*100}")
for lab,m in [("treated (6+ of 9)",tr),("clean controls",clean)]:
    v=dec[m]
    print(f"  {lab:20s} n={int(m.sum()):5d}  mean {v.mean():5.1f}%  "
          f"p10 {np.percentile(v,10):5.1f}%  p50 {np.percentile(v,50):5.1f}%  "
          f"p90 {np.percentile(v,90):5.1f}%")
q=np.percentile(dec[tr],[10,90])
frac=np.mean((dec[clean]>=q[0])&(dec[clean]<=q[1]))
print(f"  -> {100*frac:.0f}% of clean controls sit inside the treated units' p10-p90 "
      f"exposure range ({q[0]:.1f}%-{q[1]:.1f}%)")
hi=dec>=25
print(f"  in sectors with China >= 25%: treated {int((tr&hi).sum()):,} of {int(tr.sum()):,} "
      f"({100*(tr&hi).sum()/tr.sum():.0f}%), clean controls {int((clean&hi).sum()):,} of "
      f"{int(clean.sum()):,} ({100*(clean&hi).sum()/clean.sum():.0f}%)")

print(f"\n{'='*100}\n2. SECTOR-BY-SECTOR FEASIBILITY (exposure held exactly fixed by "
      f"construction)\n{'='*100}")
rows=[]
for k in sorted(set(SE)):
    m=(SE==k); rows.append((k,int((m&tr).sum()),int((m&clean).sum()),
                            float(dec[m][0]), sz[m&tr].sum()/1e3))
R=pd.DataFrame(rows,columns=['sector','treated','donors','dec','usd_m'])
for nd in [40,30,20,10]:
    ok=R[(R.treated>=1)&(R.donors>=nd)]
    print(f"  sectors with >=1 treated and >={nd:3d} clean donors: {len(ok):3d} of {len(R)}"
          f"   covering {ok.treated.sum():5d} treated units "
          f"({100*ok.treated.sum()/R.treated.sum():3.0f}%) and "
          f"${ok.usd_m.sum()/1e3:7.1f}bn of treated US imports "
          f"({100*ok.usd_m.sum()/R.usd_m.sum():3.0f}%)")
print(f"\n  Xu cautions below N_co = 40. At that bar the per-sector design keeps "
      f"{len(R[(R.treated>=1)&(R.donors>=40)])} sectors.")
top=R[(R.treated>=3)&(R.donors>=40)].nlargest(10,'usd_m')
print(f"\n  largest such sectors (>=3 treated, >=40 donors):")
print(f"  {'sector':>7s} {'treated':>8s} {'donors':>7s} {'China share':>12s} "
      f"{'treated US imports $bn':>23s}")
for r in top.itertuples():
    print(f"  {str(r.sector):>7s} {r.treated:8d} {r.donors:7d} {r.dec:11.1f}% "
          f"{r.usd_m/1e3:23.2f}")

print(f"\n{'='*100}\n3. ADVANCED ECONOMIES as additional donors\n{'='*100}")
raw2=pd.read_pickle("agg_for_estimation.pkl")
raw2['i']=raw2['i'].astype('int32'); raw2['j']=raw2['j'].astype('int32')
raw2['t']=raw2['t'].astype('int32')
u2=raw2[raw2['j']==USA]
A=u2.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
A.columns=[int(c) for c in A.columns]; A=A.reindex(columns=YRS)
advset=[i for i in A.index.get_level_values(0).unique() if adv.get(i,0)==1]
Aa=A.loc[[(i,k) for i,k in A.index if adv.get(i,0)==1]]
Aa=Aa[(Aa[YRS]>0).all(axis=1) & Aa[YRS].notna().all(axis=1)]
print(f"  advanced-economy country-sectors with a balanced positive US series: "
      f"{len(Aa):,} (vs {int(clean.sum()):,} clean developing controls)")
print(f"  they would nearly double the donor pool, but they run their own industrial policy"
      f"\n  (CHIPS Act, EU programmes) and are not 'never exposed' in any meaningful sense,"
      f"\n  so they are kept OUT of the primary pool and offered as a robustness only.")
