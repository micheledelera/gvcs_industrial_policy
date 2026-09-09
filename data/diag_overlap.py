"""Is there common support? Do non-policy pairs exist that look like policy pairs?

§3x left a residual pre-trend of +0.012 log points/year after SDiD weighting. Two very
different explanations:

  (a) the donor pool CONTAINS good matches but my regularisation (zeta, driven by trade
      volatility) forced near-uniform weights and prevented finding them
  (b) the donor pool DOES NOT contain them -- policy-active pairs were simply growing
      faster than anything available to match them

If (b), no reweighting, no matching, and no alternative design fixes it: it is a
common-support failure, and the honest conclusion is that the comparison cannot be made.
If (a), a smaller zeta or covariate matching rescues it.

Test: compare treated and donor distributions of PRE-period log-export growth
(2010-12 to 2015-17 annualised), overall and within sector x MVA decile -- the cell the
matched design would actually use.
"""
import pandas as pd, numpy as np, gc
exec(open("fit_sdid.py").read().split("def sdid_sector")[0]
     .replace('MEASURE = sys.argv[1] if len(sys.argv)>1 else','MEASURE ='))
C2ISO={699:356,757:756,579:578,251:250,842:840,381:380,490:158,729:729}
XLSX=("/root/.claude/uploads/618110ef-d894-58be-8e00-e7f2bf548b03/"
      "4ad6510b-0154963EF4289D02DEA410D8D38E4E6A8C6D.xlsx")
P0=[2010,2011,2012]; P1=[2015,2016,2017]
u=pd.read_excel(XLSX,sheet_name='Data'); u=u[u['Year'].isin(P1)]
U=u.pivot_table(index='CountryCode',columns='VariableCode',values='Value',aggfunc='mean')
ctry=sorted({i for i,_ in Y.index})
mva=pd.Series(U['NV_IND_MANF'].reindex([C2ISO.get(c,c) for c in ctry]).values,index=ctry)

m=meta.copy()
m['g_pre']=(Y[P1].mean(axis=1)-Y[P0].mean(axis=1))/5.0     # annualised log growth
m['mva']=[mva.get(i,np.nan) for i,_ in m.index]
m=m.dropna(subset=['mva'])
m['mva_dec']=pd.qcut(m['mva'].rank(method='first'),10,labels=False)
tr=m[m['T']==1]; dn=m[m['T']==0]
print(f"treated {len(tr):,}   donors {len(dn):,}\n")

print("="*74); print("PRE-PERIOD ANNUALISED LOG-EXPORT GROWTH, 2010-12 -> 2015-17"); print("="*74)
print(f"{'':10s}{'mean':>9s}{'sd':>9s}{'p10':>9s}{'p25':>9s}{'median':>9s}{'p75':>9s}{'p90':>9s}")
for lab,g in [("treated",tr),("donors",dn)]:
    q=g['g_pre'].quantile([.1,.25,.5,.75,.9])
    print(f"{lab:10s}{g['g_pre'].mean():>9.4f}{g['g_pre'].std():>9.4f}"
          f"{q[.1]:>9.4f}{q[.25]:>9.4f}{q[.5]:>9.4f}{q[.75]:>9.4f}{q[.9]:>9.4f}")
d=tr['g_pre'].mean()-dn['g_pre'].mean()
sp=np.sqrt((tr['g_pre'].var()+dn['g_pre'].var())/2)
print(f"\n  raw difference {d:+.4f}/yr   standardised (Cohen's d) {d/sp:+.3f}")

print("\n"+"="*74)
print("OVERLAP: where does each treated pair sit in its own cell's donor distribution?")
print("="*74)
res=[]
for (k,dc),g in m.groupby(['ISIC4c','mva_dec'], observed=True):
    t_=g[g['T']==1]; d_=g[g['T']==0]
    if len(t_)==0 or len(d_)<3: continue
    for _,row in t_.iterrows():
        pct=(d_['g_pre']<row['g_pre']).mean()
        res.append({'k':k,'dec':dc,'pct':pct,'n_don':len(d_),
                    'inside':(row['g_pre']>=d_['g_pre'].min())&(row['g_pre']<=d_['g_pre'].max())})
res=pd.DataFrame(res)
print(f"treated pairs in a sector x MVA-decile cell with >=3 donors: {len(res):,} "
      f"of {len(tr):,} ({100*len(res)/len(tr):.0f}%)")
print(f"  inside the donor convex hull (min-max) : {100*res['inside'].mean():.1f}%")
print(f"  percentile within donors: mean {res['pct'].mean():.3f}  median {res['pct'].median():.3f}")
print(f"  above the donor MEDIAN  : {100*(res['pct']>0.5).mean():.1f}%")
print(f"  above the donor 90th pct: {100*(res['pct']>0.9).mean():.1f}%")
print(f"  below the donor 10th pct: {100*(res['pct']<0.1).mean():.1f}%")

print("\n"+"="*74); print("SAME, without the MVA decile restriction (sector only)"); print("="*74)
res2=[]
for k,g in m.groupby(level='ISIC4c', observed=True):
    t_=g[g['T']==1]; d_=g[g['T']==0]
    if len(t_)==0 or len(d_)<3: continue
    for _,row in t_.iterrows():
        res2.append({'pct':(d_['g_pre']<row['g_pre']).mean()})
res2=pd.DataFrame(res2)
print(f"  percentile within donors: mean {res2['pct'].mean():.3f}  median {res2['pct'].median():.3f}")
print(f"  above the donor median: {100*(res2['pct']>0.5).mean():.1f}%")
