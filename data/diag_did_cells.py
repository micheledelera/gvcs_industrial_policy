"""The DiD as raw cell means, before regression.

Outcome windows: pre = 2015-17, post = 2022-24, shares of total US imports in sector k.
Treatment: IP = share_frac_policies > 0 in 2015-17.  Exposure: Dec_k above/below the
weighted median Chinese share.  All means weighted by pre-period US imports.

Panel 1 gives the raw 2x2x2 -- the arithmetic the regression is a smoothed version of.
Panel 2 repeats it after removing the sector mean change (the alpha_k the regression
uses), which is the comparison the spec actually makes.
"""
import pandas as pd, numpy as np, sys
src=open("fit_longdiff_iso.py").read().split('lev=" + ".join(CTRL)')[0]
src=src.replace("CTRL=sys.argv[1].split(',') if len(sys.argv)>1 else ['mva_gdp']","CTRL=['mva_gdp']")
src=src.replace("SAMP=sys.argv[2].split(',') if len(sys.argv)>2 else CTRL","SAMP=CTRL")
exec(src)

M='share_frac_policies'
d['T']=(d[M]>0).astype(int)
med=np.average(d['Dec'],weights=d['w'])
dk=d.groupby('ISIC4c',observed=True)['Dec'].first()
cut=dk.median()
d['E']=(d['Dec']>cut).astype(int)
def wm(x,w): return np.average(x,weights=w) if w.sum()>0 else np.nan

print(f"treatment: {M} > 0   |   exposure: Dec_k > median ({cut:.1f}pp)")
print(f"cells: {len(d):,}   treated {d['T'].sum():,} ({100*d['T'].mean():.1f}%)   "
      f"high-exposure {d['E'].sum():,} ({100*d['E'].mean():.1f}%)\n")

for panel,col in [("PANEL 1 -- raw", 'd_post'),
                  ("PANEL 2 -- after removing the sector mean change (alpha_k)", 'resid')]:
    if col=='resid':
        skm=d.groupby('ISIC4c',observed=True).apply(
            lambda g: np.average(g['d_post'],weights=g['w']), include_groups=False)
        d['resid']=d['d_post']-d['ISIC4c'].astype(str).map({str(k):v for k,v in skm.items()})
    print("="*78); print(panel); print("="*78)
    print(f"mean change in share of US imports, pp   (n cells / share of weight)")
    print(f"{'':22s}{'IP = 0':>26s}{'IP > 0':>26s}{'difference':>18s}")
    dif={}
    for e,lab in [(1,"HIGH exposure"),(0,"LOW exposure")]:
        row=f"{lab:22s}"; v={}
        for t in [0,1]:
            g=d[(d['E']==e)&(d['T']==t)]
            v[t]=wm(g[col],g['w'])
            row+=f"{v[t]:+8.4f}  ({len(g):>5,} / {100*g['w'].sum()/d['w'].sum():4.1f}%)".rjust(26)
        dif[e]=v[1]-v[0]
        row+=f"{dif[e]:+10.4f}".rjust(18)
        print(row)
    print(f"{'':22s}{'':26s}{'':26s}{'-'*12:>18s}")
    print(f"{'DDD (high - low)':22s}{'':26s}{'':26s}{dif[1]-dif[0]:+10.4f}".rjust(18+74))
    # pooled DiD ignoring exposure
    g1=d[d['T']==1]; g0=d[d['T']==0]
    print(f"\n  pooled DiD, ignoring exposure: {wm(g1[col],g1['w'])-wm(g0[col],g0['w']):+.4f}")
    print()
