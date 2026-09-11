"""Is US-China decoupling a staggered event across sectors, or one common date?

The whole staggered design hinges on this. Four candidate definitions of sector k's
event year E_k, all computed on China's share of TOTAL US imports in sector k:

  A. peak       year of the maximum share, after which it declines
  B. maxdrop    year of the largest one-year fall in share (pp)
  C. rel10      first year the share is >=10% below its own 2015-17 mean, and stays below
  D. rel25      same at 25%

Restricted to sectors where China had meaningful presence -- a 2015-17 mean share of at
least 5pp -- since a share that starts at 0.3pp produces noise, not events.
"""
import pandas as pd, numpy as np, gc
USA, CHINA = 842, 156
BASE=[2015,2016,2017]

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
us=raw[raw['j']==USA]
x=us.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum()
tot=x.groupby(level=['ISIC4c','t'],observed=True).transform('sum')
s=(x/tot.replace(0,np.nan)*100)
chn=s.xs(CHINA,level='i').unstack('t')      # sector x year, pp
chn.columns=[int(c) for c in chn.columns]
chn=chn.reindex(columns=sorted(chn.columns))
del raw,us; gc.collect()

base=chn[BASE].mean(axis=1)
keep=base[base>=5.0].index
C=chn.loc[keep]; B=base.loc[keep]
yrs=[c for c in C.columns if c>=2012]
C=C[yrs]
print(f"{len(C)} of {len(chn)} sectors have a 2015-17 China share >= 5pp "
      f"(mean {B.mean():.1f}pp, max {B.max():.1f}pp)\n")

ev={}
ev['A. peak']      = C.idxmax(axis=1)
ev['B. maxdrop']   = C.diff(axis=1).idxmin(axis=1)
for lab,thr in [('C. rel10',0.10),('D. rel25',0.25)]:
    below = C.lt(B*(1-thr), axis=0)
    # first year below that is also below in every later year
    stay = below[::1].iloc[:, ::-1].cummin(axis=1).iloc[:, ::-1].astype(bool)
    ev[lab] = stay.apply(lambda r: r.index[r.values.argmax()] if r.any() else np.nan, axis=1)

out=pd.DataFrame(ev)
for c in out.columns:
    v=out[c].dropna()
    print(f"--- {c}   defined for {len(v)}/{len(C)} sectors")
    tab=v.value_counts().sort_index()
    for y,n in tab.items():
        print(f"      {int(y)}  {n:>3}  {'#'*int(round(40*n/len(v)))}")
    if len(v)>1:
        print(f"      modal year {int(tab.idxmax())} holds {100*tab.max()/len(v):.0f}%   "
              f"sd {v.std():.2f} years   IQR {v.quantile(.25):.0f}-{v.quantile(.75):.0f}")
    print()
out.to_csv("event_timing.csv")
print("saved event_timing.csv")
