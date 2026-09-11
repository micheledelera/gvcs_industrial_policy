"""Route C's central question: does moving to country-sector units actually solve the
convex hull problem, and if so, WHERE?

At country level (§4a) Mexico, Vietnam, India, Malaysia and Thailand are outside the donor
hull in all 11 pre-treatment years. Within a narrow sector the size dispersion is smaller
and there are many suppliers, so the hull should bind less. But it will not bind evenly:
non-policy developing countries barely export electronics or machinery at all, so the
sectors where decoupling mattered most are exactly where donors are thinnest.

If hull feasibility is negatively correlated with decoupling intensity, route C buys
common support by confining the study to sectors where the shock was small -- which would
be a selection problem, not a solution.

Treatment is at the COUNTRY level (policy user vs not), applied to all a country's
sectors, since that is the research question. Outcome: log US exports, 2007-2017.
"""
import pandas as pd, numpy as np, gc
USA, CHINA = 842, 156
PRE=list(range(2007,2018)); BASE=[2015,2016,2017]
NAME={484:'Mexico',699:'India',704:'Vietnam',458:'Malaysia',764:'Thailand',76:'Brazil',
      360:'Indonesia',608:'Philippines',792:'Turkiye',50:'Bangladesh',616:'Poland'}
def nm(c): return NAME.get(int(c),str(int(c)))

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
raw['n_policies']=pd.to_numeric(raw['n_policies'],errors='coerce').fillna(0)
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
us=raw[raw['j']==USA]
X=us.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
X.columns=[int(c) for c in X.columns]; X=X[sorted(X.columns)]
tot=us.groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
tot.columns=[int(c) for c in tot.columns]
chn=us[us['i']==CHINA].groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
chn.columns=[int(c) for c in chn.columns]
chshare=(chn/tot*100)
pol=raw.groupby('i')['n_policies'].max()
del raw,us; gc.collect()

X=X[[c for c in X.columns if c in PRE or c>=2018]]
keep=[(i,k) for i,k in X.index if adv.get(i,1)==0 and i!=CHINA]
X=X.loc[keep]
X=X[(X[PRE]>0).all(axis=1)]
T=pd.Series({ik: bool(pol.get(ik[0],0)>0) for ik in X.index})
print(f"country-sector pairs with a complete 2007-17 US series: {len(X):,}")
print(f"  from policy-using countries : {int(T.sum()):,}")
print(f"  from non-policy countries   : {int((~T).sum()):,}\n")

dec_pre=chshare[BASE].mean(axis=1)
rows=[]
for k,g in X.groupby(level='ISIC4c', observed=True):
    tt=g[T.reindex(g.index).values]; dd=g[~T.reindex(g.index).values]
    if len(tt)==0 or len(dd)<5: continue
    lo,hi=np.log(dd[PRE]).min(axis=0).values, np.log(dd[PRE]).max(axis=0).values
    ins=[]
    for i,_ in tt.index:
        b=np.log(tt.loc[(i,k),PRE].values)
        ins.append(bool((b>=lo).all() and (b<=hi).all()))
    ins=np.array(ins)
    rows.append({'k':k,'n_tr':len(tt),'n_don':len(dd),'share_inside':ins.mean(),
                 'dec_pre':dec_pre.get(k,np.nan),
                 'tr_value':tt[BASE].mean(axis=1).sum(),
                 'val_inside':tt[BASE].mean(axis=1).values[ins].sum()})
r=pd.DataFrame(rows).dropna(subset=['dec_pre'])
r.to_csv("sc02_hull_by_sector.csv",index=False)
print(f"sectors with >=1 treated and >=5 donors: {len(r)} of 125")
print(f"  treated pairs inside their sector's donor hull: "
      f"{100*(r['n_tr']*r['share_inside']).sum()/r['n_tr'].sum():.1f}% of pairs")
print(f"  ...but only {100*r['val_inside'].sum()/r['tr_value'].sum():.1f}% of TREATED TRADE VALUE\n")

print("="*84)
print("IS FEASIBILITY WORSE WHERE DECOUPLING WAS BIGGER?")
print("="*84)
print(f"  corr(share of treated pairs inside hull, China's pre-period US share) = "
      f"{r['share_inside'].corr(r['dec_pre']):+.3f}")
q=r['dec_pre'].quantile([.25,.5,.75])
for lab,sel in [("China share <25th pct", r['dec_pre']<=q.iloc[0]),
                ("25th-50th", (r['dec_pre']>q.iloc[0])&(r['dec_pre']<=q.iloc[1])),
                ("50th-75th", (r['dec_pre']>q.iloc[1])&(r['dec_pre']<=q.iloc[2])),
                ("China share >75th pct", r['dec_pre']>q.iloc[2])]:
    s=r[sel]
    print(f"  {lab:24s} n={len(s):>3}  median China share {s['dec_pre'].median():>5.1f}%   "
          f"treated inside hull {100*(s['n_tr']*s['share_inside']).sum()/s['n_tr'].sum():>5.1f}%   "
          f"donors/sector {s['n_don'].median():>4.0f}")

print("\n"+"="*84)
print("THE BIG EXPORTERS SPECIFICALLY")
print("="*84)
print(f"{'country':<13s}{'sectors':>9s}{'inside hull':>13s}{'% of its US trade inside':>26s}")
for i in [484,704,699,458,764,360,608,76,792,50]:
    sub=[(ii,k) for ii,k in X.index if ii==i]
    if not sub: continue
    ins=0; val=0.0; tot_v=0.0; n=0
    for ii,k in sub:
        g=X.loc[[(a,b) for a,b in X.index if b==k]]
        dd=g[~T.reindex(g.index).values]
        if len(dd)<5: continue
        lo,hi=np.log(dd[PRE]).min(axis=0).values, np.log(dd[PRE]).max(axis=0).values
        b=np.log(X.loc[(ii,k),PRE].values)
        v=X.loc[(ii,k),BASE].mean(); tot_v+=v; n+=1
        if (b>=lo).all() and (b<=hi).all(): ins+=1; val+=v
    if n: print(f"{nm(i):<13s}{n:>9d}{f'{ins}/{n} ({100*ins/n:.0f}%)':>13s}"
                f"{100*val/tot_v if tot_v>0 else 0:>25.1f}%")
