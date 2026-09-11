"""NEW DESIGN, STEP 0. Feasibility of a country-SECTOR synthetic control comparing HIGH
against LOW industrial policy, within similar decoupling exposure and similar size.

The question: does Vietnam's policy-targeted sector do better than comparable sectors in
both Vietnam AND Bangladesh, once decoupling hits?

Why this should work where Routes A and B failed. There is no untreated Vietnam (§4a), but
there are plenty of UNTARGETED VIETNAMESE SECTORS. Making the unit a country-sector puts
them in the donor pool, so the hull no longer needs an untreated economy of Vietnam's size.
It compares HIGH vs LOW rather than ANY vs NONE, which §4i showed is where the variation
is. And because donors are selected on similar DECOUPLING EXPOSURE rather than same sector,
the treated unit is not competing against its own counterfactual, weakening §3x's SUTVA
problem.

  unit       country-sector (i,k), developing ex-China, complete positive US series 2007-17
  Dec_k      China's share of US imports in sector k, 2015-17 mean
  treatment  HIGH = top quartile of the positive-IP distribution
             LOW  = bottom quartile of positives, or zero
             measures: n_policies (VOLUME), share_frac_policies (TARGETING)
  donors     LOW IP, |Dec - Dec_treated| <= DEC_BAND, |log size - log size_treated| <=
             SIZE_BAND. Same country or any other -- that is the point.
"""
import pandas as pd, numpy as np, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
USA, CHINA = 842, 156
PRE=list(range(2007,2018)); BASE=[2015,2016,2017]
NAME={484:'Mexico',699:'India',704:'Vietnam',458:'Malaysia',764:'Thailand',76:'Brazil',
 360:'Indonesia',608:'Philippines',792:'Turkiye',50:'Bangladesh',616:'Poland',348:'Hungary',
 643:'Russia',152:'Chile',710:'South Africa',604:'Peru',586:'Pakistan',32:'Argentina',
 170:'Colombia',642:'Romania',818:'Egypt',504:'Morocco',100:'Bulgaria',191:'Croatia',
 144:'Sri Lanka',116:'Cambodia',214:'Dominican Rep',188:'Costa Rica',222:'El Salvador',
 340:'Honduras',558:'Nicaragua',320:'Guatemala',400:'Jordan',748:'Eswatini',686:'Senegal',
 288:'Ghana',404:'Kenya',834:'Tanzania',508:'Mozambique',524:'Nepal',418:'Laos',
 104:'Myanmar',862:'Venezuela',600:'Paraguay',68:'Bolivia',218:'Ecuador',591:'Panama',
 858:'Uruguay',780:'Trinidad&Tob',388:'Jamaica',328:'Guyana',12:'Algeria',788:'Tunisia',
 566:'Nigeria',688:'Serbia',804:'Ukraine',112:'Belarus',860:'Uzbekistan',417:'Kyrgyzstan'}
def nm(c): return NAME.get(int(c),str(int(c)))

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
for c in ['n_policies','share_frac_policies']:
    raw[c]=pd.to_numeric(raw[c],errors='coerce').fillna(0)
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
us=raw[raw['j']==USA]
X=us.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
X.columns=[int(c) for c in X.columns]; X=X[sorted(X.columns)]
tot_k=us.groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
chn_k=us[us['i']==CHINA].groupby(['ISIC4c','t'],observed=True)['imports'].sum().unstack('t')
tot_k.columns=[int(c) for c in tot_k.columns]; chn_k.columns=[int(c) for c in chn_k.columns]
DEC=(chn_k[BASE].mean(axis=1)/tot_k[BASE].mean(axis=1)*100)
ip=(raw[raw['t'].isin(BASE)][['i','ISIC4c','t','n_policies','share_frac_policies']]
    .drop_duplicates(subset=['i','ISIC4c','t'])
    .groupby(['i','ISIC4c'],observed=True)[['n_policies','share_frac_policies']].mean())
del raw,us; gc.collect()

X=X.loc[[(i,k) for i,k in X.index if adv.get(i,1)==0 and i!=CHINA]]
X=X[(X[PRE]>0).all(axis=1)]
L=np.log(X)
M=pd.DataFrame(index=X.index)
M['vol']=ip['n_policies'].reindex(X.index).fillna(0.0).values
M['tgt']=ip['share_frac_policies'].reindex(X.index).fillna(0.0).values
M['Dec']=[DEC.get(k,np.nan) for _,k in X.index]
M['lsize']=L[BASE].mean(axis=1).values
M=M.dropna(subset=['Dec']); L=L.loc[M.index]
log(f"country-sector units, complete 2007-17 US series: {len(M):,}  "
    f"({M.index.get_level_values('i').nunique()} countries, "
    f"{M.index.get_level_values('ISIC4c').nunique()} sectors)")
print(f"  Dec_k   median {M['Dec'].median():.1f}%   IQR {M['Dec'].quantile(.25):.1f}-{M['Dec'].quantile(.75):.1f}")
print(f"  logsize median {M['lsize'].median():.2f}    IQR {M['lsize'].quantile(.25):.2f}-{M['lsize'].quantile(.75):.2f}\n")

idx=list(M.index)
DECv=M['Dec'].values; SZv=M['lsize'].values
CTRY=np.array([u[0] for u in idx]); SECT=np.array([u[1] for u in idx])
Lpre=L[PRE].values
BANDS=[(10,1.0),(10,2.0),(20,2.0)]

for meas,lab in [('vol','VOLUME (n_policies)'),('tgt','TARGETING (share_frac_policies)')]:
    v=M[meas].values; pos=v[v>0]
    q75,q25=np.quantile(pos,.75),np.quantile(pos,.25)
    hi=np.where(v>=q75)[0]; lo=np.where(v<=q25)[0]
    print(f"{'='*100}\n{lab}   positives {len(pos):,}/{len(M):,} ({100*len(pos)/len(M):.0f}%)"
          f"   p25 {q25:.5g}   p75 {q75:.5g}")
    print(f"  HIGH (>= p75 of positives): {len(hi):,}     LOW (<= p25, incl zeros): {len(lo):,}")
    Dlo,Slo,Clo,Llo = DECv[lo],SZv[lo],CTRY[lo],Lpre[lo]
    lomin,lomax = Llo.min(axis=0),Llo.max(axis=0)
    for DECB,SZB in BANDS:
        nd=np.zeros(len(hi),int); ins=np.zeros(len(hi),bool); own=np.full(len(hi),np.nan)
        for a,u in enumerate(hi):
            m=(np.abs(Dlo-DECv[u])<=DECB)&(np.abs(Slo-SZv[u])<=SZB)
            n=int(m.sum()); nd[a]=n
            if n<5: continue
            Dm=Llo[m]; b=Lpre[u]
            ins[a]=bool((b>=Dm.min(axis=0)).all() and (b<=Dm.max(axis=0)).all())
            own[a]=float((Clo[m]==CTRY[u]).mean())
        k=nd>=5
        print(f"    Dec +/-{DECB:>2}pp, size +/-{SZB:.1f}: usable {int(k.sum()):>5,}/{len(hi):,}"
              f" ({100*k.mean():>3.0f}%)   median donors {np.median(nd[k]):>5.0f}"
              f"   inside hull {100*ins[k].mean():>4.0f}%"
              f"   own-country donors {100*np.nanmean(own[k]):>4.1f}%")
    print()

print(f"{'='*100}\nVIETNAM's HIGH-TARGETING SECTORS and their donor pools "
      f"(Dec +/-10pp, size +/-2.0)\n{'='*100}")
v=M['tgt'].values; pos=v[v>0]
q75,q25=np.quantile(pos,.75),np.quantile(pos,.25)
hiT=np.where(v>=q75)[0]; loT=np.where(v<=q25)[0]
Dlo,Slo,Clo,Llo=DECv[loT],SZv[loT],CTRY[loT],Lpre[loT]
vn=[a for a in hiT if CTRY[a]==704]
print(f"Vietnam: {len(vn)} high-targeting sectors of {int((CTRY==704).sum())} with a complete series\n")
for a in sorted(vn,key=lambda z:-SZv[z])[:8]:
    m=(np.abs(Dlo-DECv[a])<=10)&(np.abs(Slo-SZv[a])<=2.0)
    n=int(m.sum()); nown=int((Clo[m]==704).sum())
    ownsec=[str(SECT[loT[j]]) for j in np.where(m)[0] if Clo[j]==704]
    others=sorted({nm(c) for c in Clo[m] if c!=704})
    b=Lpre[a]; inside = n>=5 and bool((b>=Llo[m].min(axis=0)).all() and (b<=Llo[m].max(axis=0)).all())
    print(f"  sector {SECT[a]}  Dec {DECv[a]:>4.0f}%  logsize {SZv[a]:>5.1f}  "
          f"donors {n:>4d} ({nown} own)  in hull: {'YES' if inside else 'no'}")
    if ownsec: print(f"      own-country donor sectors: {', '.join(ownsec[:10])}")
    if others: print(f"      other countries: {', '.join(others[:10])}"
                     f"{f' (+{len(others)-10} more)' if len(others)>10 else ''}")
log("DONE")
