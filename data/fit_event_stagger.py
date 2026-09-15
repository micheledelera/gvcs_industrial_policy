"""Staggered event study on sector-specific decoupling dates.

Event E_k = first year China's share of US imports in sector k is >=25% below its own
2015-17 mean AND stays below thereafter (definition D of diag_event_timing.py). 62
sectors qualify; events run 2017-2024 with no year holding more than 27%.

  s_ikt = sum_tau beta_tau (IP_ik x 1[t-E_k=tau])
        + sum_tau psi_tau  (mva_gdp_i x 1[t-E_k=tau])
        + a_ik + a_kt + e_ikt                                 reference tau = -1

a_kt absorbs the event itself and each sector's own timing, so beta_tau is the IP
gradient in event time and every comparison is within a sector-year -- no forbidden
comparisons of the Goodman-Bacon kind, because all units in a sector share E_k and the
contrast is high-IP vs low-IP inside it.

psi_tau does parametrically what "compare country-sectors at similar MVA" asks for.
Panel 2 then does it non-parametrically, re-running beta_tau within MVA terciles: the
high-MVA stratum is the direct test -- among equally industrialised countries, did the
high-IP ones gain when China left their sector?

Event time binned at [-6,+5]; endpoints absorb everything beyond.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, sys, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
USA, CHINA = 842, 156
BASE=[2015,2016,2017]; THR=0.25
LO,HI=-6,5; REF=-1
C2ISO={699:356,757:756,579:578,251:250,842:840,381:380,490:158,729:729}
XLSX=("/root/.claude/uploads/618110ef-d894-58be-8e00-e7f2bf548b03/"
      "4ad6510b-0154963EF4289D02DEA410D8D38E4E6A8C6D.xlsx")
MEASURE = sys.argv[1] if len(sys.argv)>1 else 'share_frac_policies'

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
raw[MEASURE]=pd.to_numeric(raw[MEASURE],errors='coerce').fillna(0).astype('float32')
us=raw[raw['j']==USA]
x=us.groupby(['i','ISIC4c','t'],observed=True)['imports'].sum()
tot=x.groupby(level=['ISIC4c','t'],observed=True).transform('sum')
s=(x/tot.replace(0,np.nan)*100).rename('s')
wgt=(us[us['t'].isin(BASE)].groupby(['i','ISIC4c'],observed=True)['imports'].sum()/3).rename('w')

# --- event dates ---
chn=s.xs(CHINA,level='i').unstack('t'); chn.columns=[int(c) for c in chn.columns]
chn=chn.reindex(columns=sorted(chn.columns))
b=chn[BASE].mean(axis=1); chn=chn.loc[b>=5.0]; b=b.loc[chn.index]
below=chn.lt(b*(1-THR),axis=0)
stay=below.iloc[:,::-1].cummin(axis=1).iloc[:,::-1].astype(bool)
E=stay.apply(lambda r: r.index[r.values.argmax()] if r.any() else np.nan, axis=1).dropna()
E=E.astype(int).rename('E')
log(f"{len(E)} sectors with an event; years {E.min()}-{E.max()}, "
    f"modal {E.mode().iloc[0]} ({100*(E==E.mode().iloc[0]).mean():.0f}%)")

ip=(raw[raw['t'].isin(BASE)][['i','ISIC4c','t',MEASURE]].drop_duplicates(subset=['i','ISIC4c','t'])
    .groupby(['i','ISIC4c'],observed=True)[MEASURE].mean().rename('IP'))
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
act=raw.loc[raw['imports']>0,['i','ISIC4c']].drop_duplicates()
years=sorted(raw['t'].unique())
prew=raw[raw['t'].isin(BASE)]
X=(prew.groupby(['i','ISIC4c'],observed=True)['imports'].sum()/3).unstack(fill_value=0.0)
del raw,us,prew; gc.collect()

u=pd.read_excel(XLSX,sheet_name='Data'); u=u[u['Year'].isin(BASE)]
U=u.pivot_table(index='CountryCode',columns='VariableCode',values='Value',aggfunc='mean')

p=act.loc[act.index.repeat(len(years))].copy(); p['t']=np.tile(years,len(act))
p=p.set_index(['i','ISIC4c','t']).join(s).reset_index()
p=p.merge(wgt,on=['i','ISIC4c'],how='left').merge(ip,on=['i','ISIC4c'],how='left')
p['s']=p['s'].fillna(0.0); p['w']=p['w'].fillna(0.0); p['IP']=p['IP'].fillna(0.0)
p['Advanced_i']=p['i'].map(adv); p=p[(p['Advanced_i']==0)&(p['i']!=CHINA)&(p['w']>0)].copy()
p=p.merge(E,left_on='ISIC4c',right_index=True,how='inner')      # event sectors only
ctry=sorted(p['i'].unique()); iso=[C2ISO.get(c,c) for c in ctry]
mva=pd.Series(U['NV_IND_MANF'].reindex(iso).values,index=ctry)
p['mva']=p['i'].map(mva); p=p.dropna(subset=['mva']).copy()
p['mva']=(p['mva']-p['mva'].mean())/p['mva'].std()
p['IPz']=p['IP']/p['IP'].std()
p['tau']=(p['t']-p['E']).clip(LO,HI).astype(int)
p['fe_ik']=p.groupby(['i','ISIC4c'],observed=True).ngroup().astype('int32')
p['fe_kt']=p.groupby(['ISIC4c','t'],observed=True).ngroup().astype('int32')
p['cl']=p['i'].astype('int32')
log(f"panel {p.shape} | {p['i'].nunique()} countries | {p['ISIC4c'].nunique()} sectors "
    f"| {p['fe_ik'].nunique():,} pairs")
print("\nsectors contributing at each event time:")
print(p.groupby('tau')['ISIC4c'].nunique().to_string())

TAUS=[t for t in range(LO,HI+1) if t!=REF]
def run(df,label):
    d=df.copy()
    ni,nm=[],[]
    tv=d['tau'].values
    for k in TAUS:
        tag = f"m{abs(k)}" if k<0 else f"p{k}"
        d[f"B{tag}"]=(d['IPz'].values*(tv==k)).astype('float32'); ni.append(f"B{tag}")
        d[f"M{tag}"]=(d['mva'].values*(tv==k)).astype('float32'); nm.append(f"M{tag}")
    fml=f"s ~ {' + '.join(ni+nm)} | fe_ik + fe_kt"
    m=pf.feols(fml,data=d,weights='w',vcov={'CRV1':'cl'},lean=True,
               store_data=False,copy_data=False)
    t=m.tidy()
    print(f"\n{'='*66}\n{label}   N={m._N:,}  countries {d['cl'].nunique()}  "
          f"pairs {d['fe_ik'].nunique():,}\n{'='*66}")
    print(f"{'tau':>5s}{'IP x 1[tau]':>24s}{'MVA x 1[tau]':>24s}")
    rows=[]
    for k in range(LO,HI+1):
        if k==REF:
            print(f"{k:>5d}{'- reference -':>24s}{'- reference -':>24s}"); continue
        o=f"{k:>5d}"
        for pre in ['B','M']:
            nm_=f"{pre}"+(f"m{abs(k)}" if k<0 else f"p{k}")
            if nm_ in t.index:
                r=t.loc[nm_]
                st='***' if r['Pr(>|t|)']<.01 else '**' if r['Pr(>|t|)']<.05 else '*' if r['Pr(>|t|)']<.10 else ''
                o+=f"{r['Estimate']:+8.4f} ({r['Std. Error']:.4f}){st:3s}".rjust(24)
                if pre=='B': rows.append({'label':label,'tau':k,'coef':r['Estimate'],
                                          'se':r['Std. Error'],'p':r['Pr(>|t|)']})
            else: o+=f"{'dropped':>24s}"
        print(o)
    tg=lambda k: f"m{abs(k)}" if k<0 else f"p{k}"
    pre=[f"B{tg(k)}" for k in TAUS if k<REF and f"B{tg(k)}" in t.index]
    po=[f"B{tg(k)}" for k in TAUS if k>REF and f"B{tg(k)}" in t.index]
    from scipy import stats
    bb=t.loc[pre,'Estimate'].values
    idx=[list(t.index).index(n) for n in pre]; V=m._vcov[np.ix_(idx,idx)]
    W=float(bb@np.linalg.pinv(V)@bb); F=W/len(pre); G=d['cl'].nunique()
    print(f"  PRE-TREND joint F({len(pre)},{G-1}) = {F:.2f}   p = {1-stats.f.cdf(F,len(pre),G-1):.4f}")
    print(f"  post mean {t.loc[po,'Estimate'].mean():+.4f}   "
          f"n(p<.05) {int((t.loc[po,'Pr(>|t|)']<.05).sum())}/{len(po)}")
    del m; gc.collect()
    return rows

allrows=run(p,"PANEL 1 -- all countries")
q=p.groupby('i')['mva'].first().quantile([1/3,2/3])
for lab,sel in [("LOW MVA tercile", p['mva']<=q.iloc[0]),
                ("MID MVA tercile",(p['mva']>q.iloc[0])&(p['mva']<=q.iloc[1])),
                ("HIGH MVA tercile",p['mva']>q.iloc[1])]:
    try: allrows+=run(p[sel],f"PANEL 2 -- {lab}")
    except Exception as e: print(f"\n{lab} FAILED: {type(e).__name__}: {e}")
pd.DataFrame(allrows).to_csv(f"event_stagger_{MEASURE}.csv",index=False)
log("DONE")
