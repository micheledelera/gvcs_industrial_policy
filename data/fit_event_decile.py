"""Staggered event study with comparisons confined to the same MVA bracket.

Replaces alpha_kt with alpha_{k,bracket,t}. Every comparison is then among countries in
the SAME sector, SAME MVA bracket and SAME year, as that sector passes its own
decoupling date. This is the non-parametric version of "compare countries at similar
levels of industrial capability".

  s_ikt = sum_tau beta_tau (IP_ik x 1[t-E_k=tau]) + a_ik + a_{k,br,t} + e_ikt

The MVA x event-time terms of SS3t are dropped: MVA is near-constant inside a bracket,
so alpha_{k,br,t} does that job non-parametrically and the interactions are collinear.

Cells containing only treated or only untreated countries contribute nothing and are
absorbed. SS3u support check: 61% of decile cells have both, holding 99.7% of weight.

Run at deciles, quintiles and terciles, weighted and unweighted. The unweighted version
matters because Mexico carries 37.6% of the weighted estimate.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, sys, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
USA, CHINA = 842, 156
BASE=[2015,2016,2017]; THR=0.25; LO,HI=-6,5; REF=-1
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
chn=s.xs(CHINA,level='i').unstack('t'); chn.columns=[int(c) for c in chn.columns]
chn=chn.reindex(columns=sorted(chn.columns))
b=chn[BASE].mean(axis=1); chn=chn.loc[b>=5.0]; b=b.loc[chn.index]
below=chn.lt(b*(1-THR),axis=0)
stay=below.iloc[:,::-1].cummin(axis=1).iloc[:,::-1].astype(bool)
E=stay.apply(lambda r: r.index[r.values.argmax()] if r.any() else np.nan,axis=1).dropna().astype(int).rename('E')
ip=(raw[raw['t'].isin(BASE)][['i','ISIC4c','t',MEASURE]].drop_duplicates(subset=['i','ISIC4c','t'])
    .groupby(['i','ISIC4c'],observed=True)[MEASURE].mean().rename('IP'))
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
act=raw.loc[raw['imports']>0,['i','ISIC4c']].drop_duplicates()
years=sorted(raw['t'].unique())
del raw,us; gc.collect()
u=pd.read_excel(XLSX,sheet_name='Data'); u=u[u['Year'].isin(BASE)]
U=u.pivot_table(index='CountryCode',columns='VariableCode',values='Value',aggfunc='mean')

p=act.loc[act.index.repeat(len(years))].copy(); p['t']=np.tile(years,len(act))
p=p.set_index(['i','ISIC4c','t']).join(s).reset_index()
p=p.merge(wgt,on=['i','ISIC4c'],how='left').merge(ip,on=['i','ISIC4c'],how='left')
p['s']=p['s'].fillna(0.0); p['w']=p['w'].fillna(0.0); p['IP']=p['IP'].fillna(0.0)
p['Advanced_i']=p['i'].map(adv); p=p[(p['Advanced_i']==0)&(p['i']!=CHINA)&(p['w']>0)].copy()
p=p.merge(E,left_on='ISIC4c',right_index=True,how='inner')
ctry=sorted(p['i'].unique())
mva=pd.Series(U['NV_IND_MANF'].reindex([C2ISO.get(c,c) for c in ctry]).values,index=ctry)
p['mva']=p['i'].map(mva); p=p.dropna(subset=['mva']).copy()
p['IPz']=p['IP']/p['IP'].std()
p['tau']=(p['t']-p['E']).clip(LO,HI).astype(int)
p['fe_ik']=p.groupby(['i','ISIC4c'],observed=True).ngroup().astype('int32')
p['cl']=p['i'].astype('int32')
log(f"panel {p.shape} | {p['i'].nunique()} countries | {p['ISIC4c'].nunique()} sectors "
    f"| {p['fe_ik'].nunique():,} pairs")

TAUS=[t for t in range(LO,HI+1) if t!=REF]
tg=lambda k: f"m{abs(k)}" if k<0 else f"p{k}"
cm=p.groupby('i')['mva'].first()
from scipy import stats
rows=[]
for nb,blab in [(10,'DECILE'),(5,'QUINTILE'),(3,'TERCILE')]:
    br=pd.qcut(cm,nb,labels=False,duplicates='drop')
    p['br']=p['i'].map(br)
    p['fe_kbt']=p.groupby(['ISIC4c','br','t'],observed=True).ngroup().astype('int32')
    for wlab,wt in [('weighted','w'),('unweighted',None)]:
        d=p.copy(); tv=d['tau'].values; ni=[]
        for k in TAUS:
            d[f"B{tg(k)}"]=(d['IPz'].values*(tv==k)).astype('float32'); ni.append(f"B{tg(k)}")
        try:
            m=pf.feols(f"s ~ {' + '.join(ni)} | fe_ik + fe_kbt", data=d,
                       weights=wt, vcov={'CRV1':'cl'}, lean=True,
                       store_data=False, copy_data=False)
        except Exception as e:
            print(f"\n{blab} {wlab} FAILED: {type(e).__name__}: {e}"); continue
        t=m.tidy()
        print(f"\n{'='*62}\n{blab} brackets, {wlab}   N={m._N:,}   "
              f"{d['fe_kbt'].nunique():,} sector-bracket-year cells\n{'='*62}")
        print(f"{'tau':>5s}{'IP x 1[tau]':>26s}")
        for k in range(LO,HI+1):
            if k==REF: print(f"{k:>5d}{'- reference -':>26s}"); continue
            nm=f"B{tg(k)}"
            if nm in t.index:
                r=t.loc[nm]
                st='***' if r['Pr(>|t|)']<.01 else '**' if r['Pr(>|t|)']<.05 else '*' if r['Pr(>|t|)']<.10 else ''
                print(f"{k:>5d}{r['Estimate']:+10.4f} ({r['Std. Error']:.4f}){st:3s}".rjust(31))
                rows.append({'brackets':blab,'weighting':wlab,'tau':k,'coef':r['Estimate'],
                             'se':r['Std. Error'],'p':r['Pr(>|t|)']})
            else: print(f"{k:>5d}{'dropped':>26s}")
        pre=[f"B{tg(k)}" for k in TAUS if k<REF and f"B{tg(k)}" in t.index]
        po =[f"B{tg(k)}" for k in TAUS if k>REF and f"B{tg(k)}" in t.index]
        bb=t.loc[pre,'Estimate'].values
        idx=[list(t.index).index(n) for n in pre]; V=m._vcov[np.ix_(idx,idx)]
        W=float(bb@np.linalg.pinv(V)@bb); F=W/len(pre); G=d['cl'].nunique()
        print(f"  PRE-TREND joint F({len(pre)},{G-1}) = {F:.2f}   p = {1-stats.f.cdf(F,len(pre),G-1):.4f}")
        print(f"  post mean {t.loc[po,'Estimate'].mean():+.4f}   "
              f"n(p<.05) {int((t.loc[po,'Pr(>|t|)']<.05).sum())}/{len(po)}")
        del m,d; gc.collect()
pd.DataFrame(rows).to_csv(f"event_decile_{MEASURE}.csv",index=False)
log("DONE")
