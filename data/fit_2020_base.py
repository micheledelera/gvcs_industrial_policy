"""Both specifications on the 2020 base, all four measures x three timings.

DIFFERENCES   d_s_ik = b1(IP x Dec) + b2 IP + g d_s_pre + dl s_base + a_k + e
LEVELS        s_ikt  = sum_tau b1_tau(IP x Dec x 1[t=tau]) + sum_tau b2_tau(IP x 1[t=tau])
                       + a_ik + a_kt + e            ref tau = 2020

Both weighted by 2018-20 US imports and clustered by country. Developing ex-China.
"""
import pandas as pd, numpy as np, statsmodels.api as sm, pyfixest as pf, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
REF=2020
MEAS=['n_policies','frac_policies','share_n_policies','share_frac_policies']
TIM=['flow','stock','lag']

d=pd.read_pickle("base2020_cross.pkl")
p=pd.read_pickle("base2020_panel.pkl")
for f in (d,p): f['Dz']=(f['Dec']-f['Dec'].mean())/f['Dec'].std()
sec=pd.get_dummies(d['ISIC4c'].astype(str),prefix='k',drop_first=True).astype(float)
w=d['w'].values
p['fe_ik']=p.groupby(['i','ISIC4c'],observed=True).ngroup().astype('int32')
p['fe_kt']=p.groupby(['ISIC4c','t'],observed=True).ngroup().astype('int32')
p['cl_i']=p['i'].astype('int32')
years=sorted(p['t'].unique()); tv=p['t'].values

def st(pv): return '***' if pv<.01 else '**' if pv<.05 else '*' if pv<.10 else ''

print("="*96); print("TABLE 1 — DIFFERENCES.  DV: s(2022-24) - s(2018-20), pp"); print("="*96)
print(f"{'measure':24s}{'timing':8s}{'IP x Dec':>22s}{'IP':>22s}{'d_pre':>16s}")
rows1=[]
for M in MEAS:
    for T in TIM:
        c=f"{M}__{T}"
        d['IPz']=d[c]/d[c].std(); d['IPxDec']=d['IPz']*d['Dz']
        X=['IPxDec','IPz','d_pre','s_base']
        m=sm.WLS(d['d_post'],sm.add_constant(pd.concat([d[X],sec],axis=1)),weights=w)\
            .fit(cov_type='cluster',cov_kwds={'groups':d['i']})
        f1=f"{m.params['IPxDec']:+.4f}{st(m.pvalues['IPxDec'])} ({m.bse['IPxDec']:.4f})"
        f2=f"{m.params['IPz']:+.4f}{st(m.pvalues['IPz'])} ({m.bse['IPz']:.4f})"
        f3=f"{m.params['d_pre']:+.3f}{st(m.pvalues['d_pre'])}"
        print(f"{M:24s}{T:8s}{f1:>22s}{f2:>22s}{f3:>16s}")
        rows1.append({'measure':M,'timing':T,'b_int':m.params['IPxDec'],'p_int':m.pvalues['IPxDec'],
                      'b_ip':m.params['IPz'],'p_ip':m.pvalues['IPz'],'N':int(m.nobs)})
print(f"\nN={len(d):,}   sector FE=124   clusters={d['i'].nunique()}   weighted by 2018-20 US imports")
pd.DataFrame(rows1).to_csv("base2020_diff_results.csv",index=False)

print("\n"+"="*96); print(f"TABLE 2 — LEVELS, event study, ref {REF}"); print("="*96)
rows2=[]
for M in MEAS:
    for T in TIM:
        c=f"{M}__{T}"; ipz=(p[c]/p[c].std()).values
        ni,nd=[],[]
        for y in years:
            if y==REF: continue
            p[f"I_{y}"]=ipz*(tv==y); ni.append(f"I_{y}")
            p[f"D_{y}"]=ipz*p['Dz'].values*(tv==y); nd.append(f"D_{y}")
        fit=pf.feols(f"s ~ {' + '.join(nd+ni)} | fe_ik + fe_kt", data=p, weights="w",
                     vcov={"CRV1":"cl_i"}, lean=True, store_data=False, copy_data=False)
        t=fit.tidy()
        print(f"\n--- {M} / {T} ---   N={fit._N:,}")
        print(f"{'year':>6s}{'IP x Dec x year':>28s}{'IP x year':>28s}")
        for y in years:
            if y==REF: print(f"{y:>6d}{'— ref —':>28s}{'— ref —':>28s}"); continue
            o=f"{y:>6d}"
            for nm in [f"D_{y}",f"I_{y}"]:
                r=t.loc[nm]
                o+=f"{r['Estimate']:+.4f}{st(r['Pr(>|t|)']):<3s}({r['Std. Error']:.4f})".rjust(28)
            print(o)
        pre=[f"D_{y}" for y in years if y<REF]; prei=[f"I_{y}" for y in years if y<REF]
        post=[y for y in years if y>REF]
        s1=int((t.loc[pre,'Pr(>|t|)']<.05).sum()); s2=int((t.loc[prei,'Pr(>|t|)']<.05).sum())
        pm=np.mean([t.loc[f'D_{y}','Estimate'] for y in post])
        im=np.mean([t.loc[f'I_{y}','Estimate'] for y in post])
        print(f"   pre-period ({len(pre)} yrs) sig@5%:  IPxDec {s1}   IP {s2}"
              f"   |  post mean: IPxDec {pm:+.4f}  IP {im:+.4f}")
        rows2.append({'measure':M,'timing':T,'pre_sig_int':s1,'pre_sig_ip':s2,
                      'post_mean_int':pm,'post_mean_ip':im,'N':fit._N})
        for cc in ni+nd: del p[cc]
        del fit; gc.collect()
pd.DataFrame(rows2).to_csv("base2020_levels_results.csv",index=False)
log("ALL DONE")
