"""Capability controls applied to the two between-country results that actually carry
signal, rather than to every cell of the grid.

  PANEL A  2017-base long difference (fit_longdiff.py col 4): share_frac_policies +0.172***
  PANEL B  2020-base difference with the Dec interaction, stock timing:
           IPxDec +0.160**, IP +0.234***  for share_frac_policies

Same four rungs as fit_longdiff_ctrl.py. Rung 4 (country FE) is the test that answers
"Mexico and Vietnam are not comparable": under alpha_i they are never compared, and the
coefficient is identified only from a country's own targeted vs untargeted sectors.
"""
import pandas as pd, numpy as np, gc
import pyfixest as pf

CHINA, USA = 156, 842
PRE0, PRE, POST = [2010,2011,2012], [2015,2016,2017], [2022,2023,2024]
C2ISO = {699:356, 757:756, 579:578, 251:250, 842:840, 381:380, 490:158, 729:729}
XLSX = ("/root/.claude/uploads/618110ef-d894-58be-8e00-e7f2bf548b03/"
        "4ad6510b-0154963EF4289D02DEA410D8D38E4E6A8C6D.xlsx")
MEAS = ['n_policies','frac_policies','share_n_policies','share_frac_policies']

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
for c in MEAS: raw[c]=pd.to_numeric(raw[c],errors='coerce').fillna(0).astype('float32')

# ---------- country controls, pre-determined 2015-17 ----------
pre = raw[raw['t'].isin(PRE)]
X = (pre.groupby(['i','ISIC4c'], observed=True)['imports'].sum()/len(PRE)).unstack(fill_value=0.0)
sh = X.div(X.sum(axis=1).replace(0,np.nan), axis=0)
world = X.sum(axis=0)/X.sum().sum()
M = ((sh.div(world, axis=1)) >= 1).astype(float)
M = M.loc[M.sum(axis=1)>0, M.sum(axis=0)>0]
kc, kp = M.sum(axis=1), M.sum(axis=0)
vals, vecs = np.linalg.eig(((M.div(kc,axis=0)) @ (M.div(kp,axis=1).T)).values)
K = np.real(vecs[:, np.argsort(-np.real(vals))[1]])
ECI = pd.Series((K-K.mean())/K.std(), index=M.index)
if ECI.corr(pd.Series(kc, index=M.index)) < 0: ECI = -ECI
u = pd.read_excel(XLSX, sheet_name='Data'); u = u[u['Year'].isin(PRE)]
U = u.pivot_table(index='CountryCode', columns='VariableCode', values='Value', aggfunc='mean')
def controls(ctry):
    C = pd.DataFrame(index=pd.Index(ctry, name='i'))
    iso = [C2ISO.get(c,c) for c in ctry]
    C['log_mva_pc'] = np.log(U['NV_IND_MANFPC'].reindex(iso).values)
    C['mva_gdp']    = U['NV_IND_MANF'].reindex(iso).values
    C['mfg_emp']    = U['SL_TLF_MANF'].reindex(iso).values
    C['log_exp']    = np.log(X.sum(axis=1).reindex(ctry).replace(0,np.nan).values)
    C['ECI']        = ECI.reindex(ctry).values
    return C
CTRL = ['log_mva_pc','mva_gdp','mfg_emp','log_exp','ECI']

# ---------- PANEL A: 2017-base long difference ----------
us = raw[raw['j']==USA]
def share(years):
    x = us[us['t'].isin(years)].groupby(['i','ISIC4c'], observed=True)['imports'].sum()/len(years)
    tot = x.groupby(level='ISIC4c', observed=True).transform('sum')
    return (x/tot.replace(0,np.nan)).rename('s'), x.rename('x')
s0,_  = share(PRE0); s1,x1 = share(PRE); s2,_ = share(POST)
ip = (raw[raw['t'].isin(PRE)][['i','ISIC4c','t']+MEAS]
      .drop_duplicates(subset=['i','ISIC4c','t']).groupby(['i','ISIC4c'],observed=True)[MEAS].mean())
adv = raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
act = raw.loc[raw['imports']>0, ['i','ISIC4c']].drop_duplicates()
chn_s = s1.xs(CHINA, level='i').rename('Dec')          # China's 2015-17 share of US imports
del raw, us, pre; gc.collect()

A = act.set_index(['i','ISIC4c']).join([s0.rename('s0'),s1.rename('s1'),s2.rename('s2'),x1,ip]).reset_index()
for c in ['s0','s1','s2','x']+MEAS: A[c]=A[c].fillna(0.0)
A['Advanced_i']=A['i'].map(adv); A=A[(A['Advanced_i']==0)&(A['i']!=CHINA)].copy()
A=A.merge(chn_s, on='ISIC4c', how='left'); A['Dec']=A['Dec'].fillna(0.0)
A['d_post']=(A['s2']-A['s1'])*100; A['d_pre']=(A['s1']-A['s0'])*100; A['s_base']=A['s1']*100
A=A[A['x']>0].copy(); A=A.rename(columns={'x':'w'})

# ---------- PANEL B: 2020-base cross-section, stock timing ----------
B = pd.read_pickle("base2020_cross.pkl")
for c in MEAS: B[c]=B[f"{c}__stock"]

def ladder(d, tag, interact):
    C = controls(sorted(d['i'].unique()))
    print(f"\n{'#'*80}\n# {tag}\n{'#'*80}")
    print(f"raw sample {len(d):,} cells, {d['i'].nunique()} countries")
    d = d.merge(C, left_on='i', right_index=True, how='left')
    wtot = d['w'].sum()
    d = d.dropna(subset=CTRL).copy()
    print(f"complete-controls {len(d):,} cells, {d['i'].nunique()} countries, "
          f"{d['w'].sum()/wtot:.3%} of pre-period US import value")
    for c in CTRL:
        d[c]=(d[c]-d[c].mean())/d[c].std(); d[c+'_xD']=d[c]*d['Dec']
    lev=" + "+" + ".join(CTRL); itx=" + "+" + ".join(c+'_xD' for c in CTRL)
    rows=[]
    for MEASURE in MEAS:
        d['IPz']=d[MEASURE]/d[MEASURE].std()
        d['IPxDec']=d['IPz']*d['Dec']
        core = "IPz + IPxDec + Dec" if interact else "IPz"
        base = f"{core} + d_pre + s_base"
        SPECS=[("1. alpha_k baseline",    f"d_post ~ {base} | ISIC4c"),
               ("2. + capability levels", f"d_post ~ {base}{lev} | ISIC4c"),
               ("3. + capability x Dec",  f"d_post ~ {base}{lev}{itx} | ISIC4c"),
               ("4. + country FE",        f"d_post ~ {base}{itx} | ISIC4c + i")]
        print(f"\n--- {MEASURE} " + "-"*(60-len(MEASURE)))
        for name,fml in SPECS:
            m = pf.feols(fml, data=d, weights='w', vcov={'CRV1':'i'})
            out={'panel':tag,'measure':MEASURE,'spec':name,'N':int(m._N)}
            txt=f"  {name:24s} N={int(m._N):,} "
            for k in (['IPz','IPxDec'] if interact else ['IPz']):
                b,se,p=m.coef()[k],m.se()[k],m.pvalue()[k]
                st='***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''
                txt+=f"  {k} {b:+.4f} ({se:.4f}){st:3s}"
                out[k]=b; out[k+'_se']=se; out[k+'_p']=p
            print(txt); rows.append(out)
    return pd.DataFrame(rows)

rA = ladder(A, "PANEL A: 2017-base long difference (no Dec interaction)", False)
rB = ladder(B, "PANEL B: 2020-base difference, stock timing, with IPxDec", True)
pd.concat([rA,rB]).to_csv("ctrl_headline.csv", index=False)
print("\nsaved ctrl_headline.csv")
