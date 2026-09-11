"""Does the between-country long difference survive country-level capability controls?

The sector fixed effect alpha_k holds EXPOSURE fixed -- every country in the
regression faces the same vacated Chinese share in sector k. It does nothing about
COMPARABILITY of the countries themselves. Mexico and Vietnam both sell auto parts;
that does not make them equally able to absorb a demand shift, and it does not make
their governments equally likely to write an auto-parts policy. Selection into
treatment and heterogeneous capacity to respond are separate threats from exposure.

Four rungs, each removing a different part of the country:

  (1) alpha_k + pre-trend + initial share            the current baseline
  (2) + country capability LEVELS                    higher-capability countries may
                                                     gain share everywhere
  (3) + country capability x Dec_k                   ...and may absorb a vacated
                                                     sector faster than low-capability
                                                     ones, which is exactly the
                                                     confound the story needs ruled out
  (4) + alpha_i  (country fixed effect)              absorbs EVERYTHING country-level,
                                                     observed or not. Identification is
                                                     now purely WITHIN country, across
                                                     sectors: does a country gain more
                                                     share in the sectors it targeted
                                                     than in the ones it did not?

Rung (4) is the answer to "Mexico and Vietnam are not comparable": under alpha_i they
are never compared. The price is that the VOLUME question dies with it -- a country's
total policy effort is collinear with alpha_i -- and only TARGETING survives.

Controls (all pre-determined, 2015-17 means, so none is a function of the outcome):
  log MVA per capita   UNIDO NV_IND_MANFPC   industrial capability per head
  MVA / GDP            UNIDO NV_IND_MANF     industrialisation of the economy
  mfg emp share        UNIDO SL_TLF_MANF     manufacturing labour base
  log exports to world BACI                  size
  ECI                  BACI                  complexity of the export basket
"""
import pandas as pd, numpy as np, gc, sys
import pyfixest as pf

CHINA, USA = 156, 842
PRE = [2015,2016,2017]
C2ISO = {699:356, 757:756, 579:578, 251:250, 842:840, 381:380, 490:158, 729:729}
XLSX = ("/root/.claude/uploads/618110ef-d894-58be-8e00-e7f2bf548b03/"
        "4ad6510b-0154963EF4289D02DEA410D8D38E4E6A8C6D.xlsx")
MEAS = ['n_policies','frac_policies','share_n_policies','share_frac_policies']
TIMING = sys.argv[1] if len(sys.argv)>1 else 'flow'

d = pd.read_pickle("base2020_cross.pkl")
print(f"cross-section {len(d):,} cells | {d['i'].nunique()} countries | {d['ISIC4c'].nunique()} sectors")

# ---------- country controls ----------
raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['t']=raw['t'].astype('int32')
pre = raw[raw['t'].isin(PRE)]
X = (pre.groupby(['i','ISIC4c'], observed=True)['imports'].sum()/len(PRE)).unstack(fill_value=0.0)
del raw, pre; gc.collect()

sh = X.div(X.sum(axis=1).replace(0,np.nan), axis=0)
world = X.sum(axis=0)/X.sum().sum()
M = ((sh.div(world, axis=1)) >= 1).astype(float)
M = M.loc[M.sum(axis=1)>0, M.sum(axis=0)>0]
kc, kp = M.sum(axis=1), M.sum(axis=0)
Mcc = (M.div(kc,axis=0)) @ (M.div(kp,axis=1).T)
vals, vecs = np.linalg.eig(Mcc.values)
K = np.real(vecs[:, np.argsort(-np.real(vals))[1]])
ECI = pd.Series((K-K.mean())/K.std(), index=M.index)
if ECI.corr(pd.Series(kc, index=M.index)) < 0: ECI = -ECI

u = pd.read_excel(XLSX, sheet_name='Data')
u = u[u['Year'].isin(PRE)]
U = u.pivot_table(index='CountryCode', columns='VariableCode', values='Value', aggfunc='mean')

ctry = pd.Index(sorted(d['i'].unique()), name='i')
C = pd.DataFrame(index=ctry)
C['log_mva_pc']  = np.log(U['NV_IND_MANFPC'].reindex([C2ISO.get(c,c) for c in ctry]).values)
C['mva_gdp']     = U['NV_IND_MANF'].reindex([C2ISO.get(c,c) for c in ctry]).values
C['mfg_emp']     = U['SL_TLF_MANF'].reindex([C2ISO.get(c,c) for c in ctry]).values
C['log_exp']     = np.log(X.sum(axis=1).reindex(ctry).replace(0,np.nan).values)
C['ECI']         = ECI.reindex(ctry).values
CTRL = list(C.columns)
print("\ncountry-control coverage (of %d countries):" % len(ctry))
for c in CTRL: print(f"  {c:12s} {C[c].notna().sum():>4}")

d = d.merge(C, left_on='i', right_index=True, how='left')
wtot = d['w'].sum()
full = d[C[CTRL].notna().all(axis=1).reindex(d['i']).values].copy()
print(f"\ncomplete-controls sample: {len(full):,} cells, {full['i'].nunique()} countries, "
      f"{full['w'].sum()/wtot:.3%} of pre-period US import value")
for c in CTRL:                       # standardise so coefficients are per-sd
    full[c] = (full[c]-full[c].mean())/full[c].std()
    full[c+'_xD'] = full[c]*full['Dec']

rows=[]
for MEASURE in MEAS:
    v = f"{MEASURE}__{TIMING}"
    full['IPz'] = full[v]/full[v].std()
    base = "IPz + d_pre + s_base"
    lev  = " + " + " + ".join(CTRL)
    itx  = " + " + " + ".join(c+'_xD' for c in CTRL)
    SPECS = [
      ("1. alpha_k baseline",      f"d_post ~ {base} | ISIC4c"),
      ("2. + capability levels",   f"d_post ~ {base}{lev} | ISIC4c"),
      ("3. + capability x Dec",    f"d_post ~ {base}{lev}{itx} | ISIC4c"),
      ("4. + country FE",          f"d_post ~ {base}{itx} | ISIC4c + i"),
    ]
    print(f"\n{'='*74}\nMEASURE {v}\n{'='*74}")
    for name, fml in SPECS:
        m = pf.feols(fml, data=full, weights='w', vcov={'CRV1':'i'})
        b,se,p = m.coef()['IPz'], m.se()['IPz'], m.pvalue()['IPz']
        print(f"--- {name:24s} N={int(m._N):,}  IPz {b:+.5f} (se {se:.5f})  p={p:.4f}")
        if name.startswith("2."):
            for c in CTRL:
                print(f"      {c:12s} {m.coef()[c]:+.5f} (se {m.se()[c]:.5f}) p={m.pvalue()[c]:.3f}")
        rows.append({'measure':MEASURE,'timing':TIMING,'spec':name,'coef':b,'se':se,
                     'p':p,'N':int(m._N)})

r = pd.DataFrame(rows); r.to_csv(f"longdiff_ctrl_{TIMING}.csv", index=False)
def cell(x):
    st='***' if x['p']<.01 else '**' if x['p']<.05 else '*' if x['p']<.10 else ''
    return f"{x['coef']:+.4f} ({x['se']:.4f}){st}"
r['c']=r.apply(cell,axis=1)
print("\n"+"="*96)
print(f"IP COEFFICIENT, timing = {TIMING}   (pp change in country's share of US imports in sector k)")
print("="*96)
print(r.pivot_table(index='measure',columns='spec',values='c',aggfunc='first')
       .reindex(MEAS).to_string())
