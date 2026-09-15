"""Pure DiD versus lagged-dependent-variable, and the ladder between them.

A two-period DiD on the change assumes parallel trends and does NOT control for the
lagged outcome. An LDV model controls for the lagged outcome and does NOT assume
parallel trends. They rest on different assumptions and tend to BRACKET the truth
(Angrist & Pischke, MHE ch. 5). The recommended spec of SS3p does both at once, so it
is neither cleanly and the bracketing logic does not straightforwardly apply.

  (a) pure DiD          d_s ~ IPxDec + IP                        + a_k
  (b) + pre-trend       d_s ~ IPxDec + IP + d_s^pre              + a_k
  (c) LDV               d_s ~ IPxDec + IP           + s^pre      + a_k
  (d) both  [SS3p]      d_s ~ IPxDec + IP + d_s^pre + s^pre      + a_k

Each with mva_gdp + mva_gdp x Dec throughout, and repeated with a country FE.
(a) and (c) are the bracket.
"""
import pandas as pd, numpy as np, pyfixest as pf, gc, sys
src=open("fit_longdiff_iso.py").read().split('lev=" + ".join(CTRL)')[0]
src=src.replace("CTRL=sys.argv[1].split(',') if len(sys.argv)>1 else ['mva_gdp']","CTRL=['mva_gdp']")
src=src.replace("SAMP=sys.argv[2].split(',') if len(sys.argv)>2 else CTRL","SAMP=CTRL")
exec(src)

MEAS=['n_policies','frac_policies','share_n_policies','share_frac_policies']
CAP="mva_gdp + mva_gdp_xD"
LAD=[("a. pure DiD          ", ""),
     ("b. + pre-trend       ", " + d_pre"),
     ("c. LDV (initial only)", " + s_pre"),
     ("d. both      [SS3p]  ", " + d_pre + s_pre")]
rows=[]
for MEASURE in MEAS:
    d['IPz']=d[MEASURE]/d[MEASURE].std(); d['IPxDec']=d['IPz']*d['Dz']
    print(f"\n{'='*100}\nMEASURE {MEASURE}    N={len(d):,}  |  {d['i'].nunique()} countries  "
          f"|  {d['ISIC4c'].nunique()} sectors\n{'='*100}")
    print(f"{'spec':24s}{'--- alpha_k only ---':>46s}{'--- + country FE ---':>46s}")
    print(f"{'':24s}{'IP x Dec':>23s}{'IP':>23s}{'IP x Dec':>23s}{'IP':>23s}")
    for name,lag in LAD:
        o=f"{name:24s}"; rec={'measure':MEASURE,'spec':name.strip()}
        for tag,fe in [('k','ISIC4c'),('ki','ISIC4c + i')]:
            fml=f"d_post ~ IPxDec + IPz{lag} + {CAP} | {fe}"
            m=pf.feols(fml,data=d,weights='w',vcov={'CRV1':'i'})
            for k in ['IPxDec','IPz']:
                b,se,p=m.coef()[k],m.se()[k],m.pvalue()[k]
                st='***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''
                o+=f"{b:+.4f} ({se:.4f}){st:3s}".rjust(23)
                rec[f'{tag}_{k}']=b; rec[f'{tag}_{k}_se']=se; rec[f'{tag}_{k}_p']=p
            rec[f'{tag}_N']=int(m._N)
        print(o); rows.append(rec)
pd.DataFrame(rows).to_csv("did_vs_ldv.csv",index=False)
print("\nsaved did_vs_ldv.csv")
