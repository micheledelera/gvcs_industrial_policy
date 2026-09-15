"""Long difference: is the effect specific to developing countries?

  d_s_ik = b1 (IP_ik x Dec_k) + b2 IP_ik + g d_s^pre_ik + dl s^pre_ik + a_k + e_ik

Without b1 the specification has no decoupling content at all: Dec is a sector-level
object and alpha_k absorbs it, so beta would be "policy users gained more share in a
window that happens to contain the trade war" -- indistinguishable from COVID
reshoring or currency moves. The interaction is what makes it a decoupling result,
and it survives alpha_k because it varies across countries within a sector.

Two definitions of sector decoupling, run separately:

  Dec_pre       China's share of US imports in sector k, 2015-17. Pre-determined:
                how much was at stake before anything happened. No circularity.
  Dec_realised  China's share loss in sector k, 2015-17 -> 2022-24. Closer to
                "decoupling actually happened here", but realised. Note shares sum
                to one within a sector, so the AVERAGE gain across suppliers equals
                China's loss mechanically -- that part is absorbed by alpha_k. The
                interaction asks whether policy users captured a larger slice of
                that redistribution, which is not mechanical.

Both standardised. Weighted by pre-period US imports, clustered by country.
"""
import pandas as pd, numpy as np, statsmodels.api as sm, gc
USA, CHINA = 842, 156
PRE0,PRE,POST=[2010,2011,2012],[2015,2016,2017],[2022,2023,2024]
MEAS=['n_policies','frac_policies','share_n_policies','share_frac_policies']
raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
for c in MEAS: raw[c]=pd.to_numeric(raw[c],errors='coerce').fillna(0).astype('float32')
us=raw[raw['j']==USA]
def share(years):
    x=us[us['t'].isin(years)].groupby(['i','ISIC4c'],observed=True)['imports'].sum()/len(years)
    tot=x.groupby(level='ISIC4c',observed=True).transform('sum')
    return (x/tot.replace(0,np.nan)).rename('s'), x.rename('x')
s0,_=share(PRE0); s1,x1=share(PRE); s2,_=share(POST)
chn_pre  = s1.xs(CHINA, level='i') if CHINA in s1.index.get_level_values('i') else None
chn_post = s2.xs(CHINA, level='i')
dec = pd.DataFrame({'Dec_pre': chn_pre.fillna(0),
                    'Dec_realised': (chn_pre - chn_post.reindex(chn_pre.index)).fillna(0)})
ip=(raw[raw['t'].isin(PRE)][['i','ISIC4c','t']+MEAS].drop_duplicates(subset=['i','ISIC4c','t'])
    .groupby(['i','ISIC4c'],observed=True)[MEAS].mean())
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
act=raw.loc[raw['imports']>0,['i','ISIC4c']].drop_duplicates()
del raw, us; gc.collect()
d=act.set_index(['i','ISIC4c']).join([s0.rename('s0'),s1.rename('s1'),s2.rename('s2'),x1,ip]).reset_index()
d=d.merge(dec, on='ISIC4c', how='left')
for c in ['s0','s1','s2','x','Dec_pre','Dec_realised']+MEAS: d[c]=d[c].fillna(0.0)
d['Advanced_i']=d['i'].map(adv)
d=d[d['i']!=CHINA].copy()                      # keep advanced AND developing, drop China
d['dev']=(d['Advanced_i']==0).astype(float); d['adv']=(d['Advanced_i']==1).astype(float)
print(f"  rows: developing {int(d['dev'].sum()):,}  advanced {int(d['adv'].sum()):,}")
d['d_post']=(d['s2']-d['s1'])*100; d['d_pre']=(d['s1']-d['s0'])*100; d['s1_pp']=d['s1']*100
print(f"N={len(d):,}   sectors={d['ISIC4c'].nunique()}   countries={d['i'].nunique()}")
print("\nsector decoupling measures (cell level):")
print(d[['Dec_pre','Dec_realised']].describe().round(4).to_string())
print(f"corr(Dec_pre, Dec_realised) = {d['Dec_pre'].corr(d['Dec_realised']):+.3f}")

sec=pd.get_dummies(d['ISIC4c'].astype(str),prefix='k',drop_first=True).astype(float)
w=d['x'].clip(lower=0).values
rows=[]
for DEC in ['Dec_pre','Dec_realised']:
    d['Dz']=(d[DEC]-d[DEC].mean())/d[DEC].std()
    print(f"\n{'='*74}\nDECOUPLING MEASURE: {DEC}\n{'='*74}")
    for M in MEAS:
        d['IPz']=d[M]/d[M].std()
        d['IPxDec_dev']=d['IPz']*d['Dz']*d['dev']; d['IPxDec_adv']=d['IPz']*d['Dz']*d['adv']
        d['IP_dev']=d['IPz']*d['dev'];             d['IP_adv']=d['IPz']*d['adv']
        d['Dec_adv']=d['Dz']*d['adv']
        X=['IPxDec_dev','IPxDec_adv','IP_dev','IP_adv','Dec_adv','d_pre','s1_pp']
        Mx=pd.concat([d[X],sec],axis=1)
        m=sm.WLS(d['d_post'],sm.add_constant(Mx),weights=w).fit(
            cov_type='cluster',cov_kwds={'groups':d['i']})
        def f(k):
            st='***' if m.pvalues[k]<.01 else '**' if m.pvalues[k]<.05 else '*' if m.pvalues[k]<.10 else ''
            return f"{m.params[k]:+.4f}{st}".rjust(12)+f"({m.bse[k]:.4f})".rjust(11)
        print(f"{M:22s} IPxDec dev {f('IPxDec_dev')}   adv {f('IPxDec_adv')}")
        print(f"{'':22s} IP     dev {f('IP_dev')}   adv {f('IP_adv')}")
        rows.append({'dec':DEC,'measure':M,
                     'int_dev':m.params['IPxDec_dev'],'p_int_dev':m.pvalues['IPxDec_dev'],
                     'int_adv':m.params['IPxDec_adv'],'p_int_adv':m.pvalues['IPxDec_adv'],
                     'ip_dev':m.params['IP_dev'],'p_ip_dev':m.pvalues['IP_dev'],
                     'ip_adv':m.params['IP_adv'],'p_ip_adv':m.pvalues['IP_adv'],'N':int(m.nobs)})
pd.DataFrame(rows).to_csv("longdiff_split_results.csv",index=False)
