"""Country-level differential-exposure DiD: does well-aimed industrial policy pay?

Long difference, pre 2015-17 -> post 2022-24, top 60 developing ex-China.

Alignment is exactly 0 whenever a country has no recorded policy (27 of 60), so
entering it raw would make it near-collinear with a has-policy dummy. Split the
two margins instead:

  HasPolicy   extensive: does a policy country differ from a non-policy one at all?
  AlignDev    intensive: HasPolicy x (Alignment - mean Alignment | HasPolicy).
              Among countries that DO have policy, does better-aimed policy do
              better? This is the coefficient the targeting story predicts.

Exposure enters as a control throughout: Alignment and Exposure are both weighted
averages of the same China-share vector, so without Exposure the Alignment
coefficient would partly just be exposure.

Heteroskedasticity-robust SEs. N=60, so these are suggestive, not precise.
"""
import pandas as pd, numpy as np, statsmodels.api as sm

d = pd.read_csv("country_panel.csv", index_col=0)
d['HasPolicy'] = (d['IP_total'] > 0).astype(float)
mA = d.loc[d['HasPolicy']==1, 'Alignment'].mean()
d['AlignDev'] = d['HasPolicy'] * (d['Alignment'] - mA)
for c in ['AlignDev','Exposure','ECI','meanPCI','log_mva_pc','log_tot_pre',
          'us_share_of_own','share_of_us_mkt']:
    d[c+'_z'] = (d[c] - d[c].mean())/d[c].std()

print(f"N = {len(d)}   policy countries = {int(d['HasPolicy'].sum())}   "
      f"mean Alignment | policy = {mA:.4f}")
sub = d[d['HasPolicy']==1]
print(f"Alignment among policy countries: sd {sub['Alignment'].std():.4f}, "
      f"min {sub['Alignment'].min():.3f}, max {sub['Alignment'].max():.3f}")
print(f"corr(Alignment, Exposure) among policy countries: "
      f"{sub['Alignment'].corr(sub['Exposure']):.3f}")
print(f"corr(HasPolicy, AlignDev): {d['HasPolicy'].corr(d['AlignDev']):.3f}\n")

CAP = ['ECI_z','meanPCI_z','log_mva_pc_z','log_tot_pre_z',
       'us_share_of_own_z','share_of_us_mkt_z']
SPECS = [
    ("1. treatment only",   ['HasPolicy','AlignDev_z']),
    ("2. + Exposure",       ['HasPolicy','AlignDev_z','Exposure_z']),
    ("3. + capability",     ['HasPolicy','AlignDev_z','Exposure_z'] + CAP),
]
rows=[]
for outcome in ['dlog_us','dshare_us']:
    print("="*74); print(f"OUTCOME: {outcome}"); print("="*74)
    for name, X in SPECS:
        dd = d[[outcome]+X].dropna()
        m = sm.OLS(dd[outcome], sm.add_constant(dd[X])).fit(cov_type='HC1')
        print(f"\n--- {name}   N={int(m.nobs)}   R2={m.rsquared:.3f} ---")
        t = pd.DataFrame({'coef':m.params,'se':m.bse,'t':m.tvalues,'p':m.pvalues})
        print(t.round(4).to_string())
        for k in ['HasPolicy','AlignDev_z']:
            rows.append({'outcome':outcome,'spec':name,'term':k,'N':int(m.nobs),
                         'coef':m.params[k],'se':m.bse[k],'p':m.pvalues[k]})
pd.DataFrame(rows).to_csv("did_country_results.csv", index=False)
print("\n" + "="*74)
print("KEY COEFFICIENTS")
print("="*74)
print(pd.DataFrame(rows).pivot_table(index=['outcome','spec'],columns='term',
      values=['coef','p']).round(4).to_string())
