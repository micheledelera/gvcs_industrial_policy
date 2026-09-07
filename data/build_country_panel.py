"""Country-level panel for the differential-exposure DiD.

Design: decoupling is a single shock dated 2018, so this is an Autor-Dorn-Hanson
style long difference on PRE-DETERMINED exposure, not a staggered-adoption DiD.

  pre  = 2015-2017      post = 2022-2024   (2022+ avoids the COVID disruption)
  unit = developing country excluding China, top 60 by pre-period US-bound exports

Variables, all pre-period so nothing is a function of the outcome:

  Exposure_i  = sum_s w_is x ChinaUSshare_s      w_is  = sector s share of i's exports
  Alignment_i = sum_s pi_is x ChinaUSshare_s     pi_is = sector s share of i's POLICY effort

Exposure asks how much of a country's export basket sat in sectors China dominated
in the US market -- i.e. how much vacated share was available to it. Alignment asks
whether it aimed its industrial policy at those sectors. Alignment is the country-level
analogue of the share_ measures and the sharper test of the targeting story: a country
can be highly exposed and policy-active while aiming its policy elsewhere.

Capability controls from BACI: log total exports, ECI, mean PCI of the export basket,
US share of own exports, share of US market held. Plus log MVA per capita (UNIDO).

NOTE on ECI/PCI: computed over 125 ISIC-4 sectors rather than ~5000 HS6 products, so
they are coarser and more compressed than the published indices. Directionally useful,
not comparable in levels to Atlas figures.

NOTE on country codes: BACI uses COMTRADE codes, the UNIDO workbook uses true ISO
numeric. They differ for several countries (India 699 vs 356, Switzerland 757 vs 756,
France 251 vs 250, Norway 579 vs 578). Unmatched top-60 countries are REPORTED, never
silently dropped -- this is the discrepancy that misclassified Switzerland and Norway
earlier in this project.
"""
import pandas as pd, numpy as np, gc, time

t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

CHINA, USA = 156, 842
PRE, POST = [2015,2016,2017], [2022,2023,2024]
XLSX = ("/root/.claude/uploads/618110ef-d894-58be-8e00-e7f2bf548b03/"
        "4ad6510b-0154963EF4289D02DEA410D8D38E4E6A8C6D.xlsx")
# COMTRADE -> ISO numeric, for codes where they differ
C2ISO = {699:356, 757:756, 579:578, 251:250, 842:840, 381:380, 490:158, 729:729}

raw = pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32')
raw['t']=raw['t'].astype('int32'); raw['imports']=raw['imports'].astype('float64')
raw['n_policies']=pd.to_numeric(raw['n_policies'],errors='coerce').fillna(0).astype('float32')
adv_map = raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
log(f"loaded {raw.shape}")

def agg(years):
    d = raw[raw['t'].isin(years)]
    tot = d.groupby(['i','ISIC4c'], observed=True)['imports'].sum()/len(years)   # to world
    us  = d[d['j']==USA].groupby(['i','ISIC4c'], observed=True)['imports'].sum()/len(years)
    return tot, us
tot_pre, us_pre   = agg(PRE)
tot_post, us_post = agg(POST)
log("aggregated pre and post")

# --- China's share of US imports, by sector, pre-period ---
us_by_s_pre = us_pre.groupby('ISIC4c', observed=True).sum()
chn_us_pre  = us_pre.loc[CHINA] if CHINA in us_pre.index.get_level_values(0) else None
china_share_s = (chn_us_pre / us_by_s_pre).fillna(0).clip(0,1)
log(f"China US share by sector: mean {china_share_s.mean():.3f}, max {china_share_s.max():.3f}")

# --- export weights and policy weights, pre ---
X = tot_pre.unstack(fill_value=0.0)                       # country x sector, exports
w = X.div(X.sum(axis=1).replace(0,np.nan), axis=0).fillna(0)

pol = (raw[raw['t'].isin(PRE)][['i','ISIC4c','t','n_policies']]
       .drop_duplicates(subset=['i','ISIC4c','t'])
       .groupby(['i','ISIC4c'], observed=True)['n_policies'].sum().unstack(fill_value=0.0))
pol = pol.reindex(index=X.index, columns=X.columns, fill_value=0.0)
pi  = pol.div(pol.sum(axis=1).replace(0,np.nan), axis=0).fillna(0)
log(f"policy matrix: {int((pol.sum(axis=1)>0).sum())} countries with any policy in {PRE}")

cs = china_share_s.reindex(X.columns).fillna(0)
Exposure  = w.mul(cs, axis=1).sum(axis=1)
Alignment = pi.mul(cs, axis=1).sum(axis=1)

# --- ECI / PCI from the RCA matrix (all exporters, method of reflections eigenvector) ---
sh = X.div(X.sum(axis=1).replace(0,np.nan), axis=0)
world = X.sum(axis=0)/X.sum().sum()
RCA = sh.div(world, axis=1)
M = (RCA >= 1).astype(float)
M = M.loc[M.sum(axis=1)>0, M.sum(axis=0)>0]
kc, kp = M.sum(axis=1), M.sum(axis=0)
Mcc = (M.div(kc,axis=0)) @ (M.div(kp,axis=1).T)
vals, vecs = np.linalg.eig(Mcc.values)
K = np.real(vecs[:, np.argsort(-np.real(vals))[1]])
ECI = pd.Series((K-K.mean())/K.std(), index=M.index)
if ECI.corr(pd.Series(kc, index=M.index)) < 0: ECI = -ECI            # sign convention
Mpp = (M.div(kp,axis=1).T) @ (M.div(kc,axis=0).T.T)
vals2, vecs2 = np.linalg.eig(Mpp.values)
Q = np.real(vecs2[:, np.argsort(-np.real(vals2))[1]])
PCI = pd.Series((Q-Q.mean())/Q.std(), index=M.columns)
if PCI.corr(pd.Series(-kp, index=M.columns)) < 0: PCI = -PCI
meanPCI = w.reindex(columns=PCI.index).fillna(0).mul(PCI, axis=1).sum(axis=1)
log(f"ECI computed for {len(ECI)} countries; PCI for {len(PCI)} sectors")

# --- assemble ---
us_c_pre, us_c_post = us_pre.groupby('i').sum(), us_post.groupby('i').sum()
tot_c_pre = tot_pre.groupby('i').sum()
us_total_pre  = us_c_pre.sum()
us_total_post = us_c_post.sum()

df = pd.DataFrame({
    'us_pre': us_c_pre, 'us_post': us_c_post, 'tot_pre': tot_c_pre,
    'Exposure': Exposure, 'Alignment': Alignment,
    'IP_total': pol.sum(axis=1), 'IP_sectors': (pol>0).sum(axis=1),
    'ECI': ECI, 'meanPCI': meanPCI, 'Advanced': adv_map,
}).dropna(subset=['us_pre'])
df['us_share_of_own']  = df['us_pre']/df['tot_pre']
df['share_of_us_mkt']  = df['us_pre']/us_total_pre
df['dlog_us']          = np.log(df['us_post'].replace(0,np.nan)) - np.log(df['us_pre'].replace(0,np.nan))
df['dshare_us']        = df['us_post']/us_total_post - df['us_pre']/us_total_pre
df['log_tot_pre']      = np.log(df['tot_pre'].replace(0,np.nan))

dev = df[(df['Advanced']==0) & (df.index!=CHINA)].copy()
dev = dev.sort_values('us_pre', ascending=False).head(60)
log(f"top-60 developing ex-China selected; US exports {dev['us_pre'].min():,.0f} to {dev['us_pre'].max():,.0f}")

# --- merge UNIDO MVA per capita ---
u = pd.read_excel(XLSX, sheet_name='Data')
u = u[(u['VariableCode']=='NV_IND_MANFPC') & (u['Year'].isin(PRE))]
mva = u.groupby('CountryCode')['Value'].mean()
dev['iso'] = [C2ISO.get(c, c) for c in dev.index]
dev['mva_pc'] = dev['iso'].map(mva)
dev['log_mva_pc'] = np.log(dev['mva_pc'])
miss = dev[dev['mva_pc'].isna()]
log(f"MVA per capita merged: {dev['mva_pc'].notna().sum()}/60")
if len(miss):
    print("UNMATCHED (comtrade code, US exports pre):")
    print(miss[['us_pre']].to_string())

dev.to_csv("country_panel.csv")
log(f"saved country_panel.csv {dev.shape}")
print("\n--- top 15 by pre-period US exports ---")
print(dev[['us_pre','Exposure','Alignment','IP_total','ECI','meanPCI','mva_pc','dlog_us']]
      .head(15).round(4).to_string())
