import pandas as pd, numpy as np, pyfixest as pf, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
CHINA, BASE = 156, 2017
agg = pd.read_pickle("agg_for_estimation.pkl")
agg['i_int']=agg['i'].astype('int32'); agg['t_int']=agg['t'].astype('int32')
base = agg[agg['US_trade']==1]
base = base[(base['i_int']!=CHINA)&(base['Advanced_i']==0)].copy()

for MEASURE, LAG in [('share_frac_policies',3), ('share_n_policies',1)]:
    agg[MEASURE]=pd.to_numeric(agg[MEASURE],errors='coerce').fillna(0).astype('float32')
    look=agg[['i_int','ISIC4c','t_int',MEASURE]].drop_duplicates(subset=['i_int','ISIC4c','t_int']).copy()
    look['t_int']=look['t_int']+LAG; look=look.rename(columns={MEASURE:'IP_lag'})
    us=base.merge(look,on=['i_int','ISIC4c','t_int'],how='left'); us=us[us['IP_lag'].notna()].copy()
    us['IP_z']=(us['IP_lag']/us['IP_lag'].std()).astype('float32')
    us['fe_st']=us.groupby(['ISIC4c','t'],observed=True).ngroup().astype('int32')
    us['fe_is']=us.groupby(['i','ISIC4c'],observed=True).ngroup().astype('int32')
    us['fe_it']=us.groupby(['i','t'],observed=True).ngroup().astype('int32')
    pre=sorted(y for y in us['t_int'].unique() if y<BASE); allу=sorted(y for y in us['t_int'].unique() if y!=BASE)
    tri=[]
    for y in pre:
        us[f'TGT_y{y}']=(us['IP_z']*us['target']*(us['t_int']==y)).astype('float32'); tri.append(f'TGT_y{y}')
    us['TGT_post']=(us['IP_z']*us['target']*(us['t_int']>=2018)).astype('float32'); tri.append('TGT_post')
    own=[]
    for y in allу:
        us[f'IP_y{y}']=(us['IP_z']*(us['t_int']==y)).astype('float32'); own.append(f'IP_y{y}')
    f=f"imports ~ {' + '.join(tri+own)} | fe_st + fe_is + fe_it"
    log(f"=== {MEASURE} lag{LAG} ===")
    fit=pf.fepois(f,data=us,vcov={"CRV1":"i_int"},demeaner=pf.LsmrDemeaner(fixef_maxiter=2000))
    t=fit.tidy()
    print(f"  CRV1: coef={t.loc['TGT_post','Estimate']:+.4f} se={t.loc['TGT_post','Std. Error']:.4f} p={t.loc['TGT_post','Pr(>|t|)']:.4f}")
    for vc in ["CRV3"]:
        try:
            fit.vcov({vc: "i_int"}); tt = fit.tidy()
            print(f"  {vc}: coef={tt.loc['TGT_post','Estimate']:+.4f} "
                  f"se={tt.loc['TGT_post','Std. Error']:.4f} p={tt.loc['TGT_post','Pr(>|t|)']:.4f}")
        except Exception as e:
            log(f"  {vc} failed: {type(e).__name__}: {e}")
    del fit
log("DONE")
