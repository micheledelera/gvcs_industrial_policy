"""Does the SDiD unit-weight solver actually achieve pre-period fit?

If omega is working, the donor-weighted pre-period path should track the treated mean
path far better than uniform weights do. If it is not, the placebo failure in fit_sdid.py
is my solver, not the data. Reports per-sector pre-period RMSE under omega vs uniform,
and the fitted pre-period trend gap that omega leaves behind.
"""
import pandas as pd, numpy as np, gc, importlib.util, sys
spec=importlib.util.spec_from_file_location("sd","fit_sdid.py")
src=open("fit_sdid.py").read().split("def sdid_sector")[0]
src=src.replace('MEASURE = sys.argv[1] if len(sys.argv)>1 else','MEASURE =')
exec(src)

rows=[]
for k,g in meta.groupby(level='ISIC4c', observed=True):
    e=int(g['E'].iloc[0]); P=[y for y in years if y<e]; Q=[y for y in years if y>=e]
    Yk=Y.loc[g.index]; Tk=g['T'].values
    tr=Yk[Tk==1]; dn=Yk[Tk==0]
    if len(tr)<3 or len(dn)<8 or len(P)<3 or len(Q)<1: continue
    Ap=dn[P].values; bt=tr[P].values.mean(axis=0)
    sig=np.diff(dn[P].values,axis=1).std()
    zeta2=((len(tr)*len(Q))**0.5)*(sig**2)+1e-8
    w,w0=solve_w(Ap,bt,zeta2)
    un=np.ones(len(dn))/len(dn)
    fit_w = Ap.T@w  + w0
    fit_u = Ap.T@un + np.mean(bt - Ap.T@un)
    rmse_w=np.sqrt(((fit_w-bt)**2).mean()); rmse_u=np.sqrt(((fit_u-bt)**2).mean())
    yr=np.array(P,dtype=float); yr=yr-yr.mean()
    slope_w=np.polyfit(yr, bt-fit_w, 1)[0]          # residual pre-trend left by omega
    slope_u=np.polyfit(yr, bt-fit_u, 1)[0]
    rows.append({'k':k,'n_tr':len(tr),'n_don':len(dn),'npre':len(P),
                 'rmse_omega':rmse_w,'rmse_uniform':rmse_u,
                 'resid_slope_omega':slope_w,'resid_slope_uniform':slope_u,
                 'w_max':w.max(),'w_eff_n':1.0/(w**2).sum()})
r=pd.DataFrame(rows); r.to_csv("sdid_fit_diag.csv",index=False)
print(f"{len(r)} sectors\n")
print("pre-period fit of the donor-weighted path to the treated mean path (log points):")
print(f"  RMSE under omega   : median {r['rmse_omega'].median():.4f}   "
      f"mean {r['rmse_omega'].mean():.4f}")
print(f"  RMSE under uniform : median {r['rmse_uniform'].median():.4f}   "
      f"mean {r['rmse_uniform'].mean():.4f}")
print(f"  omega beats uniform in {int((r['rmse_omega']<r['rmse_uniform']).sum())}/{len(r)} sectors "
      f"({100*(r['rmse_omega']<r['rmse_uniform']).mean():.0f}%)")
print(f"  median ratio omega/uniform: {(r['rmse_omega']/r['rmse_uniform']).median():.3f}")
print("\nRESIDUAL PRE-TREND left behind (log points per year) -- this is what leaks into tau:")
print(f"  under omega   : median {r['resid_slope_omega'].median():+.5f}   "
      f"mean {r['resid_slope_omega'].mean():+.5f}   "
      f"mean |.| {r['resid_slope_omega'].abs().mean():.5f}")
print(f"  under uniform : median {r['resid_slope_uniform'].median():+.5f}   "
      f"mean {r['resid_slope_uniform'].mean():+.5f}")
print(f"\nweight concentration: median max weight {r['w_max'].median():.3f}, "
      f"median effective donors {r['w_eff_n'].median():.1f} of {r['n_don'].median():.0f}")
