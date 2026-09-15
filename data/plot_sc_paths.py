"""The synthetic-control figure: treated country-sectors against their synthetic
counterpart, 2010-2024.

For each sector k, omega is the §3aa/§3ad estimator (MVA_W = 2, zeta -> 0, matching the
2010-2017 log US export path plus MVA). Then

  treated_kt = mean over treated pairs of log X_ikt
  synth_kt   = omega' log X_dkt + omega_0        omega_0 fitted over the path years

Both series are then expressed relative to the TREATED group's own 2015-17 mean, so the
pre-period gap stays visible rather than being normalised away, and averaged across
sectors. Two panels: all 58 sectors, and the best-fitting third (§3ad), where the placebo
is essentially exactly zero.

Bands are +/- 1 jackknife se across sectors. Pre-period 2015-17 and post-period 2022-24
are shaded; 2018-2021 is the transition the estimator excludes from both windows.
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
exec(open("fit_sc_mvasweep2.py").read().split("def run(mw):")[0])
MW=2.0; ZS=1e-6
YRS=[y for y in years if 2010<=y<=2024]

rows_tr, rows_sy, keys, rmses = [], [], [], []
for k,g in m.groupby(level='ISIC4c', observed=True):
    t_,d_=g[g['T']==1],g[g['T']==0]
    if len(t_)==0 or len(d_)<MIN_DON: continue
    P=Yv.loc[d_.index,PATH].values; bt=Yv.loc[t_.index,PATH].values.mean(axis=0)
    A=np.hstack([P,(MW*d_['mvaz'].values)[:,None]]); b=np.append(bt,MW*t_['mvaz'].mean())
    sig=np.diff(P,axis=1).std()
    z2=(((len(t_)*len(POST))**0.5)*(sig**2))*ZS+1e-10
    w,_=solve_w2(A,b,z2,len(PATH))
    fit=P.T@w; w0=float(np.mean(bt-fit))
    tr=Yv.loc[t_.index,YRS].values.mean(axis=0)
    sy=Yv.loc[d_.index,YRS].values.T@w + w0
    base=tr[[YRS.index(y) for y in PRE]].mean()
    rows_tr.append(tr-base); rows_sy.append(sy-base); keys.append(k)
    rmses.append(float(np.sqrt(((fit+w0-bt)**2).mean())))
TR=np.array(rows_tr); SY=np.array(rows_sy); R=np.array(rmses)
print(f"{len(TR)} sectors")

def band(M):
    n=len(M); mu=M.mean(axis=0)
    jkm=np.array([np.delete(M,i,axis=0).mean(axis=0) for i in range(n)])
    se=np.sqrt((n-1)/n*((jkm-jkm.mean(axis=0))**2).sum(axis=0))
    return mu, se

SUR="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e3e2df"
C_TR="#2a78d6"; C_SY="#eb6834"
fig,axes=plt.subplots(1,2,figsize=(12.4,5.4),sharey=True,facecolor=SUR)
sel_best=R<=np.quantile(R,1/3)
for ax,(sel,title) in zip(axes,[(np.ones(len(R),bool),
        f"All sectors  (n={len(R)})"),
        (sel_best, f"Best-fitting third  (n={int(sel_best.sum())})")]):
    ax.set_facecolor(SUR)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    for s in ('left','bottom'): ax.spines[s].set_color(GRID)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.axvspan(2015,2017,color="#000000",alpha=0.035,lw=0)
    ax.axvspan(2022,2024,color="#000000",alpha=0.035,lw=0)
    ax.axvline(2017.5,color=INK2,lw=1.0,ls=(0,(4,3)))
    for M,c,lab in [(TR[sel],C_TR,"Policy-targeted"),(SY[sel],C_SY,"Synthetic control")]:
        mu,se=band(M)
        ax.fill_between(YRS,mu-se,mu+se,color=c,alpha=0.16,lw=0)
        ax.plot(YRS,mu,color=c,lw=2.0,solid_capstyle='round',zorder=3,label=lab)
        ax.plot([YRS[-1]],[mu[-1]],'o',color=c,ms=6,mec=SUR,mew=2,zorder=4)
        ax.annotate(f"{mu[-1]:+.2f}",(YRS[-1],mu[-1]),xytext=(7,0),
                    textcoords='offset points',color=INK,fontsize=10,va='center')
    gap=TR[sel].mean(axis=0)[-3:].mean()-SY[sel].mean(axis=0)[-3:].mean()
    ax.set_title(title,color=INK,fontsize=12,loc='left',pad=14)
    ax.text(0,1.005,f"2022-24 gap  {gap:+.3f} log points",transform=ax.transAxes,
            color=INK2,fontsize=10,va='bottom')
    ax.set_xlim(2009.6,2025.6); ax.set_xticks([2010,2013,2016,2019,2022,2024])
    ax.tick_params(colors=INK2,labelsize=10,length=0)
axes[0].set_ylabel("log US exports\n(relative to treated 2015-17 mean)",
                   color=INK2,fontsize=10,linespacing=1.5)
axes[0].annotate("treatment window\nbegins 2018",(2017.5,axes[0].get_ylim()[0]),
                 xytext=(-6,14),textcoords='offset points',ha='right',
                 color=INK2,fontsize=9,linespacing=1.35)
axes[0].legend(frameon=False,loc='upper left',fontsize=10,labelcolor=INK,
               handlelength=1.6,borderpad=0.2)
fig.suptitle("Policy-targeted country-sectors and their synthetic counterparts",
             x=0.012,y=0.982,ha='left',color=INK,fontsize=14)
fig.text(0.012,0.020,
    "Developing economies excluding China, 58 ISIC-4 sectors where China's share of US "
    "imports fell 25%+ from its 2015-17 level.\nSynthetic control matches the 2010-17 log "
    "US export path and MVA/GDP. Bands are +/-1 jackknife se across sectors.",
    color=INK2,fontsize=9,linespacing=1.5)
fig.tight_layout(rect=[0.008,0.135,0.995,0.925])
fig.savefig("sc_paths.png",dpi=200,facecolor=SUR)
print("saved sc_paths.png")
for lab,sel in [("all",np.ones(len(R),bool)),("best third",sel_best)]:
    g=TR[sel].mean(axis=0)-SY[sel].mean(axis=0)
    print(f"  {lab:11s} gap: 2010 {g[0]:+.3f}  2015-17 {g[[YRS.index(y) for y in PRE]].mean():+.3f}"
          f"  2021 {g[YRS.index(2021)]:+.3f}  2022-24 {g[-3:].mean():+.3f}")
