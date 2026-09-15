"""B4 + B5 for lnX -- log US imports, the A0 check outcome. Same layout as plot_b2.py so the
two figures can be read side by side. Left: treated mean vs GSC counterfactual, with the level
translated into dollars because unlike lnS this outcome has an interpretable scale. Right: the
ATT path with the country-blocked 95% band.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
P=np.load("b2_paths_lnX_sector.npy")
trt,cf,att,se=P[0],P[1],P[2],P[3]
YRS=list(range(2007,2025)); T0=11
SUR="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e3e2df"
C_TR="#2a78d6"; C_CF="#eb6834"
def usd(l):                      # imports are in THOUSANDS of USD
    v=np.exp(l)*1e3
    return f"${v/1e6:.1f}m" if v<1e9 else f"${v/1e9:.2f}bn"
fig=plt.figure(figsize=(13.4,6.4),facecolor=SUR)
gs=fig.add_gridspec(1,2,left=0.066,right=0.972,top=0.675,bottom=0.200,wspace=0.24)
ax=fig.add_subplot(gs[0,0]); bx=fig.add_subplot(gs[0,1])
for a in (ax,bx):
    a.set_facecolor(SUR)
    for s in ('top','right'): a.spines[s].set_visible(False)
    for s in ('left','bottom'): a.spines[s].set_color(GRID)
    a.grid(axis='y',color=GRID,lw=0.8); a.set_axisbelow(True)
    a.axvline(2017.5,color=INK2,lw=1.0,ls=(0,(4,3)))
    a.set_xlim(2006.4,2025.4); a.set_xticks([2007,2011,2015,2019,2024])
    a.tick_params(colors=INK2,labelsize=9.5,length=0)
for v,c,lab in [(trt,C_TR,"treated (1,083 country-sectors)"),
                (cf,C_CF,"GSC counterfactual")]:
    ax.plot(YRS,v,color=c,lw=2.1,solid_capstyle='round',label=lab,zorder=3)
    ax.plot(YRS,v,'o',color=c,ms=4,mec=SUR,mew=1.2,zorder=4)
    ax.annotate(f"{v[-1]:.2f}\n{usd(v[-1])}",(YRS[-1],v[-1]),xytext=(7,0),
                textcoords='offset points',color=c,fontsize=9,va='center',linespacing=1.4)
ax.set_ylabel("ln(US imports from the country-sector, thousands USD)",color=INK2,fontsize=9.5)
ax.set_title("Treated units and their GSC counterfactual, lnX",color=INK,fontsize=11.5,
             loc='left',pad=46)
ax.text(0,1.012,"the level is POSITIVE here because lnX is a log of dollars, not of a share:\n"
        f"{trt[0]:.2f} is {usd(trt[0])} of US imports for the mean treated country-sector",
        transform=ax.transAxes,color=INK2,fontsize=9,va='bottom',linespacing=1.6)
ax.legend(frameon=False,loc='lower right',fontsize=9.5,labelcolor=INK,handlelength=1.5)
bx.axhline(0,color=INK2,lw=0.9)
bx.fill_between(YRS,att-1.96*se,att+1.96*se,color=C_TR,alpha=0.16,lw=0)
bx.plot(YRS,att,color=C_TR,lw=2.1,zorder=3)
bx.plot(YRS,att,'o',color=C_TR,ms=4,mec=SUR,mew=1.2,zorder=4)
bx.axvspan(2019.5,2022.5,color="#000000",alpha=0.045,lw=0)
bx.set_ylabel("ATT, log points of imports",color=INK2,fontsize=9.5)
bx.set_title("The gap, with a SECTOR-blocked bootstrap band",color=INK,fontsize=11.5,
             loc='left',pad=46)
bx.text(0,1.012,f"post-2018 mean ATT {att[T0:].mean():+.3f}, blocked at 123 ISIC 4-digit "
        "sectors\nrather than 25 countries; 2021, 2022 and 2024 exclude zero on their own",
        transform=bx.transAxes,color=INK2,fontsize=9,va='bottom',linespacing=1.6)
bx.annotate("2020-22",(2021,bx.get_ylim()[1]*0.93),ha='center',color=INK2,fontsize=8.6)
fig.text(0.010,0.975,"lnX with the bootstrap blocked at SECTOR rather than country",
         ha='left',va='top',color=INK,fontsize=14.5)
fig.text(0.010,0.915,
 "Generalized synthetic control (Xu 2017) on 1,083 persistently-targeted developing "
 "country-sectors against 2,223 strictly clean controls, r = 2, covariate Dec_k x Post.\n"
 "Neither Xu nor Cunningham prescribes a clustering level: Cunningham uses randomisation "
 "inference and no standard errors, Xu's bootstrap assumes cross-sectional independence.",
 ha='left',va='top',color=INK2,fontsize=8.9,linespacing=1.7)
fig.text(0.010,0.128,
 "Blocked at 123 ISIC 4-digit sectors the ATT is significant: se 0.106, t = +2.21, p = 0.029 on "
 "a t with 122 df, CI [+3%, +55%], and 2021, 2022 and 2024 each exclude zero on their own.\n"
 "At 24 ISIC 2-digit divisions se 0.120, p = 0.064. At 25 countries se 0.239, p = 0.338 and no "
 "year excludes zero -- the same estimate and the same cluster count as the division cut, so "
 "the difference is not\ndegrees of freedom but the shape of the correlated shocks. Which level "
 "is right is a judgement about the sampling frame, not something the data settles.",
 ha='left',va='top',color=INK2,fontsize=8.6,linespacing=1.7)
fig.savefig("b2_paths_lnX_sector.png",dpi=200,facecolor=SUR)
print("saved b2_paths_lnX_sector.png  post ATT %+.4f"%att[T0:].mean())
