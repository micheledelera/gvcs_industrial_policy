"""A4 figure -- Xu (2017) Figure 3 for this panel: estimated factors, and the overlap of
treated and control factor loadings. The loading plot is the GSC analogue of the convex-hull
check, and Xu names it as essential because GSC extrapolates silently where canonical
synthetic control fails visibly.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
Lc=np.load("a4_load_ctrl.npy"); Lt=np.load("a4_load_trt.npy")
F=np.load("a4_factors.npy"); YRS=list(range(2007,2025))
sd=Lc.std(axis=0)
SUR="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e3e2df"
C_TR="#2a78d6"; C_CO="#b8b6b1"; C_F1="#2a78d6"; C_F2="#eb6834"
fig=plt.figure(figsize=(13.2,6.6),facecolor=SUR)
gs=fig.add_gridspec(1,2,left=0.062,right=0.980,top=0.660,bottom=0.215,wspace=0.24)
ax=fig.add_subplot(gs[0,0]); bx=fig.add_subplot(gs[0,1])
for a in (ax,bx):
    a.set_facecolor(SUR)
    for s in ('top','right'): a.spines[s].set_visible(False)
    for s in ('left','bottom'): a.spines[s].set_color(GRID)
    a.grid(color=GRID,lw=0.8); a.set_axisbelow(True)
    a.tick_params(colors=INK2,labelsize=9.5,length=0)
# --- factors
ax.axhline(0,color=INK2,lw=0.9); ax.axvline(2017.5,color=INK2,lw=1.0,ls=(0,(4,3)))
for d,(c,lab) in enumerate([(C_F1,"factor 1"),(C_F2,"factor 2")]):
    ax.plot(YRS,F[:,d]*sd[d],color=c,lw=2.1,solid_capstyle='round',label=lab)
    ax.plot(YRS,F[:,d]*sd[d],'o',color=c,ms=4,mec=SUR,mew=1.2)
ax.set_xticks([2007,2011,2015,2019,2024]); ax.set_xlim(2006.4,2024.6)
ax.set_ylabel("factor, scaled by its loading sd",color=INK2,fontsize=9.5)
ax.set_title("Estimated factors",color=INK,fontsize=11.5,loc='left',pad=46)
ax.text(0,1.02,"factor 1 is a smooth secular trend -- heterogeneous share trends, which"
        "\ntwo-way fixed effects cannot absorb; neither factor breaks at 2018",
        transform=ax.transAxes,color=INK2,fontsize=9,va='bottom',linespacing=1.55)
ax.legend(frameon=False,loc='upper left',fontsize=9.5,labelcolor=INK,handlelength=1.5)
ax.annotate("2018",(2017.5,ax.get_ylim()[0]),xytext=(4,8),textcoords='offset points',
            color=INK2,fontsize=8.6)
# --- loadings
h=ConvexHull(Lc)
for sx in h.simplices:
    bx.plot(Lc[sx,0],Lc[sx,1],color="#8e8c87",lw=1.0,alpha=0.8,zorder=2)
bx.scatter(Lc[:,0],Lc[:,1],s=7,color=C_CO,alpha=0.55,lw=0,label=f"clean controls ({len(Lc):,})")
bx.scatter(Lt[:,0],Lt[:,1],s=9,color=C_TR,alpha=0.65,lw=0,label=f"treated ({len(Lt):,})")
bx.set_xlabel("loading on factor 1",color=INK2,fontsize=9.5)
bx.set_ylabel("loading on factor 2",color=INK2,fontsize=9.5)
bx.set_title("Factor-loading overlap",color=INK,fontsize=11.5,loc='left',pad=46)
bx.text(0,1.02,"97.0% of treated units lie strictly inside the control cloud's convex hull;"
        "\nthe median treated unit sits at the 49.5th percentile of the control distribution",
        transform=bx.transAxes,color=INK2,fontsize=9,va='bottom',linespacing=1.55)
bx.legend(frameon=False,loc='upper left',fontsize=9,labelcolor=INK,markerscale=2.2)
fig.text(0.010,0.975,"A4 common support: the treated country-sectors are typical of the "
         "donors, not extreme",ha='left',va='top',color=INK,fontsize=14.5)
fig.text(0.010,0.912,
 "Outcome is ln(i's share of US imports in sector k); 1,083 persistently-targeted "
 "country-sectors against 2,223 strictly clean controls, 2007-2024, r = 2.\nThe interactive "
 "fixed effects model is fitted on control data only and each treated unit's loadings come "
 "from its eleven pre-2018 outcomes, so no treated post-treatment outcome is used.",
 ha='left',va='top',color=INK2,fontsize=8.9,linespacing=1.7)
fig.text(0.010,0.115,
 "This is the diagnostic §4a failed at country level, where there was no untreated Vietnam "
 "and canonical synthetic control was feasible for 5-20% of treated trade value. At "
 "country-sector level with a\nscale-free outcome the support problem does not bind: hull "
 "containment is 99.6% at r = 1, 97.0% at r = 2, and falls to 84.7% and 83.1% at r = 3 and 4 "
 "as the dimension rises.",
 ha='left',va='top',color=INK2,fontsize=8.6,linespacing=1.7)
fig.savefig("a4_support.png",dpi=200,facecolor=SUR)
print("saved a4_support.png")
