"""Publication-clean version of the lnX / sector-blocked figure: white ground, no titles or
annotation, axis labels and a two-entry legend only.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
P=np.load("b2_paths_lnX_sector.npy")
trt,cf,att,se=P[0],P[1],P[2],P[3]
YRS=list(range(2007,2025)); T0=11
INK="#000000"; GRID="#dcdcdc"; C_TR="#1f4e9c"; C_CF="#c8500f"
plt.rcParams.update({"font.size":10,"axes.linewidth":0.8})
fig=plt.figure(figsize=(11.6,4.5),facecolor="white")
gs=fig.add_gridspec(1,2,left=0.062,right=0.988,top=0.965,bottom=0.115,wspace=0.185)
ax=fig.add_subplot(gs[0,0]); bx=fig.add_subplot(gs[0,1])
for a in (ax,bx):
    a.set_facecolor("white")
    for s in ('top','right'): a.spines[s].set_visible(False)
    for s in ('left','bottom'): a.spines[s].set_color("#666666")
    a.grid(axis='y',color=GRID,lw=0.6); a.set_axisbelow(True)
    a.axvline(2017.5,color="#888888",lw=0.9,ls=(0,(4,3)))
    a.set_xlim(2006.6,2024.4); a.set_xticks([2007,2010,2013,2016,2019,2022])
    a.tick_params(colors=INK,labelsize=9.5,length=3,width=0.8,direction='out')
ax.plot(YRS,trt,color=C_TR,lw=1.9,label="Treated",zorder=3)
ax.plot(YRS,cf,color=C_CF,lw=1.9,ls=(0,(5,2)),label="Counterfactual",zorder=3)
ax.set_ylabel("ln US imports")
ax.legend(frameon=False,loc='upper left',fontsize=9.5,handlelength=2.2)
bx.axhline(0,color="#444444",lw=0.8)
bx.fill_between(YRS,att-1.96*se,att+1.96*se,color=C_TR,alpha=0.15,lw=0)
bx.plot(YRS,att,color=C_TR,lw=1.9,zorder=3)
bx.set_ylabel("Estimated effect, log points")
fig.savefig("fig_main_clean.png",dpi=400,facecolor="white")
fig.savefig("fig_main_clean.pdf",facecolor="white")
print("saved fig_main_clean.png / .pdf")
