"""§5i figure. The §5h pre-trend is induced by the comparison group, not by policy.

Same event study in every panel -- ln US exports on HIGH x year dummies, unit and
sector-by-year fixed effects, 2017 omitted, clustered on country. The ONLY thing that
changes across panels is who the HIGH units are compared to:

  left    all 3,658 LOW-policy country-sectors
  middle  the 3,583 that pass the synthetic control's band rule (different country,
          Chinese share within 10pp, log size within 2.0) -- eligible, but not chosen
  right   the 1,484 that actually received positive synthetic-control weight, which is
          the set §5h used. Donors earn weight by fitting the treated units' pre-2018
          paths, so this set is selected on the outcome's pre-period.

Selecting the comparison group on pre-period fit is what creates the U. On the full
LOW group the pre-period is flat and the post-2018 gap is the same size.
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
E=pd.read_csv("sc19d_locate.csv")
E=E[(E['meas']=='tgt')&(E['fe']=="unit + sector x year")]
PANELS=[("all LOW","All low-policy units","3,658 controls"),
        ("band-eligible","Band-eligible donors","3,583 controls"),
        ("SC weighted donors","Donors with SC weight","1,484 controls — the §5h panel")]
SUR="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e3e2df"
C="#2a78d6"; CBAD="#d13f4a"
fig=plt.figure(figsize=(14.2,6.4),facecolor=SUR)
gs=fig.add_gridspec(1,3,left=0.060,right=0.980,top=0.660,bottom=0.230,wspace=0.10)
ax0=fig.add_subplot(gs[0,0]); axs=[ax0]+[fig.add_subplot(gs[0,c],sharey=ax0) for c in (1,2)]
for ax,(key,ttl,sub) in zip(axs,PANELS):
    d=E[E['ctrl']==key].sort_values('t'); c=CBAD if key=="SC weighted donors" else C
    ax.set_facecolor(SUR)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    for s in ('left','bottom'): ax.spines[s].set_color(GRID)
    ax.grid(axis='y',color=GRID,lw=0.8); ax.set_axisbelow(True)
    ax.axhline(0,color=INK2,lw=0.9); ax.axvline(2017.5,color=INK2,lw=1.0,ls=(0,(4,3)))
    ax.axvspan(2015,2017,color="#000000",alpha=0.04,lw=0)
    ax.fill_between(d['t'],d['b']-1.96*d['s'],d['b']+1.96*d['s'],color=c,alpha=0.15,lw=0)
    ax.plot(d['t'],d['b'],color=c,lw=1.8,zorder=3)
    ax.plot(d['t'],d['b'],'o',color=c,ms=4.5,mec=SUR,mew=1.4,zorder=4)
    pre=d[(d['t']<2018)&(d['t']!=2017)]; post=d[d['t']>=2018]
    ax.set_title(ttl,color=INK,fontsize=11.5,loc='left',pad=58)
    ax.text(0,1.012,f"{sub}\npre-2018 mean {pre['b'].mean():+.3f}, max |t| "
            f"{np.abs(pre['b']/pre['s']).max():.2f}\npost-2018 mean "
            f"{post['b'].mean():+.3f}",transform=ax.transAxes,color=INK2,fontsize=9,
            va='bottom',linespacing=1.6)
    ax.set_xlim(2006.4,2024.6); ax.set_xticks([2007,2011,2015,2019,2024])
    ax.tick_params(colors=INK2,labelsize=9.5,length=0)
axs[0].set_ylabel("HIGH minus LOW, relative to 2017\n(log US exports)",
                  color=INK2,fontsize=9.5,linespacing=1.6)
fig.text(0.010,0.980,"The pre-trend is in the comparison group, not in the policy",
         ha='left',va='top',color=INK,fontsize=15)
fig.text(0.010,0.930,
 "Targeting, top quartile of the positive 2015-17 distribution, against three comparison "
 "groups. Identical specification throughout: ln US exports on HIGH x year dummies,\nunit "
 "and sector-by-year fixed effects, 2017 omitted, standard errors clustered on country. "
 "Only the control group changes.",
 ha='left',va='top',color=INK2,fontsize=8.9,linespacing=1.75)
fig.text(0.010,0.150,
 "Synthetic-control donors earn their weight by fitting the treated units' pre-2018 paths, "
 "so the right-hand panel conditions the control group on the outcome's own pre-period. "
 "That is what\nproduces the U reported in §5h -- it roughly triples the pre-2018 mean "
 "(+0.056 to +0.160) while leaving the post-2018 gap about the same. On the full low-policy "
 "group the pre-period is flat.",
 ha='left',va='top',color=INK2,fontsize=8.6,linespacing=1.75)
fig.savefig("sc19d_prepanel.png",dpi=200,facecolor=SUR)
print("saved sc19d_prepanel.png")
