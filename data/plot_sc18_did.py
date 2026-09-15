"""§5h figure. The DiD event study on the same two groups the synthetic control uses.

HIGH vs LOW policy country-sectors, unit and sector-by-year fixed effects, 2017 omitted.
The point of the picture is the shape of the PRE-treatment coefficients, which the
synthetic control cannot show: SC weights are chosen to make the pre-period gap flat, so
a flat SC pre-gap is a property of the fit. With equal weights the same two groups were
CONVERGING through 2007-2017 and then diverge again after 2018.
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
E=pd.read_csv("sc18_did.csv")
SUR="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e3e2df"
C1="#2a78d6"; C2="#eb6834"
LAB={'tgt':("Targeting: share of HS lines covered",C1,"503 HIGH vs 1,484 donor country-sectors"),
     'vol':("Volume: number of interventions",C2,"544 HIGH vs 1,193 donor country-sectors")}
fig=plt.figure(figsize=(12.6,6.6),facecolor=SUR)
gs=fig.add_gridspec(1,2,left=0.072,right=0.980,top=0.715,bottom=0.215,wspace=0.10)
axes=[fig.add_subplot(gs[0,0])]; axes.append(fig.add_subplot(gs[0,1],sharey=axes[0]))
for ax,m in zip(axes,['tgt','vol']):
    d=E[E['meas']==m].sort_values('t'); ttl,c,sub=LAB[m]
    ax.set_facecolor(SUR)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    for s in ('left','bottom'): ax.spines[s].set_color(GRID)
    ax.grid(axis='y',color=GRID,lw=0.8); ax.set_axisbelow(True)
    ax.axhline(0,color=INK2,lw=0.9)
    ax.axvline(2017.5,color=INK2,lw=1.0,ls=(0,(4,3)))
    ax.axvspan(2015,2017,color="#000000",alpha=0.04,lw=0)
    lo,hi=d['b']-1.96*d['s'],d['b']+1.96*d['s']
    ax.fill_between(d['t'],lo,hi,color=c,alpha=0.15,lw=0)
    ax.plot(d['t'],d['b'],color=c,lw=1.8,zorder=3)
    ax.plot(d['t'],d['b'],'o',color=c,ms=4.5,mec=SUR,mew=1.4,zorder=4)
    pre=d[(d['t']<2018)&(d['t']!=2017)]
    ax.set_title(ttl,color=INK,fontsize=11.5,loc='left',pad=46)
    ax.text(0,1.012,f"{sub}\npre-2018 coefficients average {pre['b'].mean():+.2f}, "
            f"max |t| {np.abs(pre['b']/pre['s']).max():.2f}",
            transform=ax.transAxes,color=INK2,fontsize=9,va='bottom',linespacing=1.6)
    ax.set_xlim(2006.4,2024.6); ax.set_xticks([2007,2011,2015,2019,2024])
    ax.tick_params(colors=INK2,labelsize=9.5,length=0)
axes[0].set_ylabel("HIGH minus LOW, relative to 2017\n(log US exports)",
                   color=INK2,fontsize=9.5,linespacing=1.6)
axes[0].annotate("2017 omitted;\nshading is the 2015-17\nwindow the SC matches on",
                 (2016,axes[0].get_ylim()[0]),xytext=(4,12),textcoords='offset points',
                 color=INK2,fontsize=8.6,linespacing=1.4)
fig.text(0.010,0.980,"The pre-period the synthetic control could not show",
         ha='left',va='top',color=INK,fontsize=15)
fig.text(0.010,0.915,
 "Event study on the §5b sample: ln US exports on HIGH x year dummies, unit and "
 "sector-by-year fixed effects, standard errors clustered on country.\nHIGH = top quartile "
 "of the positive 2015-17 policy distribution; the comparison group is the LOW-policy "
 "units that carry positive synthetic-control weight.",
 ha='left',va='top',color=INK2,fontsize=8.9,linespacing=1.75)
fig.text(0.010,0.150,
 "Both measures trace the same U: the HIGH-LOW differential shrinks from about +0.25 log "
 "points in 2007 to zero in 2015-17, then reopens after 2018. Anchoring 'before' on "
 "2015-17 -- which\nthe synthetic control does -- starts the comparison at the bottom of "
 "that dip, so part of the post-2018 gain is a return to the groups' earlier relative "
 "position rather than a new divergence.",
 ha='left',va='top',color=INK2,fontsize=8.6,linespacing=1.75)
fig.savefig("sc18_did_event.png",dpi=200,facecolor=SUR)
print("saved sc18_did_event.png")
