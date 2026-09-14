"""§5j figure. The continuous, pre-determined specification and its tail dependence.

Left  the dose-response coefficient under the saturated FE (unit + sector x year +
      country x year) as the dose is made progressively tail-insensitive. The headline
      +0.029 (t = 2.24) is the top bar; everything below it is the same specification with
      the upper tail of the policy distribution pulled in or removed.
Right the decile step function: 2015-17 policy deciles among users, zero-policy units
      omitted, same FE. A genuine dose-response should slope upward.
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
T=pd.read_csv("sc20c_tail.csv"); T=T[T['meas']=='tgt']
S=T[T['kind']=='spec'].copy(); D=T[T['kind']=='decile'].copy()
ORD=["dose in sd (headline)","winsorised at p99 of positive","winsorised at p95 of positive",
     "winsorised at p90 of positive","percentile rank","top 1% of dosed units dropped",
     "top 5% of dosed units dropped"]
NICE={"dose in sd (headline)":"dose in sd  (headline)",
      "winsorised at p99 of positive":"winsorised at p99",
      "winsorised at p95 of positive":"winsorised at p95",
      "winsorised at p90 of positive":"winsorised at p90",
      "percentile rank":"percentile rank",
      "top 1% of dosed units dropped":"top 1% of users dropped",
      "top 5% of dosed units dropped":"top 5% of users dropped"}
S['o']=S['tag'].map({t:i for i,t in enumerate(ORD)}); S=S.sort_values('o')
SUR="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e3e2df"
C="#2a78d6"; CG="#8e8c87"
fig=plt.figure(figsize=(13.4,6.2),facecolor=SUR)
gs=fig.add_gridspec(1,2,left=0.205,right=0.975,top=0.700,bottom=0.215,wspace=0.30,
                    width_ratios=[1.0,1.0])
ax=fig.add_subplot(gs[0,0]); bx=fig.add_subplot(gs[0,1])
for a in (ax,bx):
    a.set_facecolor(SUR)
    for sp in ('top','right'): a.spines[sp].set_visible(False)
    for sp in ('left','bottom'): a.spines[sp].set_color(GRID)
    a.set_axisbelow(True); a.tick_params(colors=INK2,labelsize=9.5,length=0)
# --- left: specification curve
y=np.arange(len(S))[::-1]
ax.axvline(0,color=INK2,lw=0.9)
ax.grid(axis='x',color=GRID,lw=0.8)
for yy,(_,r) in zip(y,S.iterrows()):
    c=C if r['tag']=="dose in sd (headline)" else CG
    ax.plot([r['b']-1.96*r['s'],r['b']+1.96*r['s']],[yy,yy],color=c,lw=1.6,
            solid_capstyle='round',alpha=0.55)
    ax.plot([r['b']],[yy],'o',color=c,ms=7,mec=SUR,mew=1.6,zorder=3)
    ax.annotate(f"{r['b']:+.3f}  (t={r['b']/r['s']:+.2f})",(r['b'],yy),xytext=(0,11),
                textcoords='offset points',ha='center',color=INK if c==C else INK2,
                fontsize=8.6)
ax.set_yticks(y); ax.set_yticklabels([NICE[t] for t in S['tag']],color=INK2,fontsize=9.5)
ax.set_ylim(-0.7,len(S)-0.3)
ax.set_xlabel("dose x Post, log points per sd",color=INK2,fontsize=9.5)
ax.set_title("Making the dose tail-insensitive",color=INK,fontsize=11.5,loc='left',pad=26)
ax.text(0,1.035,"same specification throughout: unit + sector x year + country x year FE",
        transform=ax.transAxes,color=INK2,fontsize=9,va='bottom')
# --- right: decile step function
D=D.dropna(subset=['b']); xs=np.arange(1,11)
bx.axhline(0,color=INK2,lw=0.9); bx.grid(axis='y',color=GRID,lw=0.8)
dd=D.set_index(D['tag'].str[1:].astype(int)).reindex(xs)
bx.fill_between(xs,dd['b']-1.96*dd['s'],dd['b']+1.96*dd['s'],color=C,alpha=0.15,lw=0,
                step='mid')
bx.step(xs,dd['b'],where='mid',color=C,lw=1.8,zorder=3)
bx.plot(xs,dd['b'],'o',color=C,ms=5,mec=SUR,mew=1.4,zorder=4)
bx.set_xticks(xs); bx.set_xlabel("decile of 2015-17 targeting among policy users",
                                 color=INK2,fontsize=9.5)
bx.set_ylabel("post-2018 gap vs zero-policy units\n(log points)",color=INK2,fontsize=9.5,
              linespacing=1.6)
bx.set_title("The dose-response, decile by decile",color=INK,fontsize=11.5,loc='left',pad=26)
bx.text(0,1.035,"zero-policy country-sectors omitted; no upward slope",
        transform=bx.transAxes,color=INK2,fontsize=9,va='bottom')
fig.text(0.010,0.965,"The continuous specification is clean on pre-trends and rests on "
         "twenty observations",ha='left',va='top',color=INK,fontsize=14.5)
fig.text(0.010,0.905,
 "Targeting, continuous and pre-determined at its 2015-17 mean, on the §5b balanced panel "
 "of 5,166 country-sectors. The joint test of the ten pre-2018 event-study coefficients\n"
 "does not reject (p = 0.73 under this FE structure), so the design is clean where every "
 "earlier one was not. But the dose is extreme: among users the median is 0.0001 and the\n"
 "maximum 0.0769, and dropping the top 1% of users -- about twenty country-sectors -- takes "
 "the estimate from +0.029 to +0.001.",
 ha='left',va='top',color=INK2,fontsize=8.9,linespacing=1.7)
fig.text(0.010,0.075,
 "Bars are 95% confidence intervals, standard errors clustered on country. This is the §6 "
 "problem in the treatment dimension rather than the outcome dimension: there, 4% of "
 "country-sectors held 83% of\ntrade value; here, a comparable handful holds the policy "
 "variation the estimate is built on.",
 ha='left',va='top',color=INK2,fontsize=8.6,linespacing=1.7)
fig.savefig("sc20_tail.png",dpi=200,facecolor=SUR)
print("saved sc20_tail.png")
