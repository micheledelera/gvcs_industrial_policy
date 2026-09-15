"""§5b figure. Country-sector synthetic control, HIGH vs LOW policy, between countries.

Reuses the §5b machinery verbatim (sample, donor rule, trimming, solver) from
sc10_estimate.py and the §5f country-demeaning from sc14_demean.py, and keeps the full
2007-2024 path for every treated unit and every placebo instead of collapsing to tau.

Three columns:
  1  TARGETING (share_frac_policies), raw log US exports          -- the §5b headline
  2  VOLUME (n_policies), raw log US exports                      -- the §5b null
  3  TARGETING, outcome demeaned by the country's own cross-sector
     average each year                                            -- §5f

Row 1  mean treated path against mean synthetic path, both expressed relative to the
       TREATED group's own 2015-17 mean, so the pre-treatment gap stays visible rather
       than being normalised away. Bands are +/- 1 jackknife se over COUNTRIES.
Row 2  the gap, treated minus synthetic, unnormalised, against the 10th-90th percentile
       envelope of the ~495 in-space placebos (LOW-policy units given their own pool by
       the identical rule). This is the Abadie placebo picture: the treated gap has to
       leave the envelope to be evidence.
"""
import pandas as pd, numpy as np, matplotlib, time
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])

Lraw=np.hstack([Lp,Lq])
cm=pd.DataFrame(Lraw,index=pd.Index(CT,name='i')).groupby(level='i').transform('mean').values
Ldm=Lraw-cm
IPRE=[YRS.index(y) for y in PRE]

def path(a, lopool, Y):
    """weights fitted on Y's pre-period + the four covariates; returns both full paths"""
    m=(np.abs(Dv[lopool]-Dv[a])<=DECB)&(np.abs(Sv[lopool]-Sv[a])<=SZB)&(CT[lopool]!=CT[a])
    cand=lopool[m]
    if len(cand)<8: return None
    d=np.sqrt(((Z[cand]-Z[a])**2).sum(axis=1))
    pool=cand[np.argsort(d)[:K]]
    yp=Y[:,:len(PRE)]
    w=sc(np.hstack([yp[pool],Z[pool]]), np.concatenate([yp[a],Z[a]]))
    return Y[a], Y[pool].T@w

def run(meas, Y):
    v=M[meas].values; pos=v[v>0]
    hi=np.where(v>=np.quantile(pos,.75))[0]; lo=np.where(v<=np.quantile(pos,.25))[0]
    TR,SY,CC=[],[],[]
    for a in hi:
        r=path(a,lo,Y)
        if r is None: continue
        TR.append(r[0]); SY.append(r[1]); CC.append(CT[a])
    PG=[]
    for a in rng.choice(lo,size=min(NPLAC,len(lo)),replace=False):
        r=path(a,lo[lo!=a],Y)
        if r is not None: PG.append(r[0]-r[1])
    log(f"{meas}: {len(TR)} treated / {len(set(CC))} countries, {len(PG)} placebos")
    return np.array(TR),np.array(SY),np.array(CC),np.array(PG)

def jk(Mx, cl):
    """mean over units, se jackknifed over clusters (countries)"""
    mu=Mx.mean(axis=0); cs=np.unique(cl)
    a=np.array([Mx[cl!=c].mean(axis=0) for c in cs])
    return mu, np.sqrt((len(cs)-1)/len(cs)*((a-a.mean(axis=0))**2).sum(axis=0))

SPECS=[('tgt','raw',"Targeting: share of HS lines covered"),
       ('vol','raw',"Volume: number of interventions"),
       ('tgt','dm', "Targeting, country-demeaned outcome")]
YMAP={'raw':Lraw,'dm':Ldm}
import os, pickle
if os.path.exists("sc10_paths.pkl"):
    OUT=pickle.load(open("sc10_paths.pkl","rb")); log("loaded cached fits")
else:
    OUT=[run(m,YMAP[y]) for m,y,_ in SPECS]
    pickle.dump(OUT,open("sc10_paths.pkl","wb"))


S10=pd.read_csv("sc10_estimate.csv"); S14=pd.read_csv("sc14_demean.csv")
TAU=[S10[S10.meas=='tgt'], S10[S10.meas=='vol'], S14[S14.meas=='tgt']]

SUR="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e3e2df"
C_TR="#2a78d6"; C_SY="#eb6834"; C_PL="#c9c7c2"; C_PL2="#a8a6a1"
fig=plt.figure(figsize=(14.2,8.9),facecolor=SUR)
gs=fig.add_gridspec(2,3,left=0.062,right=0.972,top=0.800,bottom=0.150,
                    hspace=0.50,wspace=0.20,height_ratios=[1.30,1.0])
A0=[fig.add_subplot(gs[0,c]) for c in range(3)]
A1=[fig.add_subplot(gs[1,c]) for c in range(3)]
A0[1].sharey(A0[0]); A1[1].sharey(A1[0]); A1[2].sharey(A1[0])

for col,((meas,ykind,title),(TR,SY,CC,PG),tab) in enumerate(zip(SPECS,OUT,TAU)):
    a0,a1=A0[col],A1[col]
    for ax in (a0,a1):
        ax.set_facecolor(SUR)
        for sp in ('top','right'): ax.spines[sp].set_visible(False)
        for sp in ('left','bottom'): ax.spines[sp].set_color(GRID)
        ax.grid(axis='y',color=GRID,lw=0.8); ax.set_axisbelow(True)
        ax.axvspan(2015,2017,color="#000000",alpha=0.04,lw=0)
        ax.axvline(2017.5,color=INK2,lw=1.0,ls=(0,(4,3)))
        ax.set_xlim(2006.6,2026.4); ax.set_xticks([2007,2011,2015,2019,2024])
        ax.tick_params(colors=INK2,labelsize=9.5,length=0)

    # ---- row 1: the two paths, relative to the treated group's own 2015-17 mean
    base=TR[:,IPRE].mean(axis=1)[:,None]
    ends=[]
    for Mx,c,lab in [(TR-base,C_TR,"High-policy (treated)"),(SY-base,C_SY,"Synthetic control")]:
        mu,se=jk(Mx,CC)
        a0.fill_between(YRS,mu-se,mu+se,color=c,alpha=0.15,lw=0)
        a0.plot(YRS,mu,color=c,lw=2.0,solid_capstyle='round',zorder=3,label=lab)
        a0.plot([YRS[-1]],[mu[-1]],'o',color=c,ms=5.5,mec=SUR,mew=1.8,zorder=4)
        ends.append(mu[-1])
    dy=[0,0] if abs(ends[0]-ends[1])>0.07 else [7,-7]      # keep end labels legible
    for v,c,off in zip(ends,[C_TR,C_SY],dy):
        a0.annotate(f"{v:+.2f}",(YRS[-1],v),xytext=(7,off),textcoords='offset points',
                    color=c,fontsize=9.5,va='center',fontweight='medium')
    a0.set_title(title,color=INK,fontsize=11.5,loc='left',pad=20)
    a0.text(0,1.045,f"{len(TR)} country-sectors, {len(set(CC))} countries"
            f"   ·   median pre-fit RMSPE {tab['pre'].median():.2f}",
            transform=a0.transAxes,color=INK2,fontsize=9,va='bottom')

    # ---- row 2: the gap against the in-space placebo distribution
    G=TR-SY; mu,se=jk(G,CC)
    for q,c,al,lab in [(10,C_PL,0.60,f"placebo 10-90 pct  (n={len(PG)})"),
                       (25,C_PL2,0.55,"placebo 25-75 pct")]:
        a1.fill_between(YRS,np.percentile(PG,q,axis=0),np.percentile(PG,100-q,axis=0),
                        color=c,alpha=al,lw=0,label=lab)
    a1.axhline(0,color=INK2,lw=0.9)
    a1.fill_between(YRS,mu-se,mu+se,color=C_TR,alpha=0.20,lw=0)
    a1.plot(YRS,mu,color=C_TR,lw=2.2,zorder=3,label="treated gap")
    a1.set_ylim(-1.25,1.25)
    gp=mu[IPRE].mean(); gq=mu[len(PRE):].mean()
    a1.text(0,1.055,f"gap before 2018  {gp:+.3f}      after  {gq:+.3f}\n"
            f"$\\tau$ ridge-augmented {tab['tau_aug'].mean():+.3f}"
            f"    median placebo $p$  {tab['p_aug'].median():.2f}",
            transform=a1.transAxes,color=INK2,fontsize=9,va='bottom',linespacing=1.6)

A0[0].set_ylabel("log US exports,\nrelative to treated 2015-17 mean",
                 color=INK2,fontsize=9.5,linespacing=1.6)
A1[0].set_ylabel("treated minus synthetic\n(log points)",color=INK2,fontsize=9.5,linespacing=1.6)
A0[2].set_ylabel("own scale:\ndeviation from country mean",color=INK2,fontsize=9,linespacing=1.6)
A0[2].yaxis.set_label_position("right")
A0[0].legend(frameon=False,loc='upper left',fontsize=9.5,labelcolor=INK,
             handlelength=1.5,borderpad=0.1)
A1[0].legend(frameon=False,loc='lower left',fontsize=8.6,labelcolor=INK,
             handlelength=1.5,borderpad=0.1)
A0[0].annotate("treatment\nbegins 2018",(2017.5,A0[0].get_ylim()[0]),xytext=(-6,10),
               textcoords='offset points',ha='right',color=INK2,fontsize=8.6,linespacing=1.35)

fig.text(0.012,0.972,"High-policy country-sectors against their synthetic counterparts, "
         "2007-2024",ha='left',va='top',color=INK,fontsize=15)
fig.text(0.012,0.925,
 "Developing economies excluding China, 5,166 country-sectors with a complete positive US "
 "export series. Treated = top quartile of the positive 2015-17 policy distribution.\n"
 "Donors = bottom quartile, in a DIFFERENT country, with China's share of US imports within "
 "10pp and log export size within 2.0, trimmed to the 20 nearest on MVA/GDP,\nlog MVA per "
 "capita, ECI and the sector's share of the country's exports. Weights match the eleven "
 "pre-2018 outcomes plus those four characteristics (w>=0, sum to 1, no intercept).",
 ha='left',va='top',color=INK2,fontsize=8.9,linespacing=1.75)
fig.text(0.012,0.088,
 "Row 1 bands, and the row 2 band on the treated gap, are +/-1 standard error jackknifed "
 "over countries. Placebo bands are pointwise percentiles of the in-space placebos: "
 "low-policy units\ngiven their own donor pool by the identical rule. Row 2 is clipped at "
 "+/-1.25 (the 10-90 placebo envelope reaches about +/-2 by 2024); column 3 has its own "
 "vertical scale in row 1.",
 ha='left',va='top',color=INK2,fontsize=8.6,linespacing=1.75)
fig.savefig("sc10_paths.png",dpi=200,facecolor=SUR)
log("saved sc10_paths.png")
for (meas,yk,title),(TR,SY,CC,PG) in zip(SPECS,OUT):
    g=TR.mean(axis=0)-SY.mean(axis=0)
    inside=np.mean([(np.percentile(PG[:,j],10)<=g[j]<=np.percentile(PG[:,j],90))
                    for j in range(len(YRS))])
    print(f"  {title[:36]:36s} gap 2007 {g[0]:+.3f}  2015-17 {g[IPRE].mean():+.3f}  "
          f"2018 {g[len(PRE)]:+.3f}  2021 {g[YRS.index(2021)]:+.3f}  "
          f"2022-24 {g[-3:].mean():+.3f}   inside placebo 10-90 in {100*inside:.0f}% of years")
