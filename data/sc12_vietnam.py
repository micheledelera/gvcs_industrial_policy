"""§5d. Vietnam case study — the design synthetic control was actually built for.

§5c found Vietnam has the highest share of high-targeting sectors beating their synthetic
counterparts (30% at ADH 2x, mean +1.06 log points). With ~500 treated units the aggregate
is uninformative; Abadie's method is designed for a handful of aggregate cases examined one
at a time, with the paths plotted and the weights shown. That is what this does.

For each of Vietnam's high-targeting sectors (top quartile of share_frac_policies among
positives), donor pool by the §5b rule: LOW policy, DIFFERENT country, Chinese share of the
US market within 10pp, log size within 2.0, then the 20 nearest on MVA/GDP, log MVA per
capita, ECI and the sector's export share. Canonical Abadie weights on the eleven lagged
log US exports plus those four characteristics. Placebo p from 495 low-policy units.

Outputs: per-sector table (weights, fit, gap, p), and a grid figure of treated vs synthetic
paths 2007-2024.
"""
import pandas as pd, numpy as np, matplotlib, time
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
exec(open("sc10_estimate.py").read().split("rows=[]\nfor meas,lab in")[0])
VN=704
SNAME={2630:'Communication equipment',2610:'Electronic components',
 2620:'Computers & peripherals',2410:'Basic iron & steel',2710:'Electric motors',
 2220:'Plastics products',2599:'Other metal products',2790:'Other electrical equip.',
 1410:'Wearing apparel',1520:'Footwear',3100:'Furniture',2732:'Electronic wire',
 2740:'Lighting equipment',2640:'Consumer electronics',1701:'Pulp & paper',
 2011:'Basic chemicals',2100:'Pharmaceuticals',2211:'Rubber tyres',2593:'Cutlery & tools',
 2393:'Ceramics',2431:'Iron casting',2660:'Irradiation & electro-medical',
 3030:'Air & spacecraft',3211:'Jewellery',3250:'Medical instruments',1020:'Fish processing',
 1030:'Fruit & vegetables',1512:'Luggage & handbags',1430:'Knitted apparel',2720:'Batteries',
 2670:'Optical instruments',2680:'Magnetic media',2651:'Measuring instruments',
 2652:'Watches & clocks',2825:'Food machinery',2823:'Metallurgy machinery'}
def sn(k): return SNAME.get(int(k), f"ISIC {int(k)}")

v=M['tgt'].values; pos=v[v>0]
q75,q25=np.quantile(pos,.75),np.quantile(pos,.25)
hi=np.where(v>=q75)[0]; lo=np.where(v<=q25)[0]
pl=rng.choice(lo,size=min(NPLAC,len(lo)),replace=False)
P=[]
for a in pl:
    r=one(a,lo[lo!=a])
    if r and np.isfinite(r['ratio']) and np.isfinite(r['tau_aug']): P.append(r)
ppre=np.array([x['pre'] for x in P]); ptau=np.abs([x['tau_aug'] for x in P])
log(f"{len(P)} placebos, median pre-RMSPE {np.median(ppre):.3f}")

def build(a):
    m=(np.abs(Dv[lo]-Dv[a])<=DECB)&(np.abs(Sv[lo]-Sv[a])<=SZB)&(CT[lo]!=CT[a])
    cand=lo[m]
    if len(cand)<8: return None
    d=np.sqrt(((Z[cand]-Z[a])**2).sum(axis=1))
    pool=cand[np.argsort(d)[:K]]
    A=np.hstack([Lp[pool],Z[pool]]); b=np.concatenate([Lp[a],Z[a]])
    w=sc(A,b)
    path_t=np.concatenate([Lp[a],Lq[a]])
    path_s=np.concatenate([Lp[pool].T@w, Lq[pool].T@w])
    pre=float(np.sqrt(((Lp[a]-Lp[pool].T@w)**2).mean()))
    yq=Lq[pool].mean(axis=1); Xp=Lp[pool].T
    Xc=Xp-Xp.mean(axis=1,keepdims=True)
    lam=plam(Xc,yq-yq.mean()); eta=reta(Xc,yq-yq.mean(),lam)
    ts=float(Lq[a].mean()-yq@w); ta=ts-float((Lp[a]-Xp@w)@eta)
    m2=ppre<=2.0*pre
    p2=(1+(ptau[m2]>=abs(ta)).sum())/(1+int(m2.sum())) if m2.sum()>=5 else np.nan
    return dict(pool=pool,w=w,path_t=path_t,path_s=path_s,pre=pre,tau=ts,tau_aug=ta,
                p_aug=p2,n_adm=int(m2.sum()),
                gap_pre=float((Lp[a]-Lp[pool].T@w).mean()))

vn=[a for a in hi if CT[a]==VN]
res={}
for a in sorted(vn,key=lambda z:-Sv[z]):
    r=build(a)
    if r: res[a]=r
log(f"Vietnam high-targeting sectors with a usable pool: {len(res)} of {len(vn)}")

print(f"\n{'='*118}\nVIETNAM, high-targeting sectors: treated vs synthetic\n{'='*118}")
print(f"{'sector':<32s}{'China%':>8s}{'IPshare':>9s}{'pre-RMSPE':>11s}{'gap pre':>9s}"
       f"{'tau':>8s}{'tau_aug':>9s}{'p':>7s}{'top donors (weight)':>40s}")
rows=[]
for a,r in res.items():
    top=np.argsort(-r['w'])[:3]
    ds=", ".join(f"{nm(CT[r['pool'][j]])}-{int(SE[r['pool'][j]])} {r['w'][j]:.2f}"
                 for j in top if r['w'][j]>0.02)
    st='*' if (np.isfinite(r['p_aug']) and r['p_aug']<0.10) else ' '
    print(f"{sn(SE[a]):<32s}{Dv[a]:>8.0f}{M['tgt'].values[a]*100:>9.3f}{r['pre']:>11.3f}"
          f"{r['gap_pre']:>+9.3f}{r['tau']:>+8.3f}{r['tau_aug']:>+9.3f}"
          f"{r['p_aug']:>7.3f}{st}  {ds}")
    rows.append(dict(sector=int(SE[a]),name=sn(SE[a]),Dec=Dv[a],ip=M['tgt'].values[a],
                     pre=r['pre'],gap_pre=r['gap_pre'],tau=r['tau'],tau_aug=r['tau_aug'],
                     p_aug=r['p_aug'],n_adm=r['n_adm'],donors=ds))
R=pd.DataFrame(rows); R.to_csv("sc12_vietnam.csv",index=False)
ok=R['p_aug'].notna()
print(f"\n  {len(R)} sectors   mean tau_aug {R['tau_aug'].mean():+.3f}   "
      f"median pre-RMSPE {R['pre'].median():.3f}   "
      f"sectors with p<0.10: {int((R['p_aug']<0.10).sum())}/{int(ok.sum())}")

# ---------------- figure ----------------
SUR="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e3e2df"
C_T="#2a78d6"; C_S="#eb6834"
order=sorted(res.keys(), key=lambda z:-Sv[z])
n=len(order); nc=4; nr=int(np.ceil(n/nc))
fig,axes=plt.subplots(nr,nc,figsize=(4.0*nc,3.1*nr),facecolor=SUR,squeeze=False)
for ax in axes.flat: ax.set_visible(False)
for q,a in enumerate(order):
    r=res[a]; ax=axes[q//nc][q%nc]; ax.set_visible(True); ax.set_facecolor(SUR)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    for s in ('left','bottom'): ax.spines[s].set_color(GRID)
    ax.grid(axis='y',color=GRID,lw=0.7); ax.set_axisbelow(True)
    ax.axvline(2017.5,color=INK2,lw=0.9,ls=(0,(4,3)))
    ax.plot(YRS,r['path_t'],color=C_T,lw=2.0,solid_capstyle='round',zorder=3)
    ax.plot(YRS,r['path_s'],color=C_S,lw=2.0,solid_capstyle='round',zorder=3)
    pv='' if not np.isfinite(r['p_aug']) else f"  p={r['p_aug']:.2f}"
    ax.set_title(f"{sn(SE[a])}",color=INK,fontsize=11,loc='left',pad=11)
    ax.text(0,1.005,f"China {Dv[a]:.0f}% of US mkt   tau {r['tau_aug']:+.2f}{pv}",
            transform=ax.transAxes,color=INK2,fontsize=8.5,va='bottom')
    ax.set_xlim(2006.4,2024.6); ax.set_xticks([2008,2012,2016,2020,2024])
    ax.tick_params(colors=INK2,labelsize=9,length=0)
axes[0][0].legend(handles=[plt.Line2D([],[],color=C_T,lw=2,label='Vietnam'),
                           plt.Line2D([],[],color=C_S,lw=2,label='Synthetic control')],
                  frameon=False,fontsize=9,loc='upper left',labelcolor=INK,handlelength=1.5)
fig.suptitle("Vietnam's policy-targeted sectors and their synthetic controls",
             x=0.008,y=0.995,ha='left',color=INK,fontsize=15)
fig.text(0.008,0.008,
 "log US exports. Donors: low-policy sectors in OTHER developing countries with Chinese US market share within 10pp, "
 "log size within 2.0,\nthen the 20 nearest on MVA/GDP, MVA per capita, ECI and export share. "
 "Weights match 2007-2017. p from 495 low-policy placebos (ADH 2x fit filter).",
 color=INK2,fontsize=8.5,linespacing=1.6)
fig.tight_layout(rect=[0.006,0.045,0.997,0.965])
fig.savefig("vietnam_sc.png",dpi=190,facecolor=SUR)
print("\nsaved vietnam_sc.png")
log("DONE")
