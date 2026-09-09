"""ROUTE A, properly specified. Canonical Abadie with a TRIMMED donor pool.

§4d failed because 129 donors fitting 11 pre-treatment observations gives an exact fit, so
Abadie's post/pre RMSPE ratio divides by zero. Abadie's own applications run donor-to-
pre-period ratios of 2-5 (California 38/19; Texas 45/8 with covariates), not 12.

Fix, following Abadie's practice of restricting the donor pool to plausibly comparable
units:

  1. For each treated country, rank donors by Euclidean distance on PRE-DETERMINED
     covariates only -- log total exports (size), MVA/GDP, log MVA per capita, ECI --
     standardised. Distance uses NO outcome information, so donor selection is separate
     from the outcome path; the SC then matches lagged outcomes within that pool.
  2. Keep the K nearest. Sweep K in {10,15,20,25,30,129} so the degeneracy is visible as
     a function of the donor-to-period ratio rather than assumed.
  3. Match on the eleven lagged outcomes PLUS the four covariates (Abadie's X_1, X_0),
     covariates weighted by COV_W.
  4. In-space placebo: each of the 129 donors is treated as pseudo-treated and given its
     OWN K nearest donors, so placebo and treated units are constructed identically.
     p = rank of the treated country's post/pre RMSPE ratio among the 129.

outcome  log US orientation = log(share of US imports) - log(share of RoW imports)
pre 2007-2017   post 2018-2024
"""
import pandas as pd, numpy as np, gc, time
t0=time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
USA, CHINA = 842, 156
PRE=list(range(2007,2018)); POST=list(range(2018,2025)); YRS=PRE+POST
COV_W=1.0
C2ISO={699:356,757:756,579:578,251:250,842:840,381:380,490:158,729:729}
XLSX=("/root/.claude/uploads/618110ef-d894-58be-8e00-e7f2bf548b03/"
      "4ad6510b-0154963EF4289D02DEA410D8D38E4E6A8C6D.xlsx")
NAME={484:'Mexico',699:'India',704:'Vietnam',458:'Malaysia',764:'Thailand',76:'Brazil',
 360:'Indonesia',608:'Philippines',792:'Turkiye',50:'Bangladesh',616:'Poland',348:'Hungary',
 643:'Russia',152:'Chile',710:'South Africa',604:'Peru',586:'Pakistan',32:'Argentina',
 170:'Colombia',784:'UAE',642:'Romania',218:'Ecuador',682:'Saudi Arabia',818:'Egypt',
 504:'Morocco',68:'Bolivia',634:'Qatar',804:'Ukraine',100:'Bulgaria',191:'Croatia',
 404:'Kenya',788:'Tunisia',566:'Nigeria',688:'Serbia',12:'Algeria',508:'Mozambique',
 288:'Ghana',524:'Nepal',368:'Iraq',214:'Dominican Rep',222:'El Salvador',188:'Costa Rica',
 320:'Guatemala',340:'Honduras',558:'Nicaragua',591:'Panama',858:'Uruguay',116:'Cambodia',
 144:'Sri Lanka',780:'Trinidad&Tob',328:'Guyana',388:'Jamaica',862:'Venezuela',600:'Paraguay'}
def nm(c): return NAME.get(int(c),str(int(c)))

raw=pd.read_pickle("agg_for_estimation.pkl")
raw['i']=raw['i'].astype('int32'); raw['j']=raw['j'].astype('int32'); raw['t']=raw['t'].astype('int32')
raw['n_policies']=pd.to_numeric(raw['n_policies'],errors='coerce').fillna(0)
adv=raw[['i','Advanced_i']].drop_duplicates().set_index('i')['Advanced_i']
pol=raw.groupby('i')['n_policies'].max()
us=raw[raw['j']==USA]
usi=us.groupby(['i','t'],observed=True)['imports'].sum().unstack('t')
alli=raw.groupby(['i','t'],observed=True)['imports'].sum().unstack('t')
usmkt=us.groupby('t',observed=True)['imports'].sum()
wldmkt=raw.groupby('t',observed=True)['imports'].sum()
pre_r=raw[raw['t'].isin([2015,2016,2017])]
Xk=(pre_r.groupby(['i','ISIC4c'],observed=True)['imports'].sum()/3).unstack(fill_value=0.0)
del raw,us,pre_r; gc.collect()
for D in (usi,alli): D.columns=[int(c) for c in D.columns]
usmkt.index=[int(c) for c in usmkt.index]; wldmkt.index=[int(c) for c in wldmkt.index]

sh=Xk.div(Xk.sum(axis=1).replace(0,np.nan),axis=0); world=Xk.sum(axis=0)/Xk.sum().sum()
M=((sh.div(world,axis=1))>=1).astype(float); M=M.loc[M.sum(axis=1)>0,M.sum(axis=0)>0]
kc,kp=M.sum(axis=1),M.sum(axis=0)
vals,vecs=np.linalg.eig(((M.div(kc,axis=0))@(M.div(kp,axis=1).T)).values)
K_=np.real(vecs[:,np.argsort(-np.real(vals))[1]])
ECI=pd.Series((K_-K_.mean())/K_.std(),index=M.index)
if ECI.corr(pd.Series(kc,index=M.index))<0: ECI=-ECI
u=pd.read_excel(XLSX,sheet_name='Data'); u=u[u['Year'].isin([2015,2016,2017])]
U=u.pivot_table(index='CountryCode',columns='VariableCode',values='Value',aggfunc='mean')

S_us=(usi.div(usmkt,axis=1))[YRS]; S_row=((alli-usi).div(wldmkt-usmkt,axis=1))[YRS]
keep=[i for i in S_us.index if adv.get(i,1)==0 and i!=CHINA]
S_us,S_row=S_us.loc[keep],S_row.loc[keep]
ok=((S_us>0).all(axis=1))&((S_row>0).all(axis=1))
S_us,S_row=S_us[ok],S_row[ok]
LU,LR=np.log(S_us),np.log(S_row); Y=LU-LR

C=pd.DataFrame(index=Y.index)
iso=[C2ISO.get(c,c) for c in Y.index]
C['lexp']=np.log(Xk.sum(axis=1).reindex(Y.index).replace(0,np.nan).values)
C['mva_gdp']=U['NV_IND_MANF'].reindex(iso).values
C['lmva_pc']=np.log(U['NV_IND_MANFPC'].reindex(iso).values)
C['eci']=ECI.reindex(Y.index).values
C=C.dropna()
Y=Y.loc[C.index]; LU=LU.loc[C.index]; LR=LR.loc[C.index]
Cz=(C-C.mean())/C.std()
T=pd.Series({i: bool(pol.get(i,0)>0) for i in Y.index})
TR=list(Y.index[T.values]); DN=list(Y.index[~T.values])
log(f"with complete covariates: treated {len(TR)}  donors {len(DN)}")

def proj_simplex(v):
    u=np.sort(v)[::-1]; c=np.cumsum(u)-1.0; r=np.arange(1,len(v)+1)
    cond=u-c/r>0
    if not cond.any(): return np.ones_like(v)/len(v)
    return np.maximum(v-c[cond][-1]/r[cond][-1],0.0)
def sc(A,b,iters=6000):
    n=A.shape[0]; w=np.ones(n)/n; L=np.linalg.norm(A,2)**2/A.shape[1]+1e-12
    for _ in range(iters): w=proj_simplex(w-(A@(A.T@w-b)*2.0/A.shape[1])/L)
    return w
def rms(b,f): return float(np.sqrt(((np.asarray(b)-np.asarray(f))**2).mean()))

def nearest(i, pool, K):
    d=np.sqrt(((Cz.loc[pool].values-Cz.loc[i].values)**2).sum(axis=1))
    return [pool[j] for j in np.argsort(d)[:K]]

def fit_one(i, pool):
    A=np.hstack([Y.loc[pool,PRE].values, COV_W*Cz.loc[pool].values])
    b=np.concatenate([Y.loc[i,PRE].values, COV_W*Cz.loc[i].values])
    w=sc(A,b)
    pre_=rms(Y.loc[i,PRE].values, Y.loc[pool,PRE].values.T@w)
    post_=rms(Y.loc[i,POST].values, Y.loc[pool,POST].values.T@w)
    return w, pre_, post_

print(f"\n{'='*104}\nDONOR-POOL SIZE SWEEP  (T_pre = {len(PRE)} years + 4 covariates = "
      f"{len(PRE)+4} matching targets)\n{'='*104}")
print(f"{'K donors':>10s}{'ratio K/T_pre':>15s}{'median pre-RMSPE':>19s}"
      f"{'zero-fit countries':>21s}{'median post/pre':>17s}{'median p':>11s}")
allres={}
for K in [10,15,20,25,30,len(DN)]:
    plac=[]
    for d in DN:
        pool=nearest(d,[x for x in DN if x!=d],K)
        _,p_,q_=fit_one(d,pool); plac.append(q_/max(p_,1e-8))
    plac=np.array(plac)
    rows=[]
    for i in TR:
        pool=nearest(i,DN,K)
        w,p_,q_=fit_one(i,pool)
        ratio=q_/max(p_,1e-8)
        gU=float((LU.loc[i,POST]-(LU.loc[pool,POST].values.T@w)).mean())
        gR=float((LR.loc[i,POST]-(LR.loc[pool,POST].values.T@w)).mean())
        gUp=float((LU.loc[i,PRE]-(LU.loc[pool,PRE].values.T@w)).mean())
        gRp=float((LR.loc[i,PRE]-(LR.loc[pool,PRE].values.T@w)).mean())
        rows.append({'K':K,'i':i,'name':nm(i),'pre':p_,'post':q_,'ratio':ratio,
                     'p':float((1+(plac>=ratio).sum())/(1+len(plac))),
                     'gap_pre':float((Y.loc[i,PRE]-(Y.loc[pool,PRE].values.T@w)).mean()),
                     'gap_post':float((Y.loc[i,POST]-(Y.loc[pool,POST].values.T@w)).mean()),
                     'g_US':gU,'g_RoW':gR,'g_US_pre':gUp,'g_RoW_pre':gRp,
                     'w_max':w.max(),'n_pos':int((w>0.01).sum())})
    R=pd.DataFrame(rows); allres[K]=R
    print(f"{K:>10d}{K/len(PRE):>15.1f}{R['pre'].median():>19.5f}"
          f"{int((R['pre']<1e-6).sum()):>21d}{R['ratio'].median():>17.2f}{R['p'].median():>11.3f}")
A=pd.concat(allres.values()); A.to_csv("sc05_trimmed.csv",index=False)
log("sweep done")
