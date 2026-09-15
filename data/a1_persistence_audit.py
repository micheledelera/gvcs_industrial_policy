"""Interrogating the persistence treatment. Two worries never checked.

WORRY 1 -- DURATION, NOT PRIORITY. §7 itself flagged that AD/CVD duties carry statutory
five-year lives, so "targeted in 6+ of 9 years" may record ONE long-duration instrument
rather than repeated political prioritisation. A1 chose 6+ anyway without checking. If the
persistent units average about one intervention per active year, and their active years form
a single contiguous run, persistence is measuring instrument duration.

WORRY 2 -- SIZE. Big sectors attract attention and are likelier to have some measure in any
year, so persistence may proxy for size -- and §A0 already showed treated units sit 2.6-2.8
log points above controls.

Also: how much does persistence overlap the share_frac_policies definition that §5k voided?
"""
import pandas as pd, numpy as np, gc
exec(open("a1_treated.py").read().split('print(f"sample:')[0])
nev=(pre_yrs==0); clean=nev&(post_yrs==0); tr=(pre_yrs>=6)
sz=XUS[BASE].mean(axis=1).values
raw2=pd.read_pickle("agg_for_estimation.pkl")
raw2['i']=raw2['i'].astype('int32'); raw2['t']=raw2['t'].astype('int32')
pl=raw2[['i','ISIC4c','t','n_policies','share_frac_policies']].drop_duplicates(
    subset=['i','ISIC4c','t'])
for c in ['n_policies','share_frac_policies']:
    pl[c]=pd.to_numeric(pl[c],errors='coerce').fillna(0)
del raw2; gc.collect()
NP=pl.pivot_table(index=['i','ISIC4c'],columns='t',values='n_policies',
                  aggfunc='sum').reindex(idx).fillna(0)
SF=(pl[pl['t'].isin(BASE)].groupby(['i','ISIC4c'],observed=True)['share_frac_policies']
    .mean().reindex(idx).fillna(0).values)
P9=list(range(2009,2018))
A=(NP[P9].values>0)
cnt=NP[P9].values
print(f"treated (6+ of 9): {int(tr.sum()):,}\n")

print(f"{'='*100}\nWORRY 1a: interventions per ACTIVE year -- one/yr suggests duration "
      f"recording, many suggests real activity\n{'='*100}")
per=np.divide(cnt.sum(axis=1), np.maximum(A.sum(axis=1),1))
v=per[tr]
print(f"  treated units: mean {v.mean():.2f}  p25 {np.percentile(v,25):.2f}  "
      f"median {np.median(v):.2f}  p75 {np.percentile(v,75):.2f}  p95 {np.percentile(v,95):.2f}")
print(f"  share of treated units averaging <= 1.5 interventions per active year: "
      f"{100*np.mean(v<=1.5):.0f}%")
print(f"  share averaging <= 1.0: {100*np.mean(v<=1.0):.0f}%")

print(f"\n{'='*100}\nWORRY 1b: are the active years ONE CONTIGUOUS RUN (duration) or "
      f"scattered (repeated attention)?\n{'='*100}")
def longest_run(b):
    m=c=0
    for x in b:
        c = c+1 if x else 0
        m = max(m,c)
    return m
lr=np.array([longest_run(A[i]) for i in range(len(A))])
ratio=lr/np.maximum(A.sum(axis=1),1)
print(f"  treated: mean longest run {lr[tr].mean():.2f} of mean {A[tr].sum(axis=1).mean():.2f} "
      f"active years  -> ratio {ratio[tr].mean():.2f}")
print(f"  share of treated whose active years are a SINGLE unbroken run: "
      f"{100*np.mean(ratio[tr]==1.0):.0f}%")
print(f"  share with ratio >= 0.9: {100*np.mean(ratio[tr]>=0.9):.0f}%")

print(f"\n{'='*100}\nWORRY 1c: sector composition of the treated group -- is it trade-remedy "
      f"heavy?\n{'='*100}")
vc=pd.Series([str(s) for s in SE[tr]]).value_counts()
print(f"  125 sectors, treated units spread over {len(vc)}.  top 12:")
print("   "+"  ".join(f"{k}:{v}" for k,v in vc.head(12).items()))
REM={'2410','2431','2432','2011','2013','2029','2394','2395','2599','2010'}
inrem=np.array([str(s) in REM for s in SE])
print(f"  treated in remedy-prone sectors (basic metals, basic chemicals, cement, "
      f"fabricated metal): {int((tr&inrem).sum())} of {int(tr.sum())} "
      f"({100*(tr&inrem).sum()/tr.sum():.0f}%)")
print(f"  same share among clean controls: {100*(clean&inrem).sum()/clean.sum():.0f}%")

print(f"\n{'='*100}\nWORRY 2: is persistence just SIZE?\n{'='*100}")
ln=np.log(sz)
print(f"  corr(pre-2018 years targeted, log US imports) over all units: "
      f"{np.corrcoef(pre_yrs,ln)[0,1]:+.3f}")
for lo,hi,lab in [(0,20,'smallest quintile'),(20,40,'2nd'),(40,60,'3rd'),
                  (60,80,'4th'),(80,100,'largest quintile')]:
    a,b=np.percentile(ln,[lo,hi]); m=(ln>=a)&(ln<=b)
    print(f"    {lab:18s} mean years targeted {pre_yrs[m].mean():4.2f}   "
          f"share treated {100*tr[m].mean():4.0f}%")

print(f"\n{'='*100}\nOVERLAP with the share_frac_policies definition §5k voided\n{'='*100}")
pos=SF[SF>0]; q75=np.quantile(pos,.75)
hi_sf=(SF>=q75)
print(f"  top-quartile share_frac units: {int(hi_sf.sum()):,};  persistence-treated: "
      f"{int(tr.sum()):,};  in BOTH: {int((tr&hi_sf).sum()):,}")
print(f"  -> {100*(tr&hi_sf).sum()/tr.sum():.0f}% of persistence-treated units are also "
      f"top-quartile on the voided measure")
print(f"  corr(years targeted, share_frac_policies) = {np.corrcoef(pre_yrs,SF)[0,1]:+.3f}")
