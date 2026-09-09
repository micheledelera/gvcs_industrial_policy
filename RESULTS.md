# GVCs, US–China decoupling, and industrial policy — results log

Data: `trade_df_isic.dta` (Zenodo 10.5281/zenodo.22282107), BACI bilateral trade
aggregated to ISIC-4, 2007–2024, 229 countries, ~20M positive flows, merged with
GTA-derived industrial policy variables at exporter–sector–year level.

Estimation: PPML (`ppmlhdfe` / `pyfixest.fepois`). Code in `code/`.

---

## 1. Specification

$$
X_{ijst} = \exp\Big[
\alpha_{ist} + \alpha_{jst} + \alpha_{ij}
+ \beta_1 DDD^{dev}_{ijst} + \beta_2 DDD^{adv}_{ijst}
+ \gamma_1 IPxUS^{dev} + \gamma_2 IPxUS^{adv} + \gamma_3 IPxUS^{chn}
+ \delta_1 DecUS^{chn} + \delta_2 AdvDecUS
\Big]\varepsilon_{ijst}
$$

with

$$
DDD^{dev}_{ijst} = \underbrace{target_s \times \mathbb{1}[t \geq 2018]}_{\text{decoupling}}
\times \underbrace{\widetilde{share\_frac\_policies}_{i,s,t-3}}_{\text{policy targeting, lagged 3y, /SD}}
\times \underbrace{\mathbb{1}[j = US]}_{\text{destination}}
\times \underbrace{\mathbb{1}[i \in dev \setminus CHN]}_{\text{exporter group}}
$$

**Fixed effects.** `α_ist` (exporter×sector×year) and `α_jst` (importer×sector×year)
are the sectoral generalisation of the structural-gravity multilateral resistance
terms — they nest the standard `α_it`/`α_jt`. `α_ij` absorbs all time-invariant
bilateral trade costs (distance, language, colonial ties).

**Treatment variable.** `share_frac_policies` = the share of country *i*'s total
industrial policy directed at sector *s*. This is **policy targeting**, not policy
volume. Standardised, so coefficients read per 1 SD.

**Three exporter groups.** China / developing ex-China / advanced. China is kept in
the sample but given its own terms (`DecUS_chn`, `IPxUS_chn`), so it continues to
discipline `α_jst` and keep sector-year totals real without contaminating `β₁`.
Developing ex-China is the omitted category for the `Dec × US` level terms, so
`DecUS_chn` reads as "China vs developing ex-China" and `AdvDecUS` as
"advanced vs developing ex-China" (= RQ1).

**Clustering** at (i,s). Clustering at (i,s,t) is wrong here: every US-bound
observation is alone in its own `ist` cell (264,606 US obs in exactly 264,606
cells), so it provides no adjustment to the identifying variation and understates
SEs by roughly a factor of two.

**Interpretation.** `β₁` is the *differential* between decoupling and non-decoupling
sectors. The *level* effect inside decoupling sectors is `γ₁ + β₁`.

---

## 2. Main results — full sample (229 destinations)

| | **lag 3** (headline) | | **lag 0** | |
|---|---:|---:|---:|---:|
| | Estimate | p | Estimate | p |
| **`DDD_dev`** | **+0.0524** (0.0168) | **0.0018** | **+0.0373** (0.0161) | **0.0204** |
| `DDD_adv` | +0.0089 (0.0127) | 0.484 | −0.0006 (0.0095) | 0.951 |
| `IPxUS_dev` | −0.0376 (0.0186) | 0.043 | −0.0328 (0.0160) | 0.040 |
| `IPxUS_adv` | −0.0038 (0.0109) | 0.724 | +0.0038 (0.0113) | 0.733 |
| `IPxUS_chn` | −0.1563 (0.0624) | 0.012 | −0.1035 (0.0602) | 0.085 |
| `Dec_US_chn` | −0.3004 (0.0914) | 0.001 | −0.3571 (0.0867) | <0.001 |
| `Adv_Dec_US` | −0.1260 (0.0700) | 0.072 | −0.1129 (0.0628) | 0.072 |
| N | 16,886,001 | | 19,542,630 | |

SEs in parentheses, clustered (i,s).

**Level effect in decoupling sectors** (γ₁ + β₁), lag 3: −0.0376 + 0.0524 = **+0.0148**.
So the *differential* is 5.2pp while the *level* is mildly positive (~+1.5%).

---

## 3. Decomposition — what the result depends on

Each row changes exactly ONE thing from the winning spec. Restricted sample
(22 advanced destinations), so compare to +0.0335 (p=0.016), not the full-sample
+0.0524.

| Change | `DDD_dev` | p | |
|---|---:|---:|---|
| *(baseline: share + target×post + lag 3)* | +0.0335 | 0.016 | |
| **(a)** level measure (`frac_policies`) instead of share | **−0.0081** | 0.556 | **collapses** |
| **(b)** continuous `Decouple_intensity` instead of target×post | **+0.1008** | 0.027 | survives |
| **(c)** contemporaneous instead of lag 3 | +0.0131 | 0.390 | weakens |

**(a) is the key discriminating test.** Policy *targeting* carries the result;
policy *volume* gives nothing. **(b)** two independently constructed decoupling
measures agree (magnitudes reconcile via scaling: intensity is 0–1 with nonzero
mean 0.20, so 0.101 × 0.20 ≈ +2.0% vs +3.4% for the binary).

---

## 3b. Build-up ladder

Restricted sample (22 destinations), `share_frac_policies`, cluster (i,s),
FE `α_ist + α_jst + α_ij` fixed. China is in the sample with its own terms
(`Dec_US_chn`, `IPxUS_chn`) at **every** step, so movement across steps reflects
only the dev/adv structure.

| Step | lag 3 `DDD_dev` | p | lag 0 `DDD_dev` | p |
|---|---:|---:|---:|---:|
| 1. DDD pooled (dev+adv together) | +0.0049 | 0.382 | +0.0017 | 0.696 |
| 2. + dev/adv split | +0.0208 | 0.085 | +0.0073 | 0.589 |
| 3. + `IPxUS` | **+0.0397** | **0.010** | +0.0185 | 0.259 |
| 4. + `Adv_Dec_US` (full) | **+0.0335** | **0.016** | +0.0131 | 0.390 |

`DDD_adv` is a stable zero throughout (lag 3: +0.0025 → −0.0015 → −0.0002).

**Reading.** The parsimonious spec shows nothing. The dev/adv split quadruples the
coefficient (advanced economies dilute rather than cancel, since `DDD_adv` ≈ 0);
adding `IPxUS` nearly doubles it again and carries it across significance, because
the general IP→US channel is negative and biases the triple toward zero when
omitted. This settles the earlier question of whether `IPxUS` belongs in the
specification: whether or not an economic channel can be named for it, omitting it
is not innocuous.

**Caveat to state explicitly in the paper.** The effect appears only with the full
control set, which is the pattern specification searching produces. The defence is
that each addition was motivated *before* estimation — the dev/adv split is the
hypothesis itself, `IPxUS` is required by the triple-difference saturation
literature (Olden & Møen 2022), and `Adv_Dec_US` was proposed on structural
grounds — and the ordering is verifiable in the commit history. Present the
build-up table with the reasoning for each step, rather than the full spec as if
it arrived fully formed.

---

## 3c. Does the result survive without `IPxUS`?

Asked because a headline that depends on one control invites scrutiny.
Answer: **no specification gives a positive, significant `DDD_dev` with `IPxUS` omitted.**

| | with `IPxUS` | without `IPxUS` |
|---|---:|---:|
| restricted, lag 3 | +0.0335 (p=0.016) | +0.0208 (p=0.085) |
| **full, lag 3** | **+0.0524 (p=0.002)** | **+0.0150 (p=0.242)** |

**But the two specs estimate different quantities, and the arithmetic confirms it.**
In the full spec `IPxUS_dev` = −0.0376 and `DDD_dev` = +0.0524, so the *level*
effect in decoupling sectors is γ₁ + β₁ = **+0.0148**. The no-`IPxUS` estimate is
**+0.0150** — a match to within 0.0002. With no `IPxUS_dev` in the model, `DDD_dev`
is the only regressor carrying `IP × US × dev` variation, so it cannot measure a
differential against non-decoupling US flows; it collapses onto the level. The two
results are internally consistent rather than contradictory.

So:
- **level** of the IP→US relationship in decoupling sectors: +1.5%, not significant
- **differential** between decoupling and non-decoupling sectors: +5.2%, p=0.002

The paper's claim is about the differential, which is the right estimand for
"targeting pays off *more* where decoupling is happening". But the headline does
depend on a control whose economic channel is not cleanly articulated, and that is
a legitimate line of referee attack. Either name the channel, or state the estimand
explicitly as a differential and report both numbers.

Side note: `Adv_Dec_US` reaches significance in this spec (−0.1376, p=0.047, vs
p=0.072 with `IPxUS`), so RQ1 firms up.

---

## 3d. Robustness to how China is handled

Option A absorbs China's US flows with a fixed effect — one level per (sector, year)
among China→US observations — instead of the two parametric terms. Non-parametric,
and reports nothing mechanical.

| | parametric (`Dec_US_chn` + `IPxUS_chn`) | China-FE |
|---|---:|---:|
| lag 3 | +0.0335 (p=0.016) | **+0.0386 (p=0.005)** |
| lag 0 | +0.0131 (p=0.390) | +0.0167 (p=0.274) |

The FE version runs slightly larger at both lags and reaches the same conclusions,
including the lag contrast. The headline does not depend on how China is controlled.

---

## 3e. Geography — does the FE structure control for Vietnam and Mexico?

Vietnam borders China and Mexico borders the US, so both are natural friendshoring
winners for reasons that have nothing to do with industrial policy. `α_ij` absorbs
the *level* of each bilateral relationship, but it is time-invariant, so it cannot
absorb any *change* in the return to geography — and decoupling is precisely a shock
to that return. `α_ist` catches a proximity advantage that lifts a country's sales
everywhere, but not one that is US-specific; `α_jst` is common across exporters. A
time-varying, US-specific, country-specific advantage therefore sits in the residual.

Restricted sample (22 advanced destinations), lag 3, cluster (i,s).

| | FE | what it absorbs | `DDD_dev` |
|---|---|---|---:|
| A | `α_ist + α_jst + α_ij` | baseline | +0.0335 (p=0.016) |
| B | `α_ist + α_jst + α_ij×post` | any post-2018 shift in the bilateral relationship | +0.0337 (p=0.013) |
| C | `α_ist + α_jst + α_ijt` | **any** year-specific bilateral shock | **+0.0358 (p=0.016)** |
| D | baseline, dropping Vietnam + Mexico | the two named cases | +0.0216 (p=0.200) |

`DDD_dev` stays identified under B and C because it varies across *sectors* within
each (i, j, t) cell, while those FE are constant within it.

**C is the answer to the geography question.** It absorbs everything specific to a
country-partner-year — proximity to China, proximity to the US, USMCA, tariff-line
reallocation, any bilateral agreement or nearshoring wave — without functional form.
What identifies the coefficient afterwards is only this: within US–Vietnam in a given
year, sectors Vietnam had subsidised three years earlier grew faster than sectors it
had not. Geography is a country-year fact and cannot generate that pattern, because
it moves every sector in the bundle together. The coefficient does not fall.

**D is a different test and it does not pass.** Dropping the two countries costs 36%
of the point estimate and all of the significance. This is *not* evidence for the
geography story — if geography were driving the result, C would have killed it.
What D measures is concentration, and the leverage calculation says why:

| | share of treated rows | share of PPML-weighted treatment mass |
|---|---:|---:|
| Mexico | 1.3% | 26.3% |
| Vietnam | 2.1% | 8.9% |
| **both** | **3.4%** | **35.1%** |

PPML weights each observation by its fitted mean, so large suppliers dominate the
score regardless of row counts. Dropping Vietnam and Mexico removes a third of the
identifying variation, and +0.0216 sits comfortably inside the confidence interval of
the baseline (the gap of 0.0119 is well under D's own SE of 0.0168). So D does not
*contradict* A — it cannot confirm it with a third of the mass gone.

The honest reading: **the result is robust to confounding by geography (C) but is not
broad-based (D).** It rests on a thin identifying base.

### How thin

`DDD_dev` is nonzero only where all of {j = USA, `target_s` = 1, t ≥ 2018,
developing ex-China, IP > 0} hold:

- **8,858 treated observations** out of 3.58m
- 2,454 treated (i, s) clusters, of 26,290
- **38 treated exporters**, 99 treated sectors
- top 5 countries (Mexico, India, Vietnam, South Africa, Bangladesh) = 64% of mass

2,454 clusters is ample for CRV1 at (i,s). But if the true correlation of the errors
is at the *exporter* level, the relevant count is 38, which is at the boundary where
CRV1 becomes liberal. This is now the most important open robustness check, ahead of
anything else on the list.

---

## 3f. Exporter-level clustering

§3e showed only 38 exporters ever carry a nonzero `DDD_dev`, so the (i,s) clustering
used throughout may assume away correlation that matters. Variants A and C refit with
CRV1 on the exporter alone — same sample, lag, regressors and fixed effects, so any
movement is attributable to the clustering level. 229 exporter clusters, vs 26,290 at
(i,s): comfortably above any small-cluster threshold, so CRV1 is on solid ground.

| | coefficient | se, cluster (i,s) | se, cluster (i) | p, (i,s) | p, (i) |
|---|---:|---:|---:|---:|---:|
| A. baseline `α_ij` | +0.0335 | 0.0138 | **0.0127** | 0.016 | 0.008 |
| C. pair-year `α_ijt` | +0.0358 | 0.0149 | **0.0129** | 0.016 | 0.006 |

**The standard errors fall.** This is not a mistake and not mechanical in the other
direction: coarser clustering permits within-cluster correlation of *either sign*, and
here the within-country, cross-sector score contributions are negatively correlated.
That is what reallocation looks like — when a country's US-bound exports tilt toward
electronics they tilt away from textiles, so sector-level terms partly cancel when
aggregated to the country. Every off-diagonal block that (i,s) clustering set to zero
is on average negative, so summing them shrinks the variance.

The pattern runs through the whole column and is sharpest where reallocation is
strongest, on the China terms (variant A):

| | se, (i,s) | se, (i) |
|---|---:|---:|
| `IPxUS_chn` | 0.0408 | 0.0227 |
| `Dec_US_chn` | 0.0868 | 0.0579 |

### What to report

**Keep (i,s) as the headline.** The conservative standard error is the larger one, and
here that is (i,s). Switching to exporter clustering because it improves the p-value
would be indefensible, and a referee who sees a *smaller* se under coarser clustering
will assume cherry-picking unless the larger one leads.

The result this delivers is narrower than hoped but was the live threat: **the finding
is not an artefact of assuming independence across sectors within a country.** Relaxing
that assumption moves the point estimate not at all (+0.0335 and +0.0358 are unchanged
to four decimals) and tightens rather than widens the interval.

---

## 3g. Leave-one-exporter-out

§3e left the concentration question open: Mexico and India are 43% of the
PPML-weighted treatment mass, so is `DDD_dev` a broad regularity or a few country
stories? Each of the top 10 exporters by treatment mass is dropped in turn from the
headline specification (`α_ist + α_jst + α_ij`, lag 3, cluster (i,s)). IP is
standardised once on the full restricted sample and that SD reused for every row, and
only rows are subset, so nothing but the sample changes across fits.

| dropped | mass share | N | treated obs | `DDD_dev` | se | p |
|---|---:|---:|---:|---:|---:|---:|
| **none (baseline)** | — | 3,543,453 | 8,858 | **+0.0335** | 0.0138 | 0.016 |
| Mexico | 26.3% | 3,507,524 | 8,746 | +0.0244 | 0.0177 | 0.168 |
| India | 16.6% | 3,503,758 | 8,203 | +0.0350 | 0.0144 | 0.015 |
| Vietnam | 8.9% | 3,507,250 | 8,669 | +0.0330 | 0.0135 | 0.014 |
| South Africa | 6.4% | 3,506,127 | 8,705 | +0.0315 | 0.0134 | 0.019 |
| Bangladesh | 5.9% | 3,521,656 | 8,794 | +0.0361 | 0.0139 | 0.009 |
| Indonesia | 4.3% | 3,507,217 | 8,248 | +0.0366 | 0.0139 | 0.008 |
| Brazil | 4.3% | 3,505,869 | 8,379 | +0.0384 | 0.0139 | 0.006 |
| UAE | 3.8% | 3,508,391 | 8,764 | +0.0291 | 0.0142 | 0.041 |
| Thailand | 3.7% | 3,504,303 | 8,776 | +0.0365 | 0.0146 | 0.012 |
| Malaysia | 3.7% | 3,507,029 | 8,746 | +0.0320 | 0.0134 | 0.017 |

Range across drops **[+0.0244, +0.0384]**, median **+0.0335** — identical to the
baseline. **Nine of ten drops stay significant at 5%.**

**The result is not a single-country artefact.** That was the live worry after §3e,
and the sweep answers it: no drop moves the point estimate by as much as one standard
error, and the estimate is bounded in a narrow band either side of the baseline.

**Mexico is the one influential case, and it is influential for precision more than
for the point estimate.** Dropping it moves the coefficient by 0.0091 — about half a
standard error, nowhere near a significant change — but *also* inflates the standard
error by 28% (0.0138 → 0.0177), and it is the combination that costs significance
rather than either alone. This is what variant D in §3e was picking up: D dropped
Mexico and Vietnam together and landed at +0.0216, but the sweep shows Vietnam
contributes essentially nothing to that (dropping Vietnam alone gives +0.0330,
p=0.014). D was a Mexico result mislabelled as a Mexico-and-Vietnam result.

Two further readings worth recording:

- **The Vietnam concerns are empirically moot.** Both the geography worry (Vietnam
  borders China, so it is a natural relocation destination) and the measurement worry
  (GTA plausibly understates Vietnamese industrial policy, so coding it low-IP is
  doubtful) bear on a country whose removal leaves the coefficient at +0.0330. Neither
  problem is propagating into the headline. Both remain live for interpreting country
  rankings; neither is generating `DDD_dev`.
- **Mexico deserves a sentence in the paper, not a robustness scare.** USMCA concluded
  in 2018 and nearshoring is a distinct Mexican story, so a referee will ask. The
  answer is §3e variant C: `α_ijt` absorbs any US–Mexico year shock, whatever its
  source, and the coefficient there is +0.0358 (p=0.016). Mexico matters for how
  precisely the effect is estimated, not for whether it survives controlling for
  Mexico-specific shocks.

---

## 3h. Pre-trends in the gravity specification

The earlier event study was a US-only DiD with a different FE structure. The headline
is a gravity result, so it needs its own test. Replace `1[t≥2018]` with year
interactions:

```
ES_τ = target_s × 1[t=τ] × IP_{i,s,t-3} × US_j × dev_i     (τ ≠ 2017)
G_τ  =            1[t=τ] × IP_{i,s,t-3} × US_j × dev_i     (all τ)
```

`ES_τ` is the year-by-year analogue of `DDD_dev`, referenced to 2017.
`target_s × 1[t=τ] × US_j` needs no term — `α_jst` absorbs it.

### The first attempt failed on scaling

Standardising IP by a single pooled SD does not work year by year. With IP lagged
three years, coverage collapses at the start of the sample:

| year | nonzero IP cells | % of cells | within-year SD ÷ pooled SD |
|---|---:|---:|---:|
| 2010 | 0 | 0.0% | 0.00 |
| 2011 | 219 | 2.3% | 0.11 |
| 2012 | 1,204 | 12.1% | 0.66 |
| 2013–2024 | 1,218–1,878 | 10.7–17.5% | 0.49–1.60 |

2010 is empty, which is why `ES_2010` and `G_2010` dropped for collinearity. 2011 has
an SD one-tenth the pooled value, so `ES_2011` came back at **+2.10** — two orders of
magnitude above the pooled DDD. That is a scaling artefact, not a pre-trend. From 2012
coverage is stable, but the within-year SD still ranges over a factor of 3.3.

This affects only the year-by-year design. In the main specification the pooled SD is
a scaling constant on a single coefficient, and `α_ist` absorbs IP's level and trend.

### Take 2: restrict to 2012+, and put IP on a comparable scale two ways

**B, IP standardised within year** — still unstable. Coefficients range 0.013 to 0.64
against a pooled anchor of +0.0373, with standard errors spanning a 12× range
(0.033–0.39), and two pre-period coefficients individually significant.

**C, binary `1[IP>0]`** — well behaved. Standard errors 0.17–0.26, a 1.5× range. So
the instability is **scale** in the skewed continuous measure (86% of cells are zero,
so its SD is driven by a thin positive tail), not collinearity between `G_τ` and
`ES_τ`. Binary and continuous have the *same* 8,858 treated observations and differ
only in how those observations are weighted.

| | coef | se | p | | coef | se | p |
|---|---:|---:|---:|---|---:|---:|---:|
| pre 2012 | +0.1599 | 0.221 | 0.468 | POST 2018 | +0.1365 | 0.172 | 0.427 |
| pre 2013 | +0.1486 | 0.253 | 0.556 | POST 2019 | −0.2347 | 0.234 | 0.317 |
| pre 2014 | +0.1215 | 0.229 | 0.596 | POST 2020 | +0.2859 | 0.200 | 0.154 |
| pre 2015 | +0.0327 | 0.242 | 0.892 | POST 2021 | −0.0043 | 0.217 | 0.984 |
| pre 2016 | +0.0301 | 0.262 | 0.909 | POST 2022 | +0.3272 | 0.174 | 0.061 |
| | | | | POST 2023 | +0.1050 | 0.197 | 0.594 |
| | | | | POST 2024 | +0.0105 | 0.179 | 0.953 |

**Pre-period: flat.** Nothing significant, max |t| = 0.73, and the path declines
monotonically toward the 2017 reference (+0.160 → +0.030), which is the right shape.
Joint Wald χ²(5) = 0.60, **p = 0.988**.

**That p-value is the caveat, not the reassurance.** Under the null the expected χ² is
5, and 0.60 is far below it. With standard errors around 0.22 the test would fail to
reject a pre-trend of +0.4 as readily as one of zero. Report it as *the pre-period is
flat and the test has little power*, not as a clean pass.

### The flat post period is a power artefact, not a null

The post-period coefficients are also insignificant, with no visible step at 2018
(pre mean +0.099, post mean +0.089). Two readings: low power from splitting the
treatment across 12 year coefficients, or a genuine null in which the finding depends
on policy *intensity* and a simple incidence indicator carries no signal.

The pooled binary DDD separates them, on the same sample and specification with only
the treatment variable changed:

| treatment | `DDD_dev` | se | p |
|---|---:|---:|---:|
| continuous, within-year SD | +0.0373 | — | 0.005 |
| **binary `1[IP>0]`** | **+0.1456** | 0.0699 | **0.037** |

**Significant — so it is low power.** The pooled binary effect of +0.1456 sits inside
the range of the year-by-year coefficients, and against year standard errors of ~0.2
implies a per-year t of about 0.73, exactly the maximum observed. Detecting +0.1456 at
5% would need a standard error of 0.074, so roughly seven times the annual data.

Two things follow that matter beyond the pre-trend question:

1. **The result does not depend on the GTA count being a meaningful cardinal measure
   of intensity.** A binary "any recorded policy" indicator delivers the same
   qualitative finding. This directly limits the measurement-validity concern: whether
   GTA understates a country's policy effort — the Vietnam problem — bears on the
   ranking of intensities, and the headline does not rest on that ranking.
2. **The sign pattern replicates.** `IPxUS_dev` = −0.1523 (p=0.007), so the level in
   decoupling cells is γ₁+β₁ = −0.007, essentially zero, with the positive coefficient
   again a *difference* from a negative baseline rather than a net advantage. Same
   catch-up reading as §3c, on a completely different treatment definition.

---

## 3i. Extensive margin, and the headline with zeros restored

The delivered data holds only positive flows, so a country that stops selling a
sector to the US simply vanishes from the sample. That makes entry and exit
unobservable and — more seriously — means every estimate above is on a sample
**selected on the outcome**, which is exactly the selection PPML with zeros exists to
avoid.

**Frame.** Keep (i, s, t) cells where exporter i ships sector s to *someone* (any of
the 229 destinations), then rectangularise those over the 22 advanced destinations.
Filling every (i, j, s, t) cell instead would manufacture structural zeros — a country
that never makes a product is not "failing to serve the US". A zero here means
*produces it, does not sell it in this market*, which is the object of interest.

| | |
|---|---:|
| active (i,s,t) cells | 451,339 |
| rectangularised over 22 destinations | 9,929,458 |
| after the lag-3 IP merge | 7,914,060 (54.7% zeros) |
| after dropping all-zero (i,s,t) cells | 7,168,238 (50.0% zeros) |
| observed positives, all matched | 4,233,360 |

Zeros are not rare in this frame. The intuition that ISIC 4-digit is too aggregated to
have zeros holds for the *observed* data, not for the destination dimension: a country
can make a product and simply not sell it in Germany.

### Descriptively, US participation peaked before the trade war

| year | developing (ex-China) active cells | serving US | % |
|---|---:|---:|---:|
| 2007 | 18,050 | 8,387 | 46.5% |
| 2012 | 20,123 | 10,130 | 50.3% |
| 2017 | 20,725 | 11,257 | **54.3%** |
| 2021 | 20,762 | 11,104 | 53.5% |
| 2024 | 20,406 | 10,599 | 51.9% |

| transition | entry | exit |
|---|---:|---:|
| 2015→2017 | 10.0% | 6.8% |
| 2017→2021 | 8.9% | 9.6% |
| 2017→2024 | 8.7% | 11.7% |

Net entry was **+3.2pp in the two years before the cutoff and −3.0pp over the seven
years after**. At the aggregate level, decoupling did not pull developing exporters
into the US market. (2024 may be incomplete in BACI; do not lean on the last year.)

### E2. Extensive margin: a tight null

LPM on 1[imports>0], same specification and clustering, N = 7,914,060.

| | coef | se | p |
|---|---:|---:|---:|
| **DDD_dev** | **+0.00023** | 0.00162 | **0.887** |
| DDD_adv | −0.00035 | 0.00110 | 0.754 |
| IPxUS_dev | −0.00343 | 0.00107 | 0.001 |
| IPxUS_adv | −0.00210 | 0.00081 | 0.010 |

This is a **precisely estimated zero, not an underpowered one**: the 95% interval is
[−0.29, +0.34] percentage points against a base participation rate of 45.3%. Even the
top of that interval is a 0.7% relative change, against an intensive-margin effect of
roughly 3.4% in flow value. The extensive margin can account for at most a small
fraction of the total, and most likely none of it.

`IPxUS_dev` = −0.0034 (p=0.001) reproduces on a binary outcome the same negative
baseline tilt that γ₁ shows on the intensive margin — high-IP developing exporters are
*less* likely to serve the US to begin with.

### E1. The headline with zeros restored

| | N | `DDD_dev` | se | p |
|---|---:|---:|---:|---:|
| positives only, tol 1e-8 (headline) | 3,543,453 | +0.0335 | 0.0138 | 0.016 |
| positives only, tol 1e-6 (control) | 3,582,183 | +0.0335 | 0.0138 | 0.016 |
| **zeros restored, tol 1e-6** | 7,092,582 | **+0.0271** | **0.0108** | **0.012** |

E1 would not converge at the default 1e-8 tolerance — restoring zeros makes the
weighted least squares inside PPML's IRLS badly conditioned, since zero observations
get tiny IRLS weights and many groups are near-separated — so it ran at 1e-6. The
middle row exists to stop that confounding the comparison: refitting positives-only at
1e-6 reproduces the headline **to four decimals**, so the solver tolerance changes
nothing and the whole +0.0335 → +0.0271 gap is attributable to the zeros.

**The headline survives, with better precision.** The point estimate falls 19% — well
inside one standard error — while the standard error tightens 22% and significance
improves. That is exactly what E2 predicts: with no extensive-margin effect, adding
zeros mainly adds information about the intensive margin and mildly dilutes the
estimate. E1 and E2 corroborate each other rather than each standing alone.

### What this settles, and what it does not

**Settles:** the result is not an artefact of selecting on positive flows, which was a
genuine vulnerability given that handling zeros is the main reason to use PPML at all.
And the story is now specific: industrial policy helps developing exporters **sell more
of what they already ship** to the US in decoupling-targeted sectors, not break into
US market segments they were not already in.

**Does not settle:** at ISIC 4-digit, "entry" means beginning to export an *entire
industry* to the US — rare and lumpy. The benchmark (IMF WP 2024/041) finds its
emerging-market extensive-margin effects at product level, where entry means adding a
product line. Aggregation is plausibly the binding constraint here rather than the
economics, so the null is "no extensive margin at this level of aggregation", and a
BACI HS6 rebuild is the test that could overturn it.

---

## 3j. All policy measures — is it targeting or volume?

The first question any referee will ask is why policy is defined in terms of
*targeting* (the `share_*` family: what fraction of a country's policy effort goes to
this sector) rather than *volume* (the `n_`/`frac_` family: how much policy the sector
gets). Every measure, run on the **full sample** (229 destinations, 16,917,915 rows),
lag 3, cluster (i,s), headline FE — only the policy variable changes. Each is
standardised by its own SD, so coefficients are per-SD comparable.

| measure | family | n treated | with `IPxUS` | without | γ₁ |
|---|---|---:|---|---|---:|
| `n_policies` | volume | 8,858 | +0.0782 (0.0467) * p=.094 | +0.0307 p=.373 | −0.0591 |
| `frac_policies` | volume | 8,858 | +0.0140 (0.0185) p=.447 | −0.0053 p=.768 | −0.0213 |
| **`share_n_policies`** | **targeting** | 8,858 | **+0.0636 (0.0212) \*\*\* p=.003** | +0.0059 p=.721 | −0.0589 |
| **`share_frac_policies`** | **targeting** | 8,858 | **+0.0524 (0.0168) \*\*\* p=.002** | +0.0150 p=.242 | −0.0376 |
| `n_sub` | volume (sub) | 847 | +0.0005 p=.980 | −0.0033 p=.847 | −0.0038 |
| `frac_sub` | volume (sub) | 847 | −0.0172 p=.500 | +0.0013 p=.951 | +0.0187 |
| `share_n_sub` | targeting (sub) | 847 | −0.0066 p=.845 | −0.0220 p=.028 ** | −0.0155 |
| `share_frac_sub` | targeting (sub) | 847 | −0.0258 * p=.097 | −0.0206 p=.129 | +0.0053 |
| `asinh_n_sub` | volume (sub) | 847 | +0.0160 p=.652 | +0.0060 p=.774 | −0.0105 |
| `asinh_frac_sub` | volume (sub) | 847 | +0.0126 p=.707 | +0.0055 p=.777 | −0.0073 |

`share_frac_policies` reproduces the known full-sample headline (+0.0524, p=0.0018)
exactly, and its without-`IPxUS` counterpart reproduces §3c (+0.0150, p=0.242) — two
internal validations that the sample construction, lag merge and 1e-6 tolerance all
match earlier runs.

### The result, stated at the strength the evidence supports

**Both targeting measures are significant at 1%; neither volume measure reaches 5%.**

But this should **not** be written as "targeting matters and volume does not", because
`n_policies` has a *larger* point estimate (+0.0782) than `share_n_policies` (+0.0636).
What separates them is precision: SE 0.0467 against 0.0212. The two are built from
identical GTA interventions and differ only in normalisation, so the honest statement
is:

> Volume and targeting measures are constructed from the same interventions and differ
> only in normalisation. The targeting measures recover the effect with roughly half
> the standard error, while the volume measures cannot reject zero. This is what one
> expects if raw counts carry country-level scale that the fixed effects cannot absorb,
> while shares isolate the within-country prioritisation that §4's FE ladder showed is
> the only place the level measures' association survives.

`frac_policies` vs `share_frac_policies` (+0.0140 vs +0.0524, a factor of 3.7) is the
cleanest single pair to show, since there the point estimates genuinely diverge rather
than only the precision.

Note also that the FE ladder in §4 is not independent confirmation of this: it already
established that the level measures' raw association is entirely between-country. The
two findings are the same fact seen from different angles.

### Subsidies: a power statement, not a finding

All six subsidy cells are null-to-negative on **847 treated observations**, against
8,858 for the policy counts — the 0.93% coverage documented in `diag_measures.py`. The
`asinh` variants matter here: the raw `n_sub` and `frac_sub` are monetary amounts with
SD/mean above 40 and maxima in the millions, so a per-SD coefficient on them is set by
a handful of enormous values, and a null would be uninterpretable. Transformed
(SD 98,143 → 2.18 for `n_sub`), they are still null. So **the subsidy nulls are lack
of power, not a scaling artefact** — and they say nothing about whether subsidies work.

### `share_n_sub` without `IPxUS`: a warning, not a result

The single significant entry in the without-column, −0.0220 (p=0.028), is a
collinearity artefact:

- its SE *falls* 3.36× when `IPxUS` is dropped, against ~1.3× for well-populated measures
- the estimate reproduces the implied level γ₁+β₁ = −0.0221 to four decimals

With 847 treated observations and a subsidy share that barely moves within
country-sector, `dec × IP × US × dev` and `IP × US × dev` are nearly the same column:
the two coefficients are separately unidentified but their sum is pinned down tightly.
Dropping one does not add information — it reports the sum with a small standard error
and labels it the treatment effect.

**This is an argument for including `IPxUS` that is independent of Olden & Møen
(2022).** Beyond the saturation requirement, omitting it lets a near-collinear
treatment manufacture significance with the wrong sign.

### γ₁ replicates everywhere

`IPxUS_dev` is negative in all four well-populated measures (−0.0591, −0.0213, −0.0589,
−0.0376). The negative baseline tilt — developing IP users being *less* US-oriented to
begin with, so that `DDD_dev` measures catch-up rather than net advantage — is not an
artefact of the headline measure. It is a property of the data visible in every
adequately-powered policy variable.

---

## 3k. Lags and the continuous decoupling measure

§3j covered lag 3 with the binary `target × post` treatment. This completes the grid:
three measures (the volume benchmark plus the two targeting measures) × lag {0, 3} ×
treatment {binary, continuous} × {with, without `IPxUS`}. Full sample, cluster (i,s),
headline FE, tol 1e-6.

**On the continuous measure.** From `build_decouple_intensity.py`:

```
baseline_s   = mean(China's US mkt_share in 2015, 2016, 2017)     # sector scalar
trailing_st  = 3-year rolling mean of that share, ending at t
frac_lost_st = clip((baseline_s − trailing_st) / baseline_s, min=0)
Decouple_intensity_st = frac_lost_st × target_s
```

It is the fractional loss of China's US market share against its 2015–17 baseline,
both ends smoothed over three years — the smoothing was the answer to the concern that
a year-on-year decoupling variable would capture noise rather than trend.

**It is only interpretable as decoupling for t ≥ 2018.** China's mean share of US
imports rose from 0.189 (2007) to 0.219 (2015) before falling to 0.149 (2024). In the
early years the trailing average therefore sits *below* the 2015–17 baseline — not
because China was losing ground but because it had not yet finished gaining it — so
`baseline − trailing > 0`, the clip does not bite, and the formula reports a positive
"fractional loss". Hence pre-period values *larger* than the early post-period (mean
0.181 in 2007 against 0.021 in 2018), and an identical zero at 2017, where the trailing
window and the baseline window coincide. Used raw, the largest treatment values in the
sample would sit in the pre-period. It therefore enters as
`Decouple_intensity_st × post`.

Two further properties. It is **clipped at zero**, so a sector where China gained share
is indistinguishable from one that never decoupled. And it is **multiplied by
`target_s`**, so it adds no treated sectors — it re-weights the same ones by how far
they actually moved. Left in natural units; mean intensity among treated is 0.242, so
multiply by ~0.24 to compare with the binary column.

**A note for the paper.** Both `target` and `Decouple_intensity_st` are functions of
China's US market share, and in the US column China losing share means non-China
exporters gain by construction. `α_jst` absorbs the sector-year total, so `DDD_dev`
asks which non-China exporters capture more of the available reallocation — which is
interpretable. But the treatment intensity *is* the size of the pot being
redistributed, and that should be stated rather than left for a referee. It is also
the precise reason `Dec_US_chn` is mechanical rather than informative.

### `DDD_dev`, full sample

| | **target** l0 with | l0 without | l3 with | l3 without | **continuous** l0 with | l0 without | l3 with | l3 without |
|---|---|---|---|---|---|---|---|---|
| `n_policies` | +0.1005** | +0.0573* | +0.0782* | +0.0307 | +0.1404 | +0.1324 | +0.1246 | +0.0914 |
| **`share_n_policies`** | +0.0826*** | +0.0357** | +0.0636*** | +0.0059 | +0.1949*** | +0.1276*** | **+0.2321\*\*\*** | +0.1096** |
| `share_frac_policies` | +0.0373** | +0.0048 | +0.0524*** | +0.0150 | +0.1189*** | +0.0626* | +0.1316*** | +0.0593* |

\*\*\* p<.01 \*\* p<.05 \* p<.10. Treated obs: binary 8,726 (l0) / 8,858 (l3);
continuous 6,719 (l0) / 6,701 (l3) — the ~2,000 target-sector-years with zero measured
decoupling drop out of treatment. `share_frac_policies` lag-0 binary reproduces the
known full-sample value (+0.0373, p=0.0204) exactly.

### 1. `share_n_policies` is significant in 8 of 8 cells

No other measure is. It survives both lags, both treatment definitions, and the
presence or absence of `IPxUS`. **This argues for promoting it over
`share_frac_policies` as the paper's primary measure** — the current headline is the
weaker of the two targeting variables on every dimension tested here.

### 2. The continuous treatment sharpens the targeting-vs-volume separation

`n_policies` is **null in all four continuous cells**. §3j could only establish that
the share measures were *more precise* — point estimates were similar, which was the
weak point of that argument. Under intensity weighting the volume measure has roughly
half the implied effect *and* twice the standard error (se 0.11–0.12 against 0.05–0.06
for `share_n_policies`). This is a better answer to "why define policy in terms of
targeting?" than the binary table alone supports.

### 3. The `IPxUS` requirement is less binding than §3c suggested

`share_n_policies` survives dropping `IPxUS` in three of four specifications
(p = .033, .006, .013). It fails only in the binary lag-3 cell — the §3c case — where
γ₁ nearly cancels β₁ and leaves the level γ₁+β₁ ≈ +0.005, too close to zero to detect.
So the §3c null is specific to one measure-lag-treatment combination and not a general
fragility of the design. The saturation argument for including `IPxUS` still holds
(Olden & Møen 2022, plus the `share_n_sub` collinearity warning in §3j), but the result
no longer *depends* on it.

### 4. Timing: no clean story, and none should be told

| measure | binary | continuous |
|---|---|---|
| `share_n_policies` | lag 0 > lag 3 | lag 3 > lag 0 |
| `share_frac_policies` | lag 3 > lag 0 | lag 3 ≈ lag 0 |
| `n_policies` | lag 0 > lag 3 | both null |

The direction flips with the treatment definition. The data do not support a claim
about how quickly industrial policy acts on trade flows, and the §3h pre-trend test is
too underpowered to arbitrate. Report both lags; do not build an argument on either.

This also weakens a defence used earlier: the three-year lag was justified partly as
protection against reverse causality. Since the contemporaneous specification is
*stronger* in several cells, that protection is not doing the work it was claimed to.

### 5. Dose-response

Scaled by mean treated intensity (0.242), `share_n_policies` implies average effects of
+0.047 (lag 0) and +0.056 (lag 3), against binary values of +0.083 and +0.064. The
continuous version consistently implies a **smaller average** — the binary assigns full
treatment to sectors that barely decoupled — while implying **larger effects where
decoupling was severe** (a sector at intensity 0.9 gets +0.209 at lag 3). That
gradient is a more credible pattern than a uniform treatment effect, and is the
strongest available evidence that the mechanism tracks actual reallocation rather than
sector designation.

---

## 4. FE ladder — where the raw association lives

US-bound only, developing ex-China, per 1 SD, lag 3, clustered by exporter.
`IP` = main effect; `IPxDec` = interaction with target×post.

| Measure | FE | `IP` | p | `IPxDec` | p |
|---|---|---:|---:|---:|---:|
| `n_policies` | L1 `st` | **+0.1225** | **0.006** | +0.0069 | 0.764 |
| | L2 `+i` | −0.0051 | 0.906 | +0.0231 | 0.324 |
| | L3 `+is` | +0.0002 | 0.947 | +0.0092 | 0.298 |
| | L4 `+is+it` | −0.0026 | 0.633 | +0.0006 | 0.928 |
| `frac_policies` | L1 `st` | **+0.0555** | **0.001** | +0.0019 | 0.885 |
| | L2 `+i` | +0.0068 | 0.429 | −0.0009 | 0.911 |
| | L3 `+is` | −0.0031 | 0.285 | +0.0108 | 0.066 |
| | L4 `+is+it` | −0.0053 | 0.182 | +0.0067 | 0.071 |
| `share_frac_policies` | L1 `st` | **+0.0394** | **0.000** | −0.0103 | 0.403 |
| | L2 `+i` | **−0.0551** | **0.015** | +0.0170 | 0.547 |
| | L3 `+is` | **−0.0150** | **0.001** | **+0.0257** | **0.000** |
| | L4 `+is+it` | **−0.0131** | **0.000** | **+0.0212** | **0.000** |

**Reading.** The level measures are significant at L1 and dead the moment a *country*
fixed effect is added — the raw descriptive association is entirely between-country
(which countries do industrial policy), not within (which sectors a country targets).
The share measure behaves differently: its main effect flips *negative* and stays
significant, while the decoupling interaction is positive and survives everything.

---

## 5. Descriptives

Share of **total US imports**, decoupling (`target`) sectors, 2017 → 2024:

| Group | 2017 | 2024 | Change |
|---|---:|---:|---:|
| China | 30.8% | 19.0% | **−11.8 pp** |
| Advanced (37) | 37.3% | 39.4% | +2.1 pp |
| Dev: high IP (18) | 22.6% | 27.2% | +4.6 pp |
| Dev: low IP (18) | 6.2% | 10.9% | +4.7 pp |
| Dev: no IP (149) | 3.2% | 3.5% | +0.4 pp |

Largest gainers 2017–24 and their pre-period IP: Vietnam +2.65pp (IP 0.034),
Mexico +2.16pp (0.078), Thailand +0.72pp (0.036), India +0.69pp (0.630).
Country-level rank correlation between pre-period IP and share change: 0.33.

---

## 6. Discussion

**The reallocation is large and real.** China lost 11.8pp of US import share in
decoupling sectors between 2017 and 2024; developing economies absorbed most of it.
`Dec_US_chn` is strongly negative in every specification.

**Policy targeting matters; policy volume does not.** This is the central finding and
it rests on the (a) decomposition plus the FE ladder. The level measures
(`n_policies`, `frac_policies`) show a large raw association that vanishes entirely
once you control for *which country* you are looking at — they are proxies for
country size and industrial capacity. The share measure, which is within-country
normalised by construction, isolates the allocation decision and survives.

**The effect is discriminating on four dimensions**, each of which could have gone
the other way:
- *US-specific* — survives `α_ist`, so it is not general export growth
- *decoupling-specific* — `IPxUS_dev` is negative, so it is not a broad US tilt
- *developing-specific* — `DDD_adv` is zero
- *lagged* — stronger at lag 3 than lag 0, the opposite of what reverse causality
  would produce

**Selection into treatment is visible and points the right way.** The share main
effect is significantly *negative* (−0.013 at L4): countries direct policy at sectors
where they are weak. That makes the positive interaction more interesting, not less —
these are not sectors that were already winning.

**But the raw descriptive is compositional.** High-IP and low-IP developing economies
gained *identically* (+4.6 vs +4.7pp), the largest winners (Vietnam, Mexico, Thailand)
are low-IP, and the IP>0 / IP=0 split is largely large-vs-small economies. The
regression finding is not "countries with more industrial policy won" — it is
"within a country, the sectors it prioritised did relatively better in the US market
in decoupling sectors, with a multi-year lag."

### Open issues

1. ~~**Pre-trend test in the gravity specification itself.**~~ Done in §3h. The
   pre-period is flat (joint Wald p=0.988) but the test has little power, so it is
   weak supporting evidence rather than a clean pass. A design with more annual
   identifying variation would be needed to do better.
2. ~~**Exporter-level clustering.**~~ Settled in §3f: the standard errors *fall*
   under exporter clustering, because within-country cross-sector residuals are
   negatively correlated. (i,s) stays the headline as the conservative choice.
5. ~~**Concentration.**~~ Settled in §3g: across leave-one-out drops of the top
   10 exporters the coefficient stays in [+0.0244, +0.0384] with a median equal
   to the baseline, and 9 of 10 remain significant. Mexico is the one influential
   case, and mainly through the standard error.
3. ~~**Extensive margin.**~~ Done in §3i on a rectangularised panel (50% zeros).
   The extensive margin is a tight null (+0.0002, 95% CI ±0.3pp on a 45.3% base),
   and the headline survives restoring zeros with better precision (+0.0271,
   p=0.012). Open only at finer aggregation: at ISIC4 "entry" means starting to
   export a whole industry, so a BACI HS6 rebuild could still overturn the null.
4. **GTA intervention types.** The benchmark finds tax breaks strongly positive and
   direct transfers negative — they cancel in any pooled measure. Splitting by
   intervention type would identify *which kinds* of targeting work. Not testable
   with the current variables (`n_sub` is nonzero in only 1.3% of developing
   ex-China US-bound observations).
5. **`frac_sub` / `share_*` definitions** were reconstructed from the raw file
   mid-session; worth confirming they mean what we assume.

---

## §3l. Country-level capability controls in the between-country designs

**Why.** The sector fixed effect α_k holds *exposure* to Chinese withdrawal fixed —
every country in the regression faces the same vacated share in sector k. It does
nothing about two other threats: selection into treatment (governments target sectors
where they already hold RCA — Juhász et al. 2022) and differential capacity to respond
(two equally exposed, equally targeted countries absorb a vacated order book at
different speeds). "Mexico and Vietnam both supply product X" answers the first threat
and neither of the other two.

What the design already does about them: differencing (Δs, and α_ik in the panel)
removes any *time-invariant* (i,k) component, which is capability in that sector — so
the RCA half of the selection problem is handled, because the RCA sits in the level and
the level is differenced away. Δs^pre conditions on the pre-2018 trajectory. Neither
touches country-level heterogeneity in the *response* to a common shock. Prior evidence
that this matters: in the country-level DiD (§ earlier), `HasPolicy` collapsed from
+0.314 (p=.048) to +0.041 (p=.86) once ECI, log MVA per capita and size entered.

**Controls**, all 2015–17 means so none is a function of the outcome, all standardised:
log MVA per capita (UNIDO `NV_IND_MANFPC`), MVA/GDP (`NV_IND_MANF`), manufacturing
employment share (`SL_TLF_MANF`), log exports to world (BACI), ECI (BACI, 125 ISIC-4
sectors). Complete-control sample: 149–150 countries, **99.94% of pre-period US import
value** — the ~30 dropped countries are rounding.

**Four rungs.** (1) α_k baseline; (2) + capability levels; (3) + capability × Dec_k;
(4) + country FE α_i. Rung 4 is the direct answer to the comparability objection: under
α_i two countries are never compared, and the estimate rests only on whether a country
gained more share in the sectors it targeted than in those it did not. The price is that
the *volume* question dies with it — total policy effort is collinear with α_i — so only
*targeting* is identified there. α_i does **not** solve *within*-country selection.

Code: `data/fit_longdiff_ctrl.py` (full grid, 2020 base), `data/fit_ctrl_headline.py`
(the two specifications that carry signal). Results: `data/ctrl_headline.csv`,
`data/longdiff_ctrl_flow.csv`.

### Panel A — 2017-base long difference
Dep. var: pp change in country's share of US imports in sector k. WLS, weighted by
pre-period US imports, clustered by country. N = 12,117.

| measure | (1) α_k baseline | (2) + capability levels | (3) + capability × Dec | (4) + country FE |
|---|---|---|---|---|
| n_policies | −0.0637 (0.0440) | −0.2748 (0.1119)** | −0.1740 (0.0672)** | +0.0625 (0.0611) |
| frac_policies | +0.0275 (0.0196) | −0.0495 (0.0530) | +0.0127 (0.0307) | +0.0701 (0.0311)** |
| share_n_policies | −0.0004 (0.0960) | −0.1036 (0.1135) | −0.0677 (0.1043) | +0.1413 (0.1396) |
| **share_frac_policies** | **+0.2362 (0.0895)\*\*\*** | **+0.1685 (0.0875)\*** | **+0.1688 (0.0853)\*\*** | **+0.1427 (0.0940)** |

### Panel B — 2020-base difference, stock timing, with IP × Dec
Dec in raw pp (mean 24.9, sd 20.1), so multiply by ~20 to compare with the
standardised-Dec coefficients in `base2020_diff_results.csv`. N = 12,276.

| measure | (1) α_k | (2) + levels | (3) + × Dec | (4) + country FE |
|---|---|---|---|---|
| n_policies | +0.0021 (0.0024) | −0.0012 (0.0026) | −0.0091 (0.0053)* | +0.0010 (0.0043) |
| frac_policies | +0.0019 (0.0037) | −0.0001 (0.0040) | −0.0045 (0.0052) | +0.0016 (0.0038) |
| **share_n_policies** | **+0.0183 (0.0076)\*\*** | **+0.0172 (0.0073)\*\*** | **+0.0098 (0.0053)\*** | **+0.0063 (0.0034)\*** |
| share_frac_policies | +0.0084 (0.0037)** | +0.0066 (0.0035)* | +0.0038 (0.0032) | +0.0014 (0.0018) |

### Reading

The controls bite, which vindicates the objection rather than dismissing it.
`n_policies` in Panel A goes from insignificant to −0.27** once capability enters, and
the auxiliary coefficients say why: `log_exp` is +1.18 (se 0.51), so the raw policy
*count* was carrying country size.

The measures that survive are the two `share_*` ones — the ones already normalised by
the country's own total policy effort. That is what one would expect if the country-level
confound operates mainly through *volume*: dividing by the country total removes it
internally, which is why those measures move least when the controls arrive.

`share_frac_policies` in Panel A attenuates ~40% (0.236 → 0.143) and loses significance
only at rung 4, where the SE widens rather than the point estimate collapsing.
`share_n_policies` in Panel B survives all four rungs including α_i.

**Not yet done.** The panel event study carries α_ik, which absorbs these controls in
levels, so testing it there requires controls × year. Separate run.

---

## §3m. Capability controls in the panel event study, and a joint pre-trend test

The pair fixed effect a_ik absorbs every country control **in levels**, so the
comparability objection has to be answered here with something that varies over time.
Three rungs, ref year 2020, N = 218,106 (12,117 pairs, 149 countries, 99.94% of
pre-period US import value):

  (1) a_ik + a_kt                       baseline
  (2) + capability_i x year dummies     parametric; countries with a deeper industrial
                                        base get their own path through the sample
  (3) a_ik + a_kt + a_it                nonparametric; country-year FE absorb ALL
                                        country-level time-varying heterogeneity,
                                        observed or not. Two countries are never
                                        compared; identification is within a
                                        country-year across sectors, i.e. TARGETING only

Rung 3 is the panel analogue of the country FE in §3l and strictly stronger than rung 2:
(2) removes what we measured, (3) removes what we did not. IP_ik x 1[t=tau] still varies
across k within (i,t), so nothing is collinear.

Code: `data/fit_panel_es_ctrl.py`, `data/fit_panel_es_wald.py`.
Results: `data/panel_es_ctrl_2020.csv`, `data/panel_es_wald_2020.csv`.

### share_frac_policies, coefficients by year (IP x Dec / IP)

| year | (1) a_ik + a_kt | (2) + capability x year | (3) + a_it |
|---|---|---|---|
| 2017 | −0.008 / −0.114 | +0.042 / −0.021 | +0.160 / +0.083 |
| 2018 | −0.008 / −0.076 | +0.056 / +0.019 | +0.137 / +0.078 |
| 2019 | +0.095* / +0.027 | +0.155* / +0.085 | +0.221* / +0.133 |
| **2020** | — ref — | — ref — | — ref — |
| 2021 | +0.270* / +0.340* | +0.211 / +0.292* | +0.324 / +0.391* |
| 2022 | +0.450* / +0.571* | +0.379* / +0.507* | +0.590* / +0.712* |
| 2023 | +0.437* / +0.453* | +0.460* / +0.462* | +0.805* / +0.774* |
| 2024 | +0.332* / +0.247* | +0.332* / +0.229* | +0.494 / +0.372 |

The post-2020 rise survives both rungs. Under a_it it is *larger*.

### Joint pre-trend test — and a correction

The 2020-reference event study was previously reported as passing with "0/11
pre-period coefficients significant". **That was too weak a test**: counting years one
at a time ignores the covariance across them. Testing the 13 pre-2020 coefficients
jointly, using the finite-sample F form (13, 148) rather than the chi2 asymptotic:

| measure | spec | path | F(13,148) | p | slope/yr | t |
|---|---|---|---|---|---|---|
| share_frac_policies | (1) | IP x Dec | 7.21 | <.0001 | −0.0100 | −1.49 |
| share_frac_policies | (1) | IP | 24.46 | <.0001 | **+0.0189** | **3.93** |
| share_frac_policies | (2) | IP x Dec | 13.92 | <.0001 | **−0.0259** | **−5.15** |
| share_frac_policies | (2) | IP | 13.51 | <.0001 | +0.0005 | 0.11 |
| share_frac_policies | (3) | IP x Dec | 12.88 | <.0001 | −0.0261 | −4.14 |
| share_frac_policies | (3) | IP | 40.67 | <.0001 | +0.0319 | 5.42 |
| share_n_policies | (1) | IP x Dec | 10.61 | <.0001 | −0.0194 | −2.51 |
| share_n_policies | (3) | IP x Dec | 25.69 | <.0001 | −0.0895 | −17.13 |

**All 24 joint tests (4 measures x 3 rungs x 2 paths) reject at p < .0001.**

The slope column is the informative one. At the baseline the *interaction* path has no
linear drift (−0.0100, t = −1.49) while the *main* path does (+0.0189, t = 3.93).
Adding capability x year flips this exactly: main-path drift goes to zero (+0.0005,
t = 0.11) and the drift moves into the interaction (−0.0259, t = −5.15). Under a_it the
path is a shallow V — 2007 at +0.69, declining monotonically to +0.08 by 2018, rising
after 2020 — so 2020 sits near the bottom of a long decline and part of the post-2020
rise is measured against that trough.

**The capability controls redistribute the pre-trend between the two paths rather than
removing it.** The panel event study has no clean pre-period under any specification
tried so far. This was true before the controls; the controls only exposed that the
year-by-year check was inadequate.

This is why the long difference is the more defensible of the two designs: it does not
claim flat pre-trends, it conditions on delta-s^pre rather than assuming it away.

**Caveat.** CRV1 Wald over 13 restrictions is known to over-reject (Cameron-Miller).
A wild cluster bootstrap is the proper check. With F of 7-41 it is unlikely to flip,
but it has not been run.

### Open

1. Wild cluster bootstrap on the joint pre-trend tests above.
2. 2021 carries most of the panel's post-period movement. Section 301 tariff coverage
   by sector would help separate decoupling from post-COVID reconfiguration.

---

## §3n. Does differencing actually remove capability?

The §3l write-up claimed that working in differences removes time-invariant
country-sector effects "which is capability in that sector". That is right about a
narrow object and wrong as a defence of the design.

Write s_ikt = mu_ik + lambda_kt + u_ikt. Then delta-s_ik = (lambda_post - lambda_pre)
+ delta-u_ik and mu_ik is gone exactly. True -- but only if capability enters as an
**additive, time-invariant level shifter**. The plausible alternative is a loading:

    s_ikt = mu_ik + phi_ik * lambda_kt + u_ikt
    =>  delta-s_ik = phi_ik (lambda_post - lambda_pre) + delta-u_ik

Differencing kills mu_ik and leaves phi_ik **interacted with the shock** -- and the
shock is the object of study. Differencing sweeps out the level, where capability was
harmless, and isolates the part where it is not. Two further gaps it does not close:
forward-looking targeting (Cov(IP, delta-u) != 0 regardless of differencing), and the
fact that mu_ik additive *in shares* is not obviously "capability" at all, since
capability plausibly scales exports multiplicatively.

### Direct test

Regress the PRE-period change delta-s^pre_ik = s(2015-17) - s(2010-12) on the five
capability controls, within sector, weighted, clustered by country. If capability were
a pure level effect it should not predict the change. Code:
`data/diag_capability_trend.py`. N = 12,117, 149 countries, 125 sectors.

| | (A) capability | (B) + initial share | (C) + capability x Dec |
|---|---|---|---|
| log_mva_pc | −0.040 (0.392) | −0.068 (0.379) | +0.932 (0.491)* |
| mva_gdp | −0.249 (0.510) | +0.025 (0.460) | −1.138 (0.708) |
| mfg_emp | +0.370 (0.479) | +0.508 (0.316) | −0.132 (0.835) |
| log_exp | +1.043 (0.497)** | +0.688 (0.458) | +0.743 (0.507) |
| ECI | −0.129 (0.419) | −0.630 (0.363)* | +0.123 (0.601) |
| s0_pp | | **+0.0958 (0.0136)\*\*\*** | |
| **log_mva_pc x Dec** | | | **−5.206 (2.497)\*\*** |
| **mva_gdp x Dec** | | | **+3.619 (1.407)\*\*** |
| mfg_emp x Dec | | | +2.474 (2.048) |
| log_exp x Dec | | | +0.519 (2.419) |
| ECI x Dec | | | +0.260 (1.210) |
| **joint capability = 0** | F(5,148)=1.14, **p=0.34** | F=1.39, p=0.23 | F=2.26, **p=0.051** |

**The level version of the worry is not supported.** Capability alone does not predict
the pre-period trend within a sector (p = 0.34). Differencing plus alpha_k does leave
measured capability roughly orthogonal to the change.

**The loading version is supported and is the one that bites.** log_mva_pc x Dec is
−5.21** and mva_gdp x Dec is +3.62**, joint p = 0.051: in the sectors China dominated,
more-capable countries were already on different trajectories before anything happened.
Dec_k is the treatment intensity, so the contaminated object is the **IP x Dec
interaction** -- the term carrying the story -- not the IP main effect.

This makes rung 3 of §3l (capability x Dec) a necessary control rather than a
robustness flourish, and it is where share_frac_policies in Panel B loses significance
(+0.0084** -> +0.0038 ns). That attenuation now reads as informative rather than as an
overzealous specification.

**Sign note.** s0_pp is **+0.0958*** -- positive. Countries with a higher initial share
gained more over 2010-17. The process is DIVERGING, not mean-reverting, so s^pre in the
specification controls for persistence and the Galton regression-to-the-mean intuition
has the wrong sign here.

### Corrected wording for the note

> Selection on prior performance is handled in part by working in differences: a
> time-invariant country-sector effect drops out, and we verify that measured
> capability does not predict the pre-2018 change within a sector (F(5,148) = 1.14,
> p = 0.34). Differencing does not, however, remove capability that acts as a loading
> on sector-level shocks rather than as a level, and we find evidence that it does:
> capability interacted with China's pre-period share predicts differential pre-trends
> (joint p = 0.051). We therefore include capability x decoupling-intensity
> interactions throughout, and report a country fixed effect as the limiting case.

---

## §3o. Recommended long difference — with the decoupling interaction restored

The §3m/§3n recommendation omitted IP x Dec_k. That was an error. Two reasons it must
be in:

1. **Without it there is no decoupling result.** With only a_k, the coefficient says
   "countries that targeted sector k gained share in sector k" -- in every sector,
   vacated by China or not. That is a generic industrial-policy finding. The decoupling
   content is whether the payoff RISES with how much China withdrew, i.e. IP x Dec.
2. **The asymmetry is indefensible.** Keeping X_i x Dec_k while dropping IP_ik x Dec_k
   is the worst combination: policy-active countries are more capable, so the capability
   interactions absorb variation belonging to the policy interaction.

a_k absorbs Dec_k's main effect only; IP x Dec varies across i within k, so the sector
FE is no substitute for it.

    d_s_ik = b1 (IP_ik x Dec_k) + b2 IP_ik + g d_s^pre_ik + dl s^pre_ik
             + X_i'th + (X_i x Dec_k)'ps + a_k [+ a_i] + e_ik

Dec_k standardised (mean 24.68pp, sd 20.07), so b1 is per sd of Chinese share and b2 is
the effect at the average sector. WLS weighted by pre-period US imports, clustered by
country. N = 12,117 | 149 countries | 125 sectors | 99.94% of pre-period US value.
Code: `data/fit_longdiff_final.py`. Results: `data/longdiff_final.csv`.

### share_frac_policies

| spec | IP x Dec | IP |
|---|---|---|
| 1. a_k only | +0.392 (0.190)** | +0.541 (0.152)*** |
| 2. + capability levels | +0.299 (0.156)* | +0.403 (0.114)*** |
| **3. + capability x Dec [recommended]** | **+0.109 (0.151)** | **+0.255 (0.125)\*\*** |
| 4. + country FE | +0.277 (0.173) | +0.384 (0.139)*** |

### All measures, rungs 3 and 4

| measure | IP x Dec (3) | IP (3) | IP x Dec (4) | IP (4) |
|---|---|---|---|---|
| n_policies | −0.442 (0.173)** | −0.455 (0.157)*** | −0.106 (0.098) | −0.019 (0.075) |
| frac_policies | −0.227 (0.156) | −0.171 (0.136) | −0.007 (0.128) | +0.064 (0.107) |
| share_n_policies | −0.035 (0.097) | −0.085 (0.119) | +0.308 (0.074)*** | +0.326 (0.093)*** |
| share_frac_policies | +0.109 (0.151) | +0.255 (0.125)** | +0.277 (0.173) | +0.384 (0.139)*** |

### Reading

The decoupling interaction **is** significant before the capability controls (+0.392**,
+0.299*) and **is not** after capability x Dec enters (+0.109). This is precisely what
§3n predicted: capability x Dec independently predicts pre-trends, and policy-active
countries are more capable, so the two interactions compete for the same variation.

The main effect survives throughout: +0.255** at rung 3, +0.384*** under country FE.

So the currently defensible statement concerns the **level of targeting**, not the claim
that the payoff is **larger where China withdrew more**. That is weaker than the
decoupling framing wants.

**Caveat, not yet checked.** Five capability x Dec terms against one IP x Dec is a lot
of collinear controls; the drop from +0.299* to +0.109 could be over-control rather than
confound removal. The tests are (a) a joint test of psi = 0, and (b) a version carrying
only log_mva_pc x Dec, the one term that was significant in §3n.

---

## §3p. One control instead of five: X = {mva_gdp}

The five-control X of §3o was overloaded. Restricting to a single capability control,
MVA as a share of GDP:

    d_s_ik = b1 (IP_ik x Dec_k) + b2 IP_ik + g d_s^pre_ik + dl s^pre_ik
             + th mva_gdp_i + ps (mva_gdp_i x Dec_k) + a_k [+ a_i] + e_ik

N = 13,208 cells | 167 countries | 125 sectors | 99.99% of pre-period US value.
Code: `data/fit_longdiff_one.py` (regressor set as argv[1]),
`data/fit_longdiff_iso.py` (argv[1] = regressors, argv[2] = sample-defining set, so the
sample can be held fixed while the control set varies).
Results: `data/longdiff_final_mva_gdp.csv`, `data/longdiff_iso.csv`.

### share_frac_policies

| spec | IP x Dec | IP |
|---|---|---|
| 1. a_k only | +0.376 (0.182)** | +0.522 (0.148)*** |
| 2. + mva_gdp | +0.379 (0.192)** | +0.512 (0.150)*** |
| **3. + mva_gdp x Dec [recommended]** | **+0.378 (0.198)\*** | **+0.518 (0.156)\*\*\*** |
| 4. + country FE | +0.394 (0.183)** | +0.488 (0.147)*** |

Both terms are stable across all four rungs, interaction included. Other measures:
share_n_policies is null except under country FE (+0.353***, +0.344***); n_policies and
frac_policies are null throughout.

### What killed the interaction in §3o

Two candidates: the extra controls, or the larger sample (167 vs 149 countries).
Holding the 149-country sample fixed:

| X = | IP x Dec, rung 3 | rung 4 |
|---|---|---|
| mva_gdp only | +0.394 (0.207)* | +0.411 (0.191)** |
| mva_gdp + log_mva_pc | +0.213 (0.160) | +0.328 (0.171)* |
| all five | +0.109 (0.151) | +0.277 (0.173) |

Not the sample. **`log_mva_pc` is the variable that does the work** -- the same one that
was −5.21** in §3n. mva_gdp (structure: how manufacturing-intensive the economy is) and
log_mva_pc (level: industrial development per head) are different objects, and only the
second competes with IP x Dec. That is the variable to think about if a second control
is wanted; the other three were noise.

This supersedes the §3o recommendation.

---

## §3q. Pure DiD versus lagged dependent variable

A two-period DiD on the change assumes parallel trends and does not control for the
lagged outcome. An LDV model controls for the lagged outcome and does not assume
parallel trends. They rest on opposite assumptions and tend to BRACKET the truth
(Angrist & Pischke, MHE ch. 5). The §3p specification does both at once, so it is
neither cleanly. This runs the ladder between them.

    (a) pure DiD      d_s ~ IPxDec + IP                    + a_k
    (b) + pre-trend   d_s ~ IPxDec + IP + d_s^pre          + a_k
    (c) LDV           d_s ~ IPxDec + IP          + s^pre   + a_k
    (d) both  [§3p]   d_s ~ IPxDec + IP + d_s^pre + s^pre  + a_k

All carry mva_gdp + mva_gdp x Dec; each repeated with a country FE.
Code: `data/fit_did_vs_ldv.py`. Results: `data/did_vs_ldv.csv`.
N = 13,208 | 167 countries | 125 sectors.

### share_frac_policies

| | a_k only: IP x Dec | IP | + country FE: IP x Dec | IP |
|---|---|---|---|---|
| **a. pure DiD** | +0.345 (0.218) | +0.495 (0.178)*** | +0.361 (0.193)* | +0.459 (0.159)*** |
| b. + pre-trend | +0.368 (0.215)* | +0.508 (0.176)*** | +0.403 (0.179)** | +0.497 (0.148)*** |
| **c. LDV** | +0.404 (0.227)* | +0.549 (0.189)*** | +0.423 (0.188)** | +0.526 (0.157)*** |
| d. both [§3p] | +0.378 (0.198)* | +0.518 (0.156)*** | +0.394 (0.183)** | +0.488 (0.147)*** |

**Bracket: [+0.345, +0.404] on the interaction, [+0.495, +0.549] on the main effect** --
about 15% and 10% wide. DiD and LDV rest on opposite assumptions and give nearly the
same answer, so the parallel-trends-versus-lagged-outcome choice is not driving the
result. This addresses the concern that the §3p hybrid is neither design cleanly.

**Qualifications.** The interaction in the *pure* DiD with only sector FE is not
significant (se 0.218, p = .11); it crosses the threshold once any lagged term or the
country FE enters. Its significance is specification-dependent in a way the main
effect's is not. Other measures: share_n_policies null under a_k, strongly positive
under country FE across all four rungs (+0.31*** to +0.37***); n_policies and
frac_policies null throughout.

---

## §3r. The DiD as raw cell means

What the regression is a smoothed version of. Treatment: share_frac_policies > 0 in
2015-17. Exposure: Dec_k above the sector median (15.7pp). Outcome: change in share of
total US imports, 2015-17 to 2022-24, pp. All means weighted by pre-period US imports.
Code: `data/diag_did_cells.py`. N = 13,208 | 167 countries | 125 sectors.

### Raw

| | IP = 0 | IP > 0 | difference |
|---|---|---|---|
| High exposure | +1.802 (5,820 cells / 21.9% wt) | +2.226 (1,273 / 25.1%) | **+0.423** |
| Low exposure | +1.927 (4,856 / 18.2%) | +1.858 (1,259 / 34.9%) | **−0.069** |
| | | | **DDD = +0.492** |

Pooled DiD ignoring exposure: **+0.153**

### After removing the sector mean change (what alpha_k does)

| | IP = 0 | IP > 0 | difference |
|---|---|---|---|
| High exposure | −0.290 | +0.253 | **+0.543** |
| Low exposure | +0.304 | −0.158 | **−0.463** |
| | | | **DDD = +1.005** |

Pooled DiD ignoring exposure: **+0.034**

### Reading

The naive contrast -- (high IP, high exposure) minus (low IP, low exposure), the two
diagonal cells -- gives +0.299 raw and −0.052 demeaned. It discards the off-diagonal
cells and the two readings disagree in sign, which is the confounding problem: it cannot
separate gains from policy from gains from being in a sector China vacated.

The DDD uses all four cells. Within high-exposure sectors, policy-active pairs gained
+0.54pp more than non-policy ones; within low-exposure sectors they gained 0.46pp LESS.
The sign flip is why the **pooled** DiD is near zero (+0.034 demeaned): averaged over
all sectors industrial policy looks like nothing, and only looks like something once
conditioned on where China was withdrawing. That is the decoupling result, visible in
raw cell means before any regression.

**Caveats.** Binary dichotomisation, no controls, no pre-trend, no capability
adjustment, so this does not equal beta1 = +0.378 -- it is the transparent version, not
the estimate. The negative in low-exposure sectors may be partly mechanical: shares sum
to 100 within a sector, so a country gaining in its targeted high-exposure sectors can
lose share elsewhere without anything real happening to it.

---

## §3s. Is the gravity result expansion or reallocation?

The gravity specification carries alpha_ist, which absorbs country i's TOTAL exports in
sector s in year t. beta1 is therefore a pure DESTINATION margin -- US-bound relative to
i's own other markets -- and three different worlds produce the same +0.0524:

  1. total exports rose and the increment went disproportionately to the US
  2. total exports were flat and shipments were redirected from other markets to the US
  3. total exports fell everywhere but fell least in the US

If policy built capacity it should raise exports to ALL destinations, with the US
component being the extra tilt from vacated demand. This drops the destination
dimension and runs the same triple difference on a country-sector-year panel:

    X_ist = exp[ a_it + a_st + a_is + b (IP_ist x target_s x post_t) + eta IP_ist ]

a_it takes out i's overall export growth, a_st world demand for s, a_is the pair level.
Same treatment as the gravity headline: share_frac_policies lagged 3, standardised,
developing ex-China. 190 countries, 125 sectors, 2010-2024, clustered (i,s). The US
takes 18.3% of these exports. Code: `data/fit_reallocation.py`, `data/fit_realloc_us.py`
(the US cell needed a demeaning tolerance ladder; converged at maxiter 10,000, atol 1e-8).

| outcome | DDD | p | N |
|---|---|---|---|
| **(1) Total exports to world** | **−0.0059 (0.0032)\*** | 0.062 | 286,534 |
| **(2) US-bound only** | **+0.0175 (0.0039)\*\*\*** | <0.0001 | 241,134 |
| **(3) Non-US only** | **−0.0081 (0.0036)\*\*** | 0.024 | 286,523 |

### This is reallocation, not expansion

In the sectors a country targeted and China was leaving, exports to the US rose,
exports everywhere else fell by a comparable amount, and total exports did not rise.
The rough accounting holds: 0.183 x (+0.0175) + 0.817 x (−0.0081) = −0.0034, against an
estimated total of −0.0059.

**The gravity result is destination composition, not capacity.** Industrial policy in
decoupling sectors is associated with reorienting existing exports toward the vacated
US market, not with producing more.

### Caveats

1. The US cell is estimated on 241,134 observations against 286,534 for the other two,
   because PPML drops all-zero fixed-effect groups and 44.8% of country-sector-years
   have no US exports. That cell is on a selected sample of pairs that ever ship to the US.
2. The negative on total exports is marginal (p = 0.062) and small. The defensible
   statement is "no increase in total exports", not "a decrease".
3. This FE structure (a_it + a_st + a_is) is not the gravity headline's
   (a_ist + a_jst + a_ij), so these magnitudes are not comparable to +0.0524. The
   contrast ACROSS the three outcomes is the finding, not the levels.

### Wording for the note

> We find that a one-standard-deviation increase in policy targeting is associated with
> 5.2% higher exports to the US in decoupling sectors than in non-decoupling ones. This
> is a *destination* margin: the exporter x sector x year effects absorb a country's
> total exports in that sector, so the coefficient identifies where output was sold, not
> how much was produced. Estimating the same triple difference on a country-sector-year
> panel confirms the distinction. Exports to the US rise (+0.018, p<0.001), exports to
> all other destinations fall by a comparable amount (−0.008, p=0.024), and total
> exports do not increase (−0.006, p=0.062). Industrial policy in decoupling sectors is
> therefore associated with reorienting exports toward the vacated US market rather than
> with expanding export capacity.

---

## §3t. Staggered event study on sector-specific decoupling dates

### Is the event staggered?

Gating question for the whole design. Four definitions of sector k's event year E_k,
computed on China's share of TOTAL US imports in k, restricted to the 99 of 125 sectors
with a 2015-17 China share of at least 5pp. Code: `data/diag_event_timing.py`.

| definition | sectors | modal year | modal share | IQR |
|---|---|---|---|---|
| A. peak year | 99 | 2018 | 26% | 2014-2018 |
| B. largest one-year drop | 99 | 2019 | 44% | 2019-2022 |
| C. first year >=10% below 2015-17 baseline, stays below | 80 | 2019 | 44% | 2019-2022 |
| **D. same at 25%** | **62** | **2019** | **27%** | **2019-2023** |

Yes, staggered. D gives real dispersion (2017-2024, no year above 27%); B and C are
bimodal, a 2019 wave and a second in 2022-23.

### Design note: alpha_kt already absorbs the timing

If the specification carries alpha_kt, each sector already has a free time path, so
calendar-time and event-time estimates identify off the same variation. Event alignment
buys two other things: a dynamic profile in event time, and a pre-trend test aligned on
the event rather than on calendar years -- a much better test than §3m's, where sectors
at different event stages were pooled so the "pre-period" mixed treated and untreated.

It also sidesteps the staggered-DiD problem. Forbidden comparisons arise when
already-treated units serve as controls for later-treated ones. Here every country in
sector k shares E_k and the contrast is high-IP vs low-IP within the same sector-year,
so alpha_kt confines all comparisons to clean cells.

    s_ikt = sum_tau beta_tau (IP_ik x 1[t-E_k=tau])
          + sum_tau psi_tau  (mva_gdp_i x 1[t-E_k=tau])
          + a_ik + a_kt + e_ikt                            ref tau = -1

Definition D, event time binned at [-6,+5]. Developing ex-China, weighted by pre-period
US imports, clustered by country. N = 118,530 | 167 countries | 62 sectors | 6,585
pairs. Sectors contributing thin out with horizon: 62 at tau<=0, 56 / 45 / 37 / 30 / 19
at tau = 1..5. Code: `data/fit_event_stagger.py`.

### Panel 1 -- all countries

| tau | IP x 1[tau] | MVA x 1[tau] |
|---:|---|---|
| −6 | −0.273 (0.148)* | −0.428 (1.146) |
| −5 | −0.119 (0.088) | −0.165 (0.501) |
| −4 | −0.129 (0.073)* | +0.006 (0.271) |
| −3 | −0.118 (0.070)* | −0.058 (0.235) |
| −2 | −0.207 (0.071)*** | +0.084 (0.110) |
| **−1** | — ref — | — ref — |
| 0 | −0.075 (0.075) | **+0.729 (0.106)\*\*\*** |
| 1 | −0.203 (0.119)* | **+0.967 (0.125)\*\*\*** |
| 2 | −0.313 (0.189) | **+0.654 (0.292)\*\*** |
| 3 | −0.067 (0.417) | +0.353 (0.271) |
| 4 | −0.160 (0.708) | −0.350 (0.524) |
| 5 | +1.592 (1.277) | −0.235 (0.499) |

Pre-trend joint **F(5,166) = 5.45, p = 0.0001**. Post: 0 of 6 significant.

### Panel 2 -- by MVA tercile

| stratum | N | pre-trend F | p | post mean | post sig |
|---|---|---|---|---|---|
| Low MVA | 30,402 | 8.47 | <0.0001 | −0.058 | 2/6 (both ~ −0.015) |
| Mid MVA | 40,428 | 2.63 | 0.034 | −0.066 | 1/6 |
| **High MVA** | 47,700 | **4.83** | **0.001** | +0.177 | **0/6** |

### Reading

**Pre-trends fail in every stratum**, including high-MVA. This was the better test that
event alignment was supposed to buy, and it rejects. The IP pre-coefficients are
consistently NEGATIVE (−0.27 to −0.12): high-IP country-sectors were losing US market
share in the run-up to their sector's event. That is selection on *decline*, the
opposite of the Rotunno-Ruta pattern -- a hypothesis deserving its own test, not a
conclusion.

**The high-MVA test is uninformative, not negative.** The stratified comparison gives
0/6 post coefficients significant, but with SEs reaching 0.74 and 1.42 at tau = 4, 5
where only 30 and 19 sectors contribute. Power problem, not evidence of no effect.

**MVA does what IP was supposed to do.** Capability x event-time is +0.73***, +0.97***,
+0.65** at tau = 0, 1, 2 -- large, precise, and timed exactly to the event. Aligned on
China's withdrawal, the manufacturing base predicts who picks up the slack and policy
adds nothing detectable on top. Given the failed pre-trends this is not stated as a
finding, but it is the clearest signal in the table and it is not friendly to the
paper's argument.

### Open

1. The tau = −6 bin pools everything six or more years before the event and could be
   driving the pre-trend rejection alone. Re-test on tau in [−5,−2] only.
2. Re-run on `share_n_policies`, and on definition C (10% threshold, 80 sectors) for
   more power at long horizons.

---

## §3u. Who the §3t event study actually compares

Code: `data/diag_who_compared.py`.

**Structure.** Unit = country-sector pair, 6,585 of them, 167 countries x 62 event
sectors. alpha_ik removes the pair level, so identification runs over time within a
pair. alpha_kt confines every comparison to a single sector-year cell. IP_ik is
time-invariant, so beta_tau traces how the CROSS-COUNTRY IP gradient inside a sector
evolves around that sector's own event.

So it is a **between-country** comparison executed inside sector-year cells. There is no
alpha_it: a country's own trajectory is not absorbed, and psi_tau (MVA x event time)
plus the tercile split are the only stand-ins. Weaker than the country-FE variant of the
long difference, where two countries are never compared at all.

**Comparison set.** Median event sector has 109 developing suppliers, ~20 with any
policy targeting (min 12, max 27; no sector has zero). Each beta_tau is roughly
"20 policy-active countries against ~89 without, inside the same sector-year." Within-
sector sd of IPz is 0.989 against a total of 1.000, so essentially all IP variation is
across countries within a sector -- which is what the design wants.

**Weight.** Policy-active pairs are 18.8% of pairs but 52.7% of weight.

| | weight | has policy in |
|---|---|---|
| Mexico | 37.6% | 18 of 62 sectors |
| Vietnam | 12.1% | 29 of 62 |
| Malaysia | 9.4% | 18 of 62 |
| India | 7.6% | 60 of 62 |
| Thailand | 7.3% | 4 of 62 |
| Indonesia | 4.3% | 61 of 62 |
| Philippines | 2.6% | **0 of 61** |

Top five = 74% of weight.

### The concrete version: sector 1410, wearing apparel, event 2022

Largest event sector by US import value, 162 developing suppliers.

| country | pre-period US imports ($k) | IP | MVA/GDP |
|---|---:|---:|---:|
| **Vietnam** | 9,697,227 | **0.0000** | 21.2 |
| Bangladesh | 5,167,714 | 0.0049 | 19.1 |
| Indonesia | 4,166,236 | 0.0014 | 20.6 |
| India | 3,606,867 | 0.0009 | 13.9 |
| Mexico | 3,492,706 | 0.0025 | 20.3 |
| Cambodia | 2,225,182 | 0.0000 | 22.0 |
| Sri Lanka | 1,944,644 | 0.0000 | 15.6 |
| Honduras | 1,637,378 | 0.0000 | 17.6 |

The estimator in one cell: Bangladesh, Indonesia, India and Mexico (some apparel
targeting) against Vietnam, Cambodia, Sri Lanka and Honduras (none), all supplying US
apparel, tracked from 2022.

**This shows the problem directly.** In the single largest event cell the dominant
supplier -- Vietnam, twice the size of the next -- sits in the CONTROL group with zero
recorded apparel policy. Under WLS that pair does a great deal of work and pulls
beta_tau toward "no policy did better". Vietnam's MVA/GDP of 21.2 is also above
Bangladesh's 19.1, so the capability control does not separate them: this is a
high-capability, zero-policy country outperforming.

That may be a real fact about apparel, or GTA may under-record Vietnamese industrial
policy. Either way the design's answer in its biggest cell is set by one country's
treatment status.

### Open

Two checks that would distinguish these: the unweighted event study, and
leave-Mexico-and-Vietnam-out.

---

## §3v. Electronics, and the within-MVA-bracket event study

### Support for bracket comparisons

Replacing alpha_kt with alpha_{k,bracket,t} confines every comparison to the same
sector, same MVA bracket, same year. Code: `data/diag_brackets_elec.py`.

| brackets | cells | countries/cell (med) | cells with both treated & untreated | weight there | pairs |
|---|---|---|---|---|---|
| Decile | 618 | 12 | 374 (61%) | **99.7%** | 4,578 |
| Quintile | 310 | 23 | 198 (64%) | 99.8% | 4,834 |
| Tercile | 186 | 38 | 137 (74%) | 99.8% | 5,358 |

Feasible: the unsupported cells are small ones.

### Electronics (ISIC 26-27), China's share of US imports

| sector | 2015-17 | 2022-24 | change | event |
|---|---:|---:|---:|---:|
| 2620 computers & peripherals | 60.2 | 30.0 | **−30.2** | 2020 |
| 2630 communication equipment | 58.8 | 45.0 | −13.7 | 2024 |
| 2740 lighting equipment | 51.4 | 32.1 | −19.3 | 2022 |
| 2732 other electronic wire | 43.1 | 23.5 | −19.6 | 2020 |
| 2790 other electrical equipment | 41.7 | 21.1 | −20.6 | 2021 |
| 2640 consumer electronics | 44.4 | 31.0 | −13.4 | 2023 |
| **2720 batteries** | 29.8 | **49.1** | **+19.4** | — |
| 2610 electronic components | 14.3 | 7.3 | −7.0 | 2019 |

Computers is the sharpest decoupling episode in the dataset. Batteries went the other
way: China GAINED 19pp.

**Top developing suppliers in 2620, event 2020**

| country | share 15-17 | share 22-24 | change | IP | MVA/GDP |
|---|---:|---:|---:|---:|---:|
| Mexico | 16.12 | 22.56 | **+6.44** | **0.0000** | 20.3 |
| Vietnam | 1.57 | 8.43 | **+6.86** | 0.0016 | 21.2 |
| Malaysia | 1.92 | 3.09 | +1.17 | 0.0003 | 21.7 |
| Thailand | 4.79 | 5.49 | +0.70 | 0.0000 | 26.6 |
| Philippines | 1.42 | 1.65 | +0.22 | 0.0000 | 18.2 |
| Indonesia | 0.26 | 0.29 | +0.03 | 0.0029 | 20.6 |

Mexico, Vietnam and Malaysia are in the same MVA bracket (20.3, 21.2, 21.7). Vietnam has
the most policy and gained most, but Mexico has ZERO recorded computer policy and gained
nearly as much, while Malaysia has intermediate policy and gained a sixth as much.
Indonesia has the highest IP of the six and gained 0.03pp.

### The bracket event study

    s_ikt = sum_tau beta_tau (IP_ik x 1[t-E_k=tau]) + a_ik + a_{k,br,t} + e_ikt

MVA x event-time terms dropped: MVA is near-constant inside a bracket, so a_{k,br,t}
does that job non-parametrically. Code: `data/fit_event_decile.py`.

| brackets | weighting | pre-trend F(5,166) | p | post mean | post sig |
|---|---|---:|---:|---:|---|
| Decile | weighted | 7.55 | <0.0001 | +0.240 | 0/6 |
| **Decile** | **unweighted** | **0.99** | **0.428** | **−0.004** | **0/6** |
| Quintile | weighted | 4.31 | 0.0010 | +0.207 | 0/6 |
| Quintile | unweighted | 0.92 | 0.468 | −0.006 | 0/6 |
| Tercile | weighted | 5.78 | 0.0001 | +0.145 | 0/6 |
| Tercile | unweighted | 0.93 | 0.461 | −0.005 | 0/6 |

**Decile, unweighted** (the specification that passes): tau = −6..−2 run
−0.045*, −0.028*, −0.033*, −0.026*, −0.022; tau = 0..+5 run −0.002, −0.016, +0.001,
+0.000, −0.005, −0.003, with SEs 0.007-0.022.

### Reading

1. **The MVA brackets barely matter.** Deciles, quintiles and terciles give almost
   identical numbers. Confining comparisons to similar countries is not what changes the
   answer, which suggests MVA was not the missing control in earlier specifications.
2. **The §3t pre-trend failure was a WEIGHTING artefact.** Weighted F = 7.55
   (p<0.0001); unweighted F = 0.99 (p = 0.43), flat. Mexico and a handful of large pairs
   drove the rejection.
3. **But the specification that passes gives a precise zero.** Post coefficients −0.016
   to +0.001, SEs 0.007-0.022, so 95% intervals of roughly ±0.03pp. Not an underpowered
   null -- a bounded one.

**The tension to resolve.** Unweighted answers "did the typical COUNTRY-SECTOR with
policy do better" -- flat pre-trends, tight zero. Weighted answers "did the typical
DOLLAR of trade flow toward policy-active suppliers" -- the economically relevant
question for who captured the decoupling flows, and where §3q found +0.38 to +0.52 --
but it fails pre-trends here and its event-time SEs are too wide to say anything. Two
different estimands that genuinely disagree.

### Open

Whether the weighted result is Mexico specifically. Leave-Mexico-out on both the
long difference and this event study would settle it.

---

## §3w. The bracket event study across all four policy measures

24 fits: 4 measures x 3 bracket schemes x weighted/unweighted. Code:
`data/run_decile_all.sh` over `data/fit_event_decile.py`. Summary:
`data/decile_all_summary.csv`; full log `data/decile_all.log`.

### Pre-trend joint test F(5,166), p in brackets

| measure | decile W | decile U | quintile W | quintile U | tercile W | tercile U |
|---|---|---|---|---|---|---|
| n_policies | 5.36 (.0001) | 2.21 (.056) | 3.94 (.002) | 2.81 (.018) | 33.74 (<.0001) | 2.89 (.016) |
| frac_policies | 14.99 (<.0001) | 3.34 (.007) | 5.09 (.0002) | 4.14 (.001) | 57.10 (<.0001) | 4.86 (.0004) |
| **share_n_policies** | 10.30 (<.0001) | **0.87 (.50)** | 5.30 (.0002) | **0.99 (.42)** | 10.35 (<.0001) | **0.56 (.73)** |
| **share_frac_policies** | 7.55 (<.0001) | **0.99 (.43)** | 4.31 (.001) | **0.92 (.47)** | 5.78 (.0001) | **0.93 (.46)** |

### Post-period mean and significant coefficients (decile brackets)

| measure | weighted | unweighted |
|---|---|---|
| n_policies | −0.397, **4/6 sig** | +0.010, 0/6 |
| frac_policies | −0.141, 2/6 sig | −0.031, 0/6 |
| share_n_policies | −0.558, 2/6 sig | +0.011, 0/6 |
| share_frac_policies | +0.240, 0/6 | −0.004, 0/6 |

### Findings

**1. Targeting measures pass pre-trends; volume measures do not.** Unweighted,
share_n_policies and share_frac_policies give F between 0.56 and 0.99 (p = 0.42-0.73)
in every bracket scheme. n_policies and frac_policies fail in five of six cells, and
frac_policies decisively (p = 0.0004 to 0.007).

This is an INDEPENDENT argument for the share measures. §3j could only establish they
were more precise, since the point estimates were similar; here they are the only ones
whose pre-period is flat.

*Caveat:* share measures have less between-country variation by construction, so they
mechanically have less scope to display differential trends. Not a clean win without
checking that.

**2. Everything weighted fails, and badly.** F reaches 33.7 and 57.1 for the volume
measures at terciles. The negative-and-significant post coefficients there (n_policies
−0.40, 4/6 significant) are not interpretable: they sit on a rejected pre-period.

### share_n_policies, decile, unweighted -- the best-behaved cell (F = 0.87, p = 0.50)

| tau | IP x 1[tau] |
|---:|---|
| −6 | −0.0455 (0.0255)* |
| −5 | −0.0310 (0.0167)* |
| −4 | −0.0313 (0.0170)* |
| −3 | −0.0242 (0.0125)* |
| −2 | −0.0196 (0.0116)* |
| **−1** | — ref — |
| 0 | +0.0000 (0.0086) |
| 1 | +0.0002 (0.0134) |
| 2 | **+0.0280 (0.0165)\*** |
| 3 | +0.0180 (0.0163) |
| 4 | +0.0177 (0.0190) |
| 5 | +0.0040 (0.0175) |

Post-period positive in five of six years and marginally significant at tau = +2, but
the post mean is +0.011 with none surviving at 5%. A hint, not a result. The
individually significant pre-coefficients converging toward zero at tau = −1 argue for
running the tau in [−5,−2] version before reading anything into the shape.

### Open

1. tau in [−5,−2] pre-trend test, dropping the endpoint bin.
2. Whether the share measures' pre-trend pass is mechanical (less between-country
   variation) rather than substantive.
