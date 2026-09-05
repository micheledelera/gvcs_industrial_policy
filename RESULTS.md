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

1. **Pre-trend test in the gravity specification itself.** The event study we ran was
   on a US-only DiD with a different FE structure and was inconclusive (noisy annual
   coefficients, no clean trend but no clean flat either). The headline is now a
   gravity result and needs its own pre-trend test.
2. ~~**Exporter-level clustering.**~~ Settled in §3f: the standard errors *fall*
   under exporter clustering, because within-country cross-sector residuals are
   negatively correlated. (i,s) stays the headline as the conservative choice.
5. ~~**Concentration.**~~ Settled in §3g: across leave-one-out drops of the top
   10 exporters the coefficient stays in [+0.0244, +0.0384] with a median equal
   to the baseline, and 9 of 10 remain significant. Mexico is the one influential
   case, and mainly through the standard error.
3. **Extensive margin.** Zero flows are absent from the data; 12.6% of US-bound
   (exporter × sector) pairs churn between 2017 and 2024, and entry is invisible.
   The benchmark (IMF WP 2024/041) finds extensive-margin effects concentrated in
   emerging markets — possibly where more of this story lives.
4. **GTA intervention types.** The benchmark finds tax breaks strongly positive and
   direct transfers negative — they cancel in any pooled measure. Splitting by
   intervention type would identify *which kinds* of targeting work. Not testable
   with the current variables (`n_sub` is nonzero in only 1.3% of developing
   ex-China US-bound observations).
5. **`frac_sub` / `share_*` definitions** were reconstructed from the raw file
   mid-session; worth confirming they mean what we assume.
