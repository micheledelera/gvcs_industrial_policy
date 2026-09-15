# Synthetic control protocol — Cunningham, *Mixtape* ch. 11, applied to this project

A working document. We walk the chapter top to bottom, perform each step, discuss it, and
record the conclusion here before moving on. Status markers:

- `[ ]` not yet done
- `[~]` partially done somewhere in RESULTS.md, needs redoing to protocol standard
- `[x]` done to protocol standard, conclusion recorded
- `[!]` done and found to break in our setting — deviation documented

## The setting mismatch, stated once

| | ADH / chapter | this project |
|---|---|---|
| treated units | 1 (California; Texas) | 500–1,083 country-sectors, or 1 country |
| donors | 38 states | 2,497–3,658 country-sectors, or ~100 countries |
| pre-periods | 19 (1970–88) | 11 (2007–17) |
| T₀ vs N₀ | T₀ ≫ needed | T₀ = 11 against thousands of donors |

**MERGED DESIGN (revised).** The two tracks are now one: follow the Mixtape's step
sequence and discipline throughout, and substitute Xu's GSC estimator at every step that
assumes a single treated unit. The single-treated-unit case study (Vietnam) is retained as a
worked **illustration**, not a parallel track, because it is the only place ADH's full
apparatus — post/pre RMSPE ratio, exact p-value, placebo spaghetti — applies literally.

### What the merge keeps, changes, and costs

| Mixtape step | under the merge |
|---|---|
| Part A, design stage | **unchanged.** And the discipline survives: GSC picks r by CV on treated *pre-treatment* outcomes only, so design is still separable from estimation (Rubin 2007/2008, ADH) |
| B1, solve W\* given V, and V by min pre-RMSPE | **dissolved.** GSC projects onto an estimated factor space instead of weighting covariates, so there is no V. This closes the single largest gap against the chapter *and* removes a specification-search margin |
| B2, balance table | **modified.** Treated mean vs imputed-counterfactual mean vs donor-pool mean, on covariates and on pre-period outcomes, plus a comparison of estimated factor loadings |
| B3, weights table | **real cost.** GSC has no explicit per-donor weights, and transparency is the feature the chapter prizes most. Mitigation: GSC's imputation is linear in the controls' outcomes through F̂ and Λ̂, so implied donor weights can be derived and tabulated. That derivation is ours, not standard — label it as such |
| B4/B5, paths and gap figures | **unchanged.** Xu's own Figure 2 is exactly these two panels |
| C1–C4, ADH exact p-value | **replaced as primary, retained as secondary.** Primary is the validated parametric bootstrap with country blocks (§9b). Secondary is an ADH-style randomisation test *on the ATT*, permuting treatment by country block. This combination **fixes** the §7/C4 defect: there is now one aggregate statistic to rank, so no aggregation of 500 exact p-values is needed |
| C8, post/pre RMSPE ratio and the 2× filter | **demoted from inference to robustness.** Both are single-unit devices; per-unit pre-fit is still computable, so the filter becomes a sample restriction. §5c's finding that the ratio statistic runs backwards here stands, so the test statistic is the ATT |
| C5–C7, Figures 11.5/11.6/11.7 | **survive modified.** Spaghetti of per-treated-unit GSC gaps against placebo gaps; histogram of permuted ATTs with the actual marked |
| D1, placebo-in-time | **unchanged**, and applies directly to GSC |
| D2, leave-one-donor-out | **improved.** With 2,497 controls, dropping one is meaningless; becomes **leave-one-country-out**, which also matches our clustering |
| D3, lag-set sweep | **improved.** Becomes the r sweep plus the CV curve — a strictly smaller specification space |
| new, GSC-specific | factor-loading overlap plot (Xu Fig. 3b), and his instruction to check imputed counterfactuals lie "within reasonable intervals" |

### The binding constraint: T₀ = 11

The Zenodo source panel is 2007–2024 (`code/gvc_ip_gravity.py` header), so **T₀ = 11 is a
data constraint, not a build choice** — extending it means rebuilding from raw BACI, which is
not in this repository. This matters because:

- Xu cautions against T₀ < 10; we are at 11, just over his floor.
- §9b found the CV selects r = 0 more often as T₀ falls, and **T₀ = 11 is shorter than every
  cell validated so far** (coverage was checked at T₀ = 15 and 20, the CV at T₀ ≥ 15).
- So the merged design needs **one more validation cell at T₀ = 11 exactly**, for coverage
  and for the CV's r distribution, before GSC is trusted on this panel. Added as B2.0b.

### Why Xu (2017) is the right estimator for the multiple-treated-unit steps

| our documented failure | what GSC does about it |
|---|---|
| §5b: 500 separate SCs → 500 exact p-values I had no valid way to aggregate (§7/C4) | IFE model is fitted **once**; all treated counterfactuals come from one run, giving one ATT and one SE |
| §4a–§4c: no untreated Vietnam; treated units outside the convex hull | GSC imputes from **factor loadings**, not a convex combination. Xu cites Gobillon–Magnac (2016): IFE beats SC exactly when treated and control factor loadings lack common support |
| §8d: τ ranged −0.05 to +0.24 across donor-pool size K | **No donor pool to choose.** "No observations are discarded from the control group" |
| never optimised V; never swept the lag set (B1, D3) | one cross-validated hyperparameter, **r**, chosen by leave-one-out CV on treated pre-periods — Xu's explicit anti-specification-search design |
| §5b: inference by median of 500 p-values; Fisher combination invalid | **parametric bootstrap** (Xu Algorithm 2), blocked at the unit level, giving interpretable frequentist SEs |
| §3t: sector-specific decoupling dates never entered an SC | staggered adoption handled natively |

### And the four caveats we must carry

1. **T₀ = 11 is marginal.** Xu: "users should be cautious when T₀ < 10 or N_co < 40." Our
   N_co = 2,497 is far above 40, but T₀ = 11 sits just over his floor, and he warns the
   incidental-parameters problem biases ATT when T₀ is small. State this up front.
2. **GSC buys our convex-hull problem off with extrapolation.** Xu, on his own method: it
   "will still impute treated counterfactuals based on model extrapolation, which may lead to
   erroneous conclusions," where canonical SC would visibly fail instead. So GSC *succeeding*
   where §4a failed is not by itself good news — it must be checked, not celebrated.
3. **Strict exogeneity (his Assumption 2) rules out treatment responding to past outcomes.**
   §5i's reverse regression found *volume* policy responds to lagged export growth (joint
   p = 0.034) while *targeting* does not (p = 0.53). So Assumption 2 is **violated for volume**
   and defensible for targeting or persistence. This settles the measure choice on the
   estimator's own terms.
4. **Assumption 5 (cross-sectional independence) fails at our unit level.** Country-sectors
   within a country are correlated. Xu's bootstrap blocks at the unit; **ours must block at
   the country.** §8d showed country-block versus unit-level resampling changed the null
   width by 2.6× and p from 0.003 to 0.033.

### The substantive question GSC will pose

GSC's factor model **nests** two-way fixed effects (set λ_i1 = α_i, f_2t = ξ_t) but it does
**not** nest country × year fixed effects — the margin on which every estimate in §5–§8 died.
With r factors it may or may not absorb country-year variation. So GSC is a strictly weaker
control than α_it, and the comparison of the three — two-way FE, GSC with r factors, and
saturated α_it + α_kt — is itself the test of whether the latent factors are picking up the
country story. Xu's **factor-loading overlap plot** (his Figure 3b) is the diagnostic that
tells us which.

## Part A — Design stage (pre-treatment data only; no peeking at outcomes)

The chapter's argument for why this stage is separable: SC needs no post-treatment outcomes
to build the counterfactual, so design can be fixed before estimation. We honour that by
freezing A0–A4 in this file before running Part B.

- `[x]` **A0. Question, treatment, date, outcome — DECIDED.**
  Evidence in `data/a0_outcomes.py`.

  **Question.** Does industrial policy help developing countries capture the reallocation of
  US import demand away from China after 2018?

  **Event and date: 2018.** US Section 301 tariffs from July 2018; §3ae dated the divergence
  in our own data to 2018–19, before COVID. The event is *US policy toward China*, hence
  exogenous to any individual developing country-sector — which is the whole reason this
  design is possible.

  **Treatment — a precision point that shapes A2/A3.** This is *not* "policy adopted in
  2018". Policy is pre-determined, measured before the event. What switches on in 2018 is the
  **interaction**: decoupling × policy status. So D_it = 1 for policy-using country-sectors
  after 2018, and the counterfactual being imputed is "what would a policy-using
  country-sector have done, facing the same 2018 shock, had it behaved like a non-using one".
  A synthetic unit built from never-targeted donors is exactly that — **provided the donors
  were exposed to the same decoupling shock.** Otherwise the counterfactual becomes "no
  policy AND no decoupling". §5b handled this by matching donors on China's share of US
  imports; GSC uses all controls, so exposure must enter either as a covariate in X or be
  left to the factors. Carried forward to A2/A3.

  **Outcome — the fork.** Measured on the balanced sample of 5,168 developing ex-China
  country-sectors (1,083 targeted 6+ of 2009–17, 2,499 never):

  | outcome | sd | skew | treated mean | donor mean | treated outside donor range |
  |---|---|---|---|---|---|
  | `lnX` ln US imports | 2.998 | +0.19 | 8.909 | 6.309 | 0.7% |
  | `lnS` ln share of US imports in sector k | 3.108 | **−0.05** | −7.102 | −9.911 | 0.0% |
  | `lnD` ln US share of the country-sector's exports | 1.827 | −0.69 | −3.427 | −2.910 | 0.0% |

  **DECIDED: `lnS` primary, `lnD` secondary, `lnX` as a check.**
  - `lnS` is the question as posed — decoupling reallocates a roughly fixed pot of US
    sectoral demand, so capturing it *is* a share gain. It is scale-free (§4c found that
    decisive for hull feasibility), the best-behaved of the three (skew −0.05), and since
    ln S = ln X − ln(sector-year total) it **absorbs sector-year shocks by construction** —
    the outcome definition does the work of α_kt, leaving the factor structure only the
    country dimension to handle.
  - `lnD` is the destination margin, where the gravity +5.2% lives, and the one outcome the
    country-year confound cannot touch because it nets out country-sector scale. It is
    arguably *better identified* but answers a narrower question — redirection toward the US
    rather than capture of US demand. Hence secondary, and the natural bridge to §3.
  - `lnX` is **retained as a check**, not dropped (user decision). It is neither scale-free
    nor free of sector-year shocks, so it makes GSC do the most work with the least help —
    but it is what §5b used, so keeping it makes every new result directly comparable to the
    existing §5 line rather than only to itself.

  **Weighting, per §6.** Log share is unit-weighted — the "among the sectors" estimand. The
  dollar-weighted version is a *different estimand*, not a robustness check, and §6 found the
  two differ by an order of magnitude. Both to be reported, labelled as distinct questions.

  **Note on the treated/donor gap.** On `lnX` and `lnS` treated units start well above donors
  (2.6–2.8 log points: policy goes to big sectors). On `lnD` they start *below* (−3.43 vs
  −2.91) — policy-using sectors are *less* US-oriented pre-2018. The sign of the baseline gap
  flips with the outcome, which is worth remembering when reading any level comparison.
- `[x]` **A1. Treated units — DECIDED.** Evidence in `data/a1_treated.py`,
  `data/a1_illustration.py`.

  **Unit.** Country-sector (i,k), developing economies excluding China, balanced positive US
  imports 2007–2024 → **5,168 units**. Country-level is ruled out by §4a (no untreated
  Vietnam; canonical SC feasible for 5–20% of treated trade value); §4b found country-sectors
  fix the hull problem; and the question is about sectors capturing demand.

  **Treated: targeted in ≥ 6 of 2009–17 → 1,083 units, 25 countries, 123 sectors.** The
  §8b persistence definition, the only one of §7's seven to survive: no country-portfolio
  denominator (§5k), best-balanced treated group in the exercise, and a genuine binary, which
  both canonical SC and GSC require.

  **Controls: strictly clean → 2,223 units.** New check, never run before. Of the 2,499 units
  never targeted 2009–17, **276 (11%) acquire policy after 2018** — so they are treated in
  the post-period, which violates Xu's requirement that controls be "never exposed to the
  treatment in the observed time span" and would bias the ATT toward zero. Dropped. Total:
  1,083 treated + 2,223 clean controls, with 1,862 intermittent/contaminated units excluded.

  **Threshold: 6+ primary, with 5+/7+/8+/9+ swept — and the sweep is itself a finding.**

  | threshold | treated | countries | largest country | top 3 | pre-2018 lnS trend gap/yr |
  |---|---|---|---|---|---|
  | 5+ | 1,276 | 29 | 9% | 28% | **+0.0035** |
  | **6+** | **1,083** | **25** | **11%** | **32%** | **+0.0092** |
  | 7+ | 898 | 22 | 13% | 35% | +0.0101 |
  | 8+ | 689 | 20 | 15% | 41% | +0.0182 |
  | 9+ | 323 | 16 | 17% | 49% | +0.0194 |

  The pre-treatment trend gap between treated and clean controls **rises monotonically with
  the threshold**, and country concentration worsens with it too. So a *tighter* definition of
  "persistent" is worse on both counts — the opposite of the natural instinct. At 9+ the gap
  compounds to ~0.21 log points over the pre-period; at 6+ it is ~0.10. Read as selection:
  the more persistently a sector is targeted, the more it was already gaining share before
  2018. 6+ is chosen as two-thirds of years, a defensible reading of "persistent", and it
  matches §8b so results stay comparable.

  **Single-unit illustration: Vietnam, ISIC 2630 (communication equipment).** $5.1bn of US
  imports 2015–17, 4.48% of US imports in the sector, **China 58.8%** of US imports in that
  sector, targeted 6 of 9 years, and **44 strictly-clean donors within the same sector**.

  **CORRECTION.** I first reported that the persistence treatment does not select Vietnam's
  significant sectors and that its largest targeted sector was $5m. That was a units error —
  `imports` is in **thousands of USD** (calibrated: US total 2016 = 1,937,439,360 raw =
  $1.94tn; Mexico $264bn; Vietnam $41.8bn; China $463bn), so the figure was $5.1**bn** and
  the size filter in the first candidate search demanded $200tn, returning zero candidates.
  With correct units, **29 of the 1,083 treated units** are large, decoupling-exposed and
  donor-rich: Bangladesh 1410 apparel ($5.2bn), Vietnam 2630 ($5.1bn), Indonesia and India
  1410, Mexico 2599 fabricated metal ($2.7bn), India 1392 textiles ($2.6bn), and so on.

  **Coverage caveat that survives the correction.** The treatment still touches only a small
  share of the value in the countries that actually gained:

  | | units in sample | targeted 6+ | their US imports | country total | share |
  |---|---|---|---|---|---|
  | Vietnam | 109 | 5 | $5.7bn | $42.1bn | 14% |
  | Mexico | 124 | 7 | $13.7bn | $271.5bn | 5% |
  | Thailand | 123 | 1 | $0.2bn | $30.0bn | 1% |
  | Malaysia | 110 | 2 | $0.1bn | $35.2bn | 0% |
  | Bangladesh | 52 | 3 | $5.2bn | $6.2bn | **84%** |

  So the ATT will be identified off a subset covering 0–14% of the winners' export value
  (Bangladesh excepted). That is a softer version of §5k/§7's point — persistent GTA policy
  is not absent from the winners, but it is not where most of their value sits either. To be
  stated as a scope condition, not discovered later.
- `[x]` **A2. Donor pool — DECIDED.** Evidence in `data/a2_donors.py`. Under the merge this
  is not "which donors per treated unit" but "which units are legitimate controls at all",
  plus the requirement carried from A0 that donors face the same decoupling shock.

  **The A0 trap does not bind.** China's share of US imports in the unit's sector, 2015-17:

  | | n | mean | p10 | p50 | p90 |
  |---|---|---|---|---|---|
  | treated (6+ of 9) | 1,083 | 23.9% | 2.7% | 17.5% | 56.0% |
  | clean controls | 2,223 | **26.7%** | 1.8% | 21.6% | 58.4% |

  **73% of clean controls sit inside the treated units' p10-p90 exposure range**, and in
  high-exposure sectors (China >= 25%) the shares are 37% of treated against 43% of controls.
  Controls are if anything *slightly more* exposed than treated units, which biases against
  finding a policy effect since controls had marginally more opportunity. So the
  counterfactual is not "no policy AND no decoupling", and **no exposure band is needed**.

  **Primary pool: all 2,223 strictly clean developing controls, pooled across sectors**, with
  **Dec_k x Post entered as a covariate in X** so the size of the reallocation opportunity is
  held constant with a common beta. No per-treated-unit pool, therefore **no K parameter at
  all** — a direct gain over §5b and §8d, where tau swung from -0.05 to +0.24 across K.

  **Explicitly rejected: a band restriction on Chinese share.** It would reintroduce exactly
  the tuning parameter §8d showed destabilises the estimate, and the overlap above makes it
  unnecessary.

  **Secondary design — per-sector GSC, exposure fixed by construction.** Because lnS is a
  share *within* a sector, donors in the same sector share the treated unit's exposure
  exactly. Feasibility against Xu's N_co >= 40 caution:

  | donors per sector | sectors | treated units covered | treated US imports covered |
  |---|---|---|---|
  | >= 40 | 12 of 125 | 140 (13%) | **$49.2bn (35%)** |
  | >= 30 | 22 | 260 (24%) | $62.5bn (44%) |
  | >= 20 | 42 | 463 (43%) | $94.2bn (67%) |
  | >= 10 | 84 | 841 (78%) | $134.0bn (95%) |

  Run per-sector on the **12 sectors with >= 40 clean donors** — a minority of units but a
  third of treated value, and it includes the substantively interesting ones: 1410 apparel
  (12 treated, 66 donors, $13.8bn), 2630 communication equipment (14 treated, 44 donors,
  China 58.8%, $6.4bn), 2599 fabricated metal, 1392 textiles (China 58.1%). This is the
  design-purist answer to A0 and is reported alongside the pooled estimate.

  **Robustness only: advanced economies.** 3,776 advanced-economy country-sectors have a
  balanced positive US series and would nearly double the donor pool. Kept out of the primary
  pool because they run their own industrial policy (CHIPS Act, EU programmes) and so are not
  "never exposed" in Xu's sense.

  **ADH's own exclusion analogue** — dropping states with their own tobacco programmes — is
  the 276 contaminated controls already removed in A1.
- `[x]` **A3. Matching variables — DECIDED.**

  **First: two of the chapter's choices dissolve here, and that is the point of the merge.**
  - *Which pre-treatment lags to match on* — ADH's `synth` makes this a genuine choice
    (`cigsale(1988) cigsale(1980) cigsale(1975)`). GSC's step 2 projects the treated unit's
    outcome onto the factor space using **all T₀ pre-periods**, so there is no lag set to
    choose. **Ferman–Pinto–Possebom's specification search over lags does not apply**, which
    retires protocol step D3's lag sweep. This is the third such margin to close under the
    merge, after V (B1) and the donor-pool size K (A2).
  - *Time-invariant covariates* — central in ADH (Table 11.1 carries GDP per capita, retail
    price, beer consumption) but **absorbed by λ_i** in GSC, since a unit-specific constant is
    exactly what a factor loading is. They cannot enter as x_it with common β, and including
    them is redundant rather than wrong.

  So the only open choice is what goes in **X**, and it should be minimal. Xu notes the field
  has moved that way: "Increasingly, researchers rely solely on lagged outcomes as covariates
  (Ben-Michael, Feller, and Rothstein 2021)."

  **Primary: X = {Dec_k × Post}.** The only covariate that is simultaneously (a) time-varying,
  so not absorbed by λ_i, (b) pre-determined in its cross-sectional part (Dec measured
  2015–17), (c) not affected by the treatment, and (d) substantively needed — within a sector
  shares sum to one, so the Chinese share that is available to be reallocated sets the size of
  the opportunity every exporter in that sector faces. Decided in A2.

  **Matching variables in full, then: the eleven pre-2018 values of `lnS`, used in their
  entirety, plus Dec_k × Post.** Nothing else.

  **Robustness: baseline capability with time-varying coefficients.** Xu's Remark 3 permits
  observed time-invariant covariates as z_i′θ_t, which is expressible in our implementation as
  z_i interacted with year dummies. Adding baseline MVA/GDP, log MVA per capita, ECI and
  sector export share this way is legitimate — they are pre-determined — and it tests
  something we have asked since §3l: whether the latent factors are in fact capturing
  industrial capability. If the estimate does not move, they were. Note this is a *restricted*
  version of λ_i′f_t with observed rather than latent loadings, so partial redundancy is
  expected.

  **Explicitly excluded as bad controls.** UNIDO turns out to have **complete annual coverage
  2007–2024 for 207 countries** on MVA/GDP and MVA per capita, so time-varying capability
  measures are available. They are still excluded from X, because post-2018 MVA/GDP is
  plausibly *affected by the treatment* — if industrial policy raises manufacturing value
  added, conditioning on it absorbs the effect — and Xu's Assumption 2 requires
  ε_it ⊥ x_js for all j, s. Same reasoning excludes total exports, non-US exports and sector
  output. Availability is not a reason to include.

  **Consequence for B2 (the balance table).** You cannot show balance on covariates that are
  not in the model. Under the merge the balance table becomes: treated vs imputed
  counterfactual vs donor-pool mean on the **pre-period outcomes**, plus a comparison of
  **estimated factor loadings** (Xu's Fig. 3b), which is where the time-invariant
  characteristics now live.
- `[~]` **A4. Common support.** Track 1: convex hull. Track 2: **factor-loading overlap**
  (Xu Fig. 3b) — the GSC analogue, and the check that distinguishes interpolation from
  extrapolation. Neither has been done for a chosen design. §4a–§4c did this at
  country level and found canonical SC infeasible for the countries that matter. Redo for
  whichever unit Track 1 picks.

## Part B — Estimation, canonical Abadie

- `[ ]` **B1. Solve W\* given V, and V by minimising pre-period RMSPE.** *We have never done
  the nested V optimisation.* Everything in RESULTS.md stacks lagged outcomes and covariates
  with implicit equal weighting — i.e. a fixed diagonal V. ADH's `synth` optimises V. This is
  the largest single gap against the chapter.
- `[ ]` **B2. Balance table** (chapter Table 11.1): treated vs synthetic vs donor-pool mean,
  for every matching variable. Not produced anywhere in this project.
- `[~]` **B3. Weights table**: who is in the synthetic unit and with what weight. Done ad hoc
  in §3af and §8d; never as a clean table for a chosen design.
- `[~]` **B4. Paths figure** (Figure 11.3): treated vs synthetic. Done as §5g/`sc10_paths.png`
  for Track 2.
- `[~]` **B5. Gap figure** (Figure 11.4): treated minus synthetic. Done as §5g row 2, but as
  percentile envelopes rather than a single gap line.

## Part B2 — Estimation, generalized SC (Xu 2017)

No R in this container, so `gsynth` is unavailable and GSC is implemented directly.

- `[x]` **B2.0. Validate the implementation on Xu's own Monte Carlo DGP** — done, §9 and
  §9b. All of Xu's stated properties now hold: bias ≤ 0.036 with SD near the 1/√5 efficiency
  floor; CV picks r = 2 in 62–86%; his three estimator comparisons hold; and after fixing a
  contaminated donor pool in the leave-one-control-out step, bootstrap coverage is 92.5% and
  96.2% against nominal 95% with se/sd of 0.97 and 1.02. The r = 0 branch of the CV is fixed.
- `[~]` **B2.0b. Validate at T₀ = 11, our actual pre-period length** — §9c. Bias passes
  (≤ 0.016) and coverage passes at the first cell (93.3%, se/sd 0.93), but two warnings
  land: the SE contains a common component of ≈ 0.47 that does NOT average out over treated
  units, so it will plateau rather than fall like 1/√N_tr; and the CV picks the true r only
  37–50% of the time, erring **downward**, which biases GSC toward the two-way-FE answer.
  Remaining coverage cells still running. Original text:** — coverage and the CV's
  r distribution, with N_co scaled toward our 2,497. The binding constraint above.
- `[ ]` **B2.1. Step 1** — fit the IFE model on controls only: minimise over β, F, Λ_co
  subject to F'F/T = I_r and Λ'Λ diagonal.
- `[ ]` **B2.2. Step 2** — estimate each treated unit's factor loadings by projecting its
  pre-treatment outcomes onto the estimated factor space.
- `[ ]` **B2.3. Step 3** — impute Y_it(0) and form ATT_t.
- `[ ]` **B2.4. Do NOT choose r by CV alone** — per §9c it selects the true r only 37–50% of
  the time at T₀ = 11 and errs downward. Report the ATT across a **sweep of r ∈ {0,1,2,3,4}**
  plus the CV curve, and treat a low CV-selected r as weak evidence of no factor structure
  rather than good evidence of it.
- `[x]` **B2.5. Inference — decided in §9b.** Use the **parametric bootstrap (Algorithm 2)
  with country blocks**, `bootstrap(..., blocks=country_id, max_loo=...)`, validated at
  se/sd ≈ 1.0. Not the nonparametric one: resampling controls only runs 12% light (it omits
  the treated units' own ε) and resampling treated units targets the population rather than
  the sample ATT, which is not Xu's estimand or ours.
- `[ ]` **B2.6. Diagnostics Xu requires** — plot raw treated and control paths with imputed
  counterfactuals; plot treated vs control factor loadings and check overlap.
- `[ ]` **B2.7. Heterogeneity** — GSC returns per-unit effects in one run, so report ATT by
  country and by sector without re-estimating.

## Part C — Inference: ADH randomisation, exact p-values

- `[~]` **C1. Placebo on every donor.** §5b used 500 randomly drawn placebos of 3,658, not all.
- `[x]` **C2. Pre- and post-treatment RMSPE per placebo.** `sc10_estimate.py`.
- `[x]` **C3. Post/pre RMSPE ratio.** Same.
- `[!]` **C4. Rank the treated unit → exact p.** Done per treated unit, but with 500 treated
  units I then summarised 500 exact p-values (median, share below 0.10, Fisher). That
  aggregation has no randomisation-inference justification, and the Fisher combination was
  invalid outright — the p-values share a donor pool and one reference distribution.
  **Track 2 needs an aggregate randomisation test instead**: the group statistic, ranked
  against placebo groups of equal size, drawn unit-wise *and* country-block-wise. §8d showed
  the block version widens the null by 2.6×.
- `[ ]` **C5. Figure 11.5** — all placebo gap paths, treated in bold.
- `[ ]` **C6. Figure 11.6** — same, dropping placebos with pre-RMSPE > 2× the treated unit's.
- `[ ]` **C7. Figure 11.7** — histogram of post/pre ratios, treated marked.
- `[!]` **C8. The 2× filter swept.** §5c did this in ADH's direction over 1×/2×/3×/5×/none and
  found the ratio statistic behaves *backwards* here: tightening the filter raises p, because
  well-fitting placebos have tiny denominators and their ratios explode. Deviation recorded.

## Part D — Falsification (chapter's own exercises)

- `[ ]` **D1. Placebo-in-time.** ADH 2015: move the treatment date earlier, truncate the
  sample before the true date, and check the "effect" disappears. The chapter sets this as an
  exercise. Never run in this project.
- `[ ]` **D2. Leave-one-donor-out.** Never run.
- `[~]` **D3. Specification robustness over lag choices** (Ferman–Pinto–Possebom). §8d swept
  donor-pool size K and found τ ranged −0.05 to +0.24 non-monotonically while pre-fit
  improved monotonically. Never swept the *lag set*.

## Part E — Second-best estimators (chapter §11.2–11.5)

- `[~]` **E1. Ferman–Pinto demeaning.** §5f, at country level.
- `[~]` **E2. Ben-Michael–Feller–Rothstein augmented SC.** Used throughout §5, but never with
  the chapter's weight-distribution figure (11.15) or the perfect-fit/imperfect-fit contrast
  (11.14 vs 11.16).
- `[ ]` **E3. Matrix completion with nuclear-norm regularisation** (Athey et al.). Never tried.
  Relevant because it handles many treated units and staggered timing natively — i.e. it is
  the chapter's answer to our Track 2 problem.
- `[!]` **E4. Synthetic DiD** (Arkhangelsky et al.). §8c/§8d. Unit weights were
  underdetermined (2,497 donors, 11 pre-periods) so ω went uniform and τ collapsed to the
  λ-weighted DiD; trimming made τ unstable in K. Time weights did work.
- `[ ]` **E5. Side-by-side comparison** — DiD, two-way FE, classic SC, augmented SC, SDiD,
  matrix completion, **GSC**, and saturated α_it + α_kt, on one figure. GSC, matrix completion
  and SDiD are one family (many treated units, parallel trends relaxed); Xu's paper is the
  member this project was missing, and his own Monte Carlos benchmark GSC against DiD, IFE
  and SC, which is exactly this comparison. The chapter recommends exactly this, and Arkhangelsky et al. do
  it. Never assembled.

## Part F — Reporting

- `[ ]` **F1.** Balance table, weights table, paths figure, gap figure, placebo figures,
  exact p, falsification table, estimator comparison.
- `[ ]` **F2.** Stata `.do` implementation (`synth`, `allsynth`, `sdid`; check whether a
  `gsynth` port exists, otherwise export the GSC estimates and ship the Python alongside) on an exported
  analysis dataset, so the final numbers are reproducible in the user's own toolchain.

## Running conclusions

*(filled in as we go)*
