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

Every step below is run on **two tracks**:

- **Track 1 — canonical Abadie.** One treated unit, ADH-faithful. Proposed: **Vietnam** at
  country level, or one Vietnamese sector. This is the chapter as written, and the design the
  method was built for. §5d/§5e already point at Vietnam as the case of interest.
- **Track 2 — generalized SC (Xu 2017).** Our actual estimand: many treated units. Xu's GSC
  is not an optional extra here — it is the estimator the chapter's method is missing, and it
  addresses our failures one for one (see below).

### Why Xu (2017) is the right Track 2 estimator

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

- `[ ]` **A0. Question, treatment, date, outcome.** What is the intervention, when does it
  bite, what is the outcome. Decide whether the outcome is US imports in levels, logs, or a
  share; ADH used per-capita sales (a rate, not a level).
- `[ ]` **A1. Treated unit(s).** Track 1: which unit, and why it is a defensible single case.
- `[ ]` **A2. Donor pool, with exclusions.** *Track 1 only — GSC uses every control unit.* ADH *dropped* states with their own tobacco
  programmes. Our analogue is untested: should we drop countries with their own large
  post-2018 shocks, their own China exposure, or their own policy regimes? We have never
  applied any exclusion rule of this kind.
- `[ ]` **A3. Matching variables.** Covariates X plus lagged outcomes. ADH use both, and
  stress lagged outcomes "to soak up the heterogeneity". Ferman–Pinto–Possebom warn that the
  choice of lags is a specification-search margin, so the set must be fixed here and swept
  later, not chosen on results.
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

- `[ ]` **B2.0. Validate the implementation on Xu's own Monte Carlo DGP** (his eq. 3:
  N_tr = 5, N_co = 45, T = 30, T₀ = 20, w = 0.8, two factors, true ATT = 5 at T₀+5). Check we
  recover his Table 1 bias, SD, RMSE and 95% coverage before touching our data.
- `[ ]` **B2.1. Step 1** — fit the IFE model on controls only: minimise over β, F, Λ_co
  subject to F'F/T = I_r and Λ'Λ diagonal.
- `[ ]` **B2.2. Step 2** — estimate each treated unit's factor loadings by projecting its
  pre-treatment outcomes onto the estimated factor space.
- `[ ]` **B2.3. Step 3** — impute Y_it(0) and form ATT_t.
- `[ ]` **B2.4. Choose r by leave-one-out cross-validation** on treated pre-periods
  (Xu Algorithm 1). Report the CV curve, not just the winner.
- `[ ]` **B2.5. Inference** — parametric bootstrap (Xu Algorithm 2), **blocked at country**
  rather than unit, per caveat 4.
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
