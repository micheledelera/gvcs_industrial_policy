# What the two panels assume

*A reader's guide to `fig_main_clean.png`. The left panel shows the average log US imports of
1,083 treated country-sectors against the path the estimator says they would have followed
without treatment. The right panel shows the difference between those two lines, year by year,
with a 95% band. Everything below is what has to be true for those two lines to mean what they
appear to mean.*

---

## 1. What a dot on the graph is

The unit of observation is a **country-sector**: one developing-country exporter paired with one
ISIC 4-digit manufacturing sector, observed annually from 2007 to 2024. India–pharmaceuticals is
one unit; Vietnam–footwear is another. The outcome is the log of US imports from that
country-sector, measured in thousands of dollars, which is why the left panel sits at around 8.3
to 9.4 rather than near zero — 8.35 is about $4.2m, 9.41 about $12.3m.

Both lines are **unweighted averages across units**. A country-sector shipping $200,000 of goods
counts exactly as much as one shipping $280m. This is the single most consequential framing
choice in the figure, and §10d and the value-weighted calculation both say so: weighting units
by their trade value instead, the same estimate falls from +0.234 to **+0.074**. The graph is
therefore a statement about the *typical targeted country-sector*, not about the dollars. If the
question is "did industrial policy move trade," the graph overstates it by roughly threefold. If
the question is "did targeted activities grow faster," the graph is the right object.

## 2. Who counts as treated, and why persistence

A unit is treated if the Global Trade Alert inventory records a promotional intervention against
it in **at least six of the nine years 2009–2017**.

The intuition behind the six-year threshold is that we are not trying to catch policy *events*.
A country-sector that appears in GTA once may have received a one-off subsidy that expired, or
a tariff line that was adjusted and forgotten. A country-sector that appears in six of nine
consecutive pre-event years is being *systematically* supported — it sits inside a standing
industrial-policy regime. The research question is whether having such a regime in place
positioned a country to catch trade displaced from China, so the regime, not the announcement,
is the thing to measure.

Two assumptions ride on this:

- **Persistence proxies a regime.** We cannot observe policy intent or budget, only the
  frequency of recorded interventions. Six-of-nine is a proxy. §10b swept the threshold from 5
  to 9 and found it sits inside a smooth distribution — there is no natural break, so the
  threshold is a judgement call rather than a discovery, and the sweep is part of the result
  rather than a robustness check.
- **Treatment is pre-determined and fixed.** Status is assigned entirely from 2009–2017 and
  never updated. Nothing a country did after the tariffs can change whether it counts as
  treated. This is what buys us the exogeneity we need: policy cannot be responding to the
  event we are studying, because we measured it before the event happened.

## 3. Who counts as a control, and the strictness problem

A control must have **zero** recorded GTA interventions in **every** year from 2009 to 2024 —
not few, not declining, zero, for sixteen straight years.

This strictness is deliberate and it buys a clean contrast. But it carries the most important
measurement assumption in the whole design: **that no recorded intervention means no
intervention**. GTA is built from official announcements and press reporting. Absence of a
record is not proof of absence of policy; it may equally indicate a state that does not announce
things, or a sector too obscure to be reported on. The control group is therefore selected, in a
correlated way, for country-sectors that are small, peripheral, or in countries with limited
administrative capacity to publicise what they do. We cannot rule this out, and it plausibly
biases the estimate upward if untargeted-but-actually-supported units are sitting in the donor
pool.

There is a second, sharper consequence. Because "strictly clean" is such a demanding bar,
**98.7% of the controls sit in countries that contribute no treated units at all** — 2,194 of
2,223. The nine largest treated countries contribute **exactly zero** controls: Poland,
India, Brazil, Hungary, Romania, Bulgaria, Indonesia, Croatia and Argentina each have 75 to 119
treated units and not one clean one.

So the comparison the figure actually runs is **between countries, not within them**. We are not
comparing India's targeted sectors to India's untargeted sectors. We are comparing India's
targeted sectors to sectors in 119 countries that never targeted anything. Anything that
distinguishes India from those countries, and that moved after 2018, is a threat to the
interpretation. This is also why the standard errors are the contested part of the figure —
see §5.

## 4. Where the counterfactual line comes from

The dashed line is not a group of comparison countries averaged together. It is a model
prediction, and the model is Xu's (2017) generalized synthetic control.

**The intuition.** Suppose the world of untreated trade is driven by a small number of global
forces — a commodity cycle, a shipping-cost shock, a swing in US consumer demand — and every
country-sector responds to each force with its own sensitivity. Bangladeshi knitwear is very
exposed to US retail demand and barely to metals prices; Chilean copper products are the
reverse. Formally, fitted on controls only:

```
y_it  =  x_it'β  +  α_i  +  ξ_t  +  λ_i'f_t  +  ε_it
```

`α_i` is the unit's own average level, `ξ_t` is the year common to everyone, `f_t` are the
global forces, `λ_i` are that unit's sensitivities to them, and `ε_it` is idiosyncratic noise.
We use **two** forces (r = 2).

Then, for each treated unit, we ask: *what set of sensitivities would have reproduced this
unit's actual path from 2007 to 2017?* We fit those sensitivities on the pre-period only, and
roll them forward. The counterfactual is what that unit would have done if it had gone on
responding to the same two global forces with the same sensitivities it displayed before 2018.

This is why the two lines track each other so closely before 2018 — within 0.025 log points in
every pre-window — and it is also why that closeness is **not** evidence of anything. The
estimator is fitted to make it so. More on this in §6.

**What the model assumes.**

1. **The untreated world really is low-dimensional.** If the true dynamics need five global
   forces and we allow two, the unexplained remainder shows up in the post-period as an
   "effect". Cross-validation picks r = 1, with r = 2 within 2%; §9c found the cross-validation
   errs downward, so r = 2 is the conservative choice, and r was swept from 0 to 4.
2. **Strict exogeneity.** The idiosyncratic noise must be unrelated to treatment, to the global
   forces, to the sensitivities and to the covariate — at all leads and lags. In plain terms:
   there must be no shock that hits targeted country-sectors specifically *and* is correlated
   with having had industrial policy. The most obvious candidate violation in this window is
   COVID: if the sectors governments chose to target are systematically the sectors COVID
   disrupted, the post-2018 gap is partly a pandemic artefact. The gap is +0.05 in 2018 and
   +0.10 in 2019, then +0.17, +0.32, +0.40 through 2020–22, so the bulk of it does accumulate
   in the pandemic window.
3. **Sensitivities are stable across 2018.** A unit's exposure to the global forces must not
   change at the event for reasons other than treatment. This is the factor-model analogue of
   parallel trends, and it is a genuinely weaker assumption: we do not need treated and control
   *levels* to move together, only their *exposures* to stay put. That is the main reason for
   preferring this estimator to difference-in-differences here.
4. **No anticipation.** Nothing in 2015–17 is already a response to 2018.
5. **Support.** Each treated unit's sensitivities must lie inside the cloud of control
   sensitivities, or we are extrapolating rather than interpolating. A4 found 97% containment at
   r = 2, with the median treated unit at the 49.5th percentile of the control distribution —
   good, and better than the canonical convex-hull requirement usually manages.
6. **No interference between units.** One country-sector's treatment must not affect another's
   outcome. **This one is almost certainly violated, and in a way the design cannot fix.**
   Trade reallocation is close to zero-sum within a sector: if Vietnam wins a footwear order,
   somebody loses it. If the losers are in the donor pool, the counterfactual is being pushed
   down by the very treatment we are measuring, and the estimate is the *difference* between
   winners and losers rather than the effect on winners. It should be read as an upper bound
   for that reason. The left panel makes the size of the concern visible: the counterfactual
   rises by +0.43 log points from pre to post while the treated rise by +0.66, so 39% of the
   gap is the counterfactual growing more slowly rather than the treated growing faster.

**The one covariate.** `Dec_k × Post` is China's 2015–17 share of US imports in the sector,
switched on from 2018. It lets sectors that were more Chinese move differently after the tariffs
*for everyone, treated and control alike*. Without it, the mechanical reallocation out of China
— which is the event, not the policy — would be credited to industrial policy. With it, the
estimate is what targeted units did *on top of* whatever their sector's China exposure was
already delivering.

## 5. Where the band comes from, and why it is the contested part

The shaded band is a 95% interval from Xu's parametric bootstrap, **blocked at ISIC 4-digit
sector**.

**The intuition.** We would like to know how far off the counterfactual line could plausibly be.
The trouble is that the errors we care about — the treated units' prediction errors — are
exactly the thing we cannot observe, because observing them would mean observing the
counterfactual. So we borrow them. Take each control unit in turn, pretend it is treated,
predict it by the same procedure, and record how wrong the prediction was. Do that across the
donor pool and you have a library of plausible prediction errors. Then build many simulated
datasets by re-drawing errors from that library, re-run the whole estimator on each, and look at
how much the answer moves.

For this to work, the errors we draw have to be exchangeable with the ones we cannot see. Xu's
version of that requirement is **cross-sectional independence**: errors independent across
units. Note what comes free and what does not. Because the bootstrap resamples *whole 18-year
series* per unit, serial correlation within a unit is handled with no assumption at all —
whatever persistence a country-sector's shocks have is carried along intact. It is only the
across-unit term that is assumed away.

And that assumption is false here. There are 1,083 treated units sitting in 25 countries. A
currency movement, a port closure, a change of government, a sanctions regime hits all of a
country's sectors at once. Measured directly, the within-country residual correlation is
**+0.033**.

The remedy is to resample in lumps rather than one unit at a time, so that whatever correlation
exists inside a lump survives into the simulated data. And this is where a genuine judgement
call enters, because **neither of the two sources this design is built on tells us what the lump
should be**:

- **Cunningham's Mixtape chapter reports no standard errors at all.** Its inference is
  randomisation-based throughout: reassign treatment to control units, recompute the statistic,
  and see where the real one ranks. This follows Abadie, Diamond and Hainmueller's explicit
  position that large-sample standard errors are inappropriate when there are one or a handful
  of treated units, because there is no asymptotics to appeal to. The chapter's answer to "at
  what level do I cluster" is: you do not, you permute.
- **Xu reports standard errors** from the bootstrap above, and requires cross-sectional
  independence to justify them. He offers no cluster-robust variant.

So blocking is an adaptation, imported from the panel difference-in-differences literature, and
the level is ours to defend. It matters enormously:

| blocking level | clusters | ATT | se | p | 95% CI |
|---|---|---|---|---|---|
| ISIC 4-digit sector | 123 | +0.234 | 0.106 | **0.029** | [+3%, +55%] |
| ISIC 2-digit division | 24 | +0.234 | 0.120 | 0.064 | [−0%, +60%] |
| country | 25 | +0.234 | 0.239 | 0.338 | [−21%, +102%] |

The band drawn in the right panel is the first row. **What it assumes** is that after the model
has removed the two global forces, two country-sectors in *different countries but the same
sector* may still move together, while two in the *same country but different sectors* do not.
In substance: that the shocks decoupling and the pandemic delivered were sector-shaped rather
than country-shaped.

The evidence points the other way. Compare the second and third rows: 24 sector divisions and 25
countries — essentially the same number of clusters, so the same degrees-of-freedom penalty —
and the standard error differs by a factor of two. Coarsening along the sector dimension from
1,083 clusters to 24 barely moves anything; coarsening along the country dimension to 25 doubles
the interval. The dependence in this panel is country-shaped. Sector blocking is the more
permissive choice, and a reader who believes decoupling hit countries rather than sectors will
insist on the third row, where the effect is indistinguishable from zero.

Two further properties of the band worth knowing. It was validated by simulation before being
used: replicating Xu's own Monte Carlo at this panel's dimensions gave 93–96% coverage with a
standard-error-to-true-dispersion ratio of 0.93–0.98, so the machinery works when its
assumptions hold. And with only 25 countries — or even 24 divisions — cluster-robust inference
is known to over-reject somewhat, so a wild cluster bootstrap or a randomisation test is the
better instrument at that end, which is precisely why the randomisation test is worth running.

## 6. Two things the figure cannot be used to check

**The flat pre-period is not a test.** It is natural to read the two lines lying on top of each
other before 2018 as evidence that the design works. It is not. The estimator fits each treated
unit's sensitivities by regressing its 2007–2017 path on the global forces *with an intercept*,
which forces that unit's average pre-period gap to be exactly zero — not approximately, not on
average across units, but exactly, unit by unit. The pre-period agreement in the left panel is
an arithmetic identity, not a finding. Only the *year-to-year shape* of the pre-period gap in
the right panel carries information, and a real test needs a placebo in time: re-fit pretending
the event was 2015, and see whether a gap appears where none should.

**Statistical significance is not the same question as economic significance.** The +0.234 in
the right panel is an unweighted average over country-sectors, and it is produced almost
entirely by the smallest ones. Split by size, the bottom quintile (mean $0.2m of pre-period
exports) is +0.99 and the top quintile (mean $282m, holding essentially all of the trade value)
is +0.10. Split by country, the outcomes run from −0.66 for Russia to +1.02 for Pakistan, and
the sign is not stable among the countries with enough units to be measured at all: of the 11
countries with 30 or more treated units, 7 are positive. Whatever the band says, the figure is
not evidence of a uniform effect.
