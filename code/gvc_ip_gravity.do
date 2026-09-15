*==============================================================================
* GVCs, US-China decoupling, and industrial policy
* Main gravity specification (PPML), plus the FE ladder and key robustness runs
*
* Input : trade_df_isic.dta  (Zenodo 10.5281/zenodo.22282107, mirrored at
*         github.com/micheledelera/gvcs_industrial_policy releases/zenodo-22282107)
* Needs : ssc install ppmlhdfe ; ssc install reghdfe ; ssc install ftools
*
* HEADLINE RESULT (full sample, 229 destinations, N = 16,886,001):
*   DDD_dev = +0.0524 (SE 0.0168, p = 0.0018)   lag 3
*   DDD_dev = +0.0373 (SE 0.0161, p = 0.0204)   lag 0
*
* The effect is on POLICY TARGETING (share_frac_policies = the share of a
* country's industrial policy directed at sector s), NOT policy volume.
* Substituting the level measure (frac_policies) kills it: -0.0081 (p=0.56).
*==============================================================================

clear all
set more off
set matsize 11000

local DATA   "trade_df_isic.dta"
local LAG    3          // 0 and 3 both reported; 3 is the headline
local MEAS   "share_frac_policies"

use "`DATA'", clear

*------------------------------------------------------------------------------
* 1. Sample construction
*------------------------------------------------------------------------------
* Drop unclassified sectors and the duplicated " old" concordance vintage.
drop if ISIC4 == ""
drop if strpos(ISIC4, " old") > 0

* Advanced-economy exporters (IMF WEO). NB these are COMTRADE/BACI codes, which
* differ from ISO-3166 numeric for three countries -- verified against the codes
* actually present in the data:
*     USA = 842 (not 840) ; Switzerland = 757 (not 756) ; Norway = 579 (not 578)
* Puerto Rico (630) is absent, folded into the US in trade statistics.
gen byte Advanced_i = inlist(i, 36,40,56,124,196,203,208,233,246,251)      ///
                    | inlist(i, 276,300,344,352,372,376,380,392,410,428)   ///
                    | inlist(i, 440,442,446,470,490,528,554,579,620,674)   ///
                    | inlist(i, 702,703,705,724,752,757,826,842)

* Three exporter groups. China gets its own terms rather than being dropped:
* it stays in the sample so it keeps disciplining the importer-sector-year FE
* and keeps sector-year totals real, but it does not contaminate DDD_dev.
gen byte china_i = (i == 156)
gen byte dev_i   = (Advanced_i == 0) & (china_i == 0)     // developing ex-China
gen byte adv_i   = Advanced_i

*------------------------------------------------------------------------------
* 2. Treatment: decoupling
*------------------------------------------------------------------------------
* Primary: author's `target' flag (sector where the US is pulling away from
* China) interacted with post-2018 (first Section 301 tariff tranche).
gen byte post = (t >= 2018)
gen double dec = target * post

*------------------------------------------------------------------------------
* 3. Policy variable: TARGETING, lagged
*------------------------------------------------------------------------------
* share_frac_policies is constant within (i, ISIC4, t) -- build the lag on the
* collapsed country-sector-year panel, then merge back onto bilateral flows.
preserve
    keep i ISIC4 t `MEAS'
    duplicates drop
    egen long is_id = group(i ISIC4)
    tsset is_id t
    gen double IP_lag = L`LAG'.`MEAS'
    keep i ISIC4 t IP_lag
    tempfile lagged
    save `lagged', replace
restore
merge m:1 i ISIC4 t using `lagged', keep(master match) nogenerate
drop if missing(IP_lag)

* Standardise so coefficients read as "per 1 SD of policy targeting"
quietly summarize IP_lag
gen double IP_z = IP_lag / r(sd)

*------------------------------------------------------------------------------
* 4. Regressors
*------------------------------------------------------------------------------
* DDD_dev is the coefficient of interest: a five-way product, nonzero only for
* a developing ex-China exporter, selling to the US, in a decoupling sector,
* after 2018, that had allocated policy to that sector `LAG' years earlier.
gen double DDD_dev    = dec * IP_z * US_trade * dev_i
gen double DDD_adv    = dec * IP_z * US_trade * adv_i

* Required lower-order terms (Olden & Moen 2022): without these the triple
* interaction is contaminated by the general IP->US channel.
gen double IPxUS_dev  = IP_z * US_trade * dev_i
gen double IPxUS_adv  = IP_z * US_trade * adv_i
gen double IPxUS_chn  = IP_z * US_trade * china_i

* China's own decoupling term, and the advanced-vs-developing structural gap.
* Developing ex-China is the omitted category for both, so Dec_US_chn reads as
* "China vs developing ex-China" and Adv_Dec_US as "advanced vs developing".
gen double Dec_US_chn = dec * US_trade * china_i
gen double Adv_Dec_US = adv_i * dec * US_trade

*------------------------------------------------------------------------------
* 5. Fixed effects and clustering
*------------------------------------------------------------------------------
* alpha_ist and alpha_jst are the sectoral generalisation of the standard
* structural-gravity multilateral resistance terms (they nest alpha_it/alpha_jt).
* alpha_ij absorbs all time-invariant bilateral trade costs.
egen long fe_ist = group(i ISIC4 t)
egen long fe_jst = group(j ISIC4 t)
egen long fe_ij  = group(i j)

* Cluster at (i,s): the level at which policy is assigned and persists over time.
* NB clustering at (i,s,t) is WRONG here -- every US-bound observation is alone
* in its own ist cell, so it provides no adjustment to the identifying variation
* and understates SEs by roughly a factor of two.
egen long cl_is = group(i ISIC4)

*------------------------------------------------------------------------------
* 6. HEADLINE ESTIMATE
*------------------------------------------------------------------------------
ppmlhdfe imports DDD_dev DDD_adv IPxUS_dev IPxUS_adv IPxUS_chn               ///
                 Dec_US_chn Adv_Dec_US,                                       ///
         absorb(fe_ist fe_jst fe_ij) cluster(cl_is) d
estimates store main_lag`LAG'

* Effect in decoupling sectors is IPxUS_dev + DDD_dev; DDD_dev alone is the
* DIFFERENTIAL between decoupling and non-decoupling sectors.
lincom IPxUS_dev + DDD_dev

*------------------------------------------------------------------------------
* 7. Robustness / decomposition (each changes ONE thing)
*------------------------------------------------------------------------------
* (a) LEVEL instead of SHARE -> effect vanishes (-0.0081, p=0.56).
*     This is the key discriminating test: targeting matters, volume does not.
* (b) Continuous decoupling intensity instead of target x post -> survives
*     (+0.1008, p=0.027 on the 22-destination sample). Construction:
*         baseline_s      = mean China->USA mkt_share in sector s, 2015-2017
*         trailing_st     = 3yr trailing mean of the same
*         intensity_st    = target_s * max(0, (baseline_s - trailing_st)/baseline_s)
*     Bounded [0,1], ~0 before 2018 by construction, so it needs no pre/post split.
* (c) Contemporaneous instead of lag 3 -> weaker but still significant on the
*     full sample (+0.0373 vs +0.0524).

*------------------------------------------------------------------------------
* 8. FE LADDER (US-bound only, developing ex-China)
*     Shows the raw descriptive association is entirely BETWEEN countries:
*     significant at L1, dead once a country fixed effect is added.
*------------------------------------------------------------------------------
preserve
    keep if US_trade == 1 & dev_i == 1
    gen double IPxDec = IP_z * dec
    egen long fe_st = group(ISIC4 t)
    egen long fe_is = group(i ISIC4)
    egen long fe_it = group(i t)

    ppmlhdfe imports IP_z IPxDec, absorb(fe_st)                  cluster(i)
    ppmlhdfe imports IP_z IPxDec, absorb(fe_st i)                cluster(i)
    ppmlhdfe imports IP_z IPxDec, absorb(fe_st fe_is)            cluster(i)
    ppmlhdfe imports IP_z IPxDec, absorb(fe_st fe_is fe_it)      cluster(i)
restore

*------------------------------------------------------------------------------
* 9. OUTSTANDING before submission
*     - Pre-trend test in THIS gravity spec (IP_z x target x year, base 2017).
*       The event study we ran was on the US-only DiD, not this specification.
*     - Exporter-level clustering as the conservative alternative to (i,s).
*     - GTA intervention-type breakdown, to identify WHICH kinds of targeting
*       work (the benchmark finds tax breaks positive, direct transfers
*       negative -- they cancel in any pooled measure).
*------------------------------------------------------------------------------
