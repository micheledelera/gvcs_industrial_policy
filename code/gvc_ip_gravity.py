"""
GVCs, US-China decoupling, and industrial policy
Main gravity specification (PPML), plus the FE ladder and key robustness runs.

Input : trade_df_isic.dta  (Zenodo 10.5281/zenodo.22282107, mirrored at
        github.com/micheledelera/gvcs_industrial_policy releases/zenodo-22282107)
Needs : pandas, pyreadstat, pyfixest>=0.60

HEADLINE RESULT (full sample, 229 destinations, N = 16,886,001):
    DDD_dev = +0.0524 (SE 0.0168, p = 0.0018)   lag 3
    DDD_dev = +0.0373 (SE 0.0161, p = 0.0204)   lag 0

The effect is on POLICY TARGETING (share_frac_policies = the share of a
country's industrial policy directed at sector s), NOT policy volume.
Substituting the level measure (frac_policies) kills it: -0.0081 (p=0.56).

Run:  python gvc_ip_gravity.py [--lag 3] [--measure share_frac_policies]
                               [--restrict-destinations] [--ladder]
"""

import argparse
import gc

import numpy as np
import pandas as pd
import pyfixest as pf

CHINA = 156

# IMF WEO advanced economies. NB COMTRADE/BACI codes, which differ from
# ISO-3166 numeric for three countries -- verified against codes present in the
# data: USA=842 (not 840), Switzerland=757 (not 756), Norway=579 (not 578).
# Puerto Rico (630) is absent, folded into the US in trade statistics.
ADVANCED = {36, 40, 56, 124, 196, 203, 208, 233, 246, 251, 276, 300, 344, 352,
            372, 376, 380, 392, 410, 428, 440, 442, 446, 470, 490, 528, 554,
            579, 620, 674, 702, 703, 705, 724, 752, 757, 826, 842}

# 22 major advanced final-demand markets, excluding entrepots (HK, Singapore,
# Macao, Taiwan). Optional: gives a "US vs other rich markets" counterfactual,
# ~4x faster, and a cleaner comparison group than "US vs everywhere".
ADVANCED_DESTINATIONS = {842, 276, 251, 380, 826, 528, 724, 56, 392, 124, 410,
                         36, 752, 40, 208, 246, 372, 620, 300, 579, 757, 554}


def load(path="trade_df_isic.dta"):
    """Load and clean: drop unclassified sectors and the ' old' concordance vintage."""
    df = pd.read_stata(path) if path.endswith(".dta") else pd.read_pickle(path)
    df = df[(df["ISIC4"] != "") & (~df["ISIC4"].str.contains(" old"))].copy()
    df["ISIC4c"] = df["ISIC4"].astype("category")
    df["i_int"] = df["i"].astype("int32")
    df["t_int"] = df["t"].astype("int32")
    df["imports"] = df["imports"].astype("float32")
    return df


def build(df, measure="share_frac_policies", lag=3, restrict_destinations=False):
    """Construct treatment, groups, lagged policy, interactions and FE keys."""
    df[measure] = pd.to_numeric(df[measure], errors="coerce").fillna(0).astype("float32")

    # Policy is constant within (i, sector, t): build the lag on the collapsed
    # country-sector-year panel so gaps in bilateral trade don't break coverage.
    lookup = (df[["i_int", "ISIC4c", "t_int", measure]]
              .drop_duplicates(subset=["i_int", "ISIC4c", "t_int"]))
    lookup = lookup.assign(t_int=lookup["t_int"] + lag).rename(columns={measure: "IP_lag"})

    if restrict_destinations:
        df = df[df["j"].astype("int32").isin(ADVANCED_DESTINATIONS)]
    df = df.merge(lookup, on=["i_int", "ISIC4c", "t_int"], how="left")
    df = df[df["IP_lag"].notna()].copy()

    # Standardise: coefficients read as "per 1 SD of policy targeting".
    ip = (df["IP_lag"] / df["IP_lag"].std()).astype("float32")

    # Three exporter groups. China stays in the sample with its own terms rather
    # than being dropped -- it keeps disciplining alpha_jst and keeps sector-year
    # totals real, while not contaminating DDD_dev. (Dropped entirely, China's
    # 32% share of the regressor mass drove DDD_dev spuriously negative, since
    # the decoupling measure is built from China's own collapse.)
    china = (df["i_int"] == CHINA).astype("float32")
    adv = df["Advanced_i"].astype("float32") if "Advanced_i" in df else \
        df["i_int"].isin(ADVANCED).astype("float32")
    dev = ((1 - adv) * (1 - china)).astype("float32")
    us = df["US_trade"].astype("float32")

    # Decoupling: author's `target` flag x post-2018 (first Section 301 tranche).
    dec = (df["target"].astype("float32") * (df["t_int"] >= 2018)).astype("float32")

    m = pd.DataFrame({
        "imports": df["imports"].values,
        # coefficient of interest: nonzero only for a developing ex-China exporter
        # selling to the US, in a decoupling sector, post-2018, that had allocated
        # policy to that sector `lag` years earlier
        "DDD_dev": (dec * ip * us * dev).values,
        "DDD_adv": (dec * ip * us * adv).values,
        # required lower-order terms (Olden & Moen 2022): without these the triple
        # interaction is contaminated by the general IP->US channel
        "IPxUS_dev": (ip * us * dev).values,
        "IPxUS_adv": (ip * us * adv).values,
        "IPxUS_chn": (ip * us * china).values,
        # China's own decoupling term and the advanced-vs-developing structural
        # gap; developing ex-China is the omitted category for both
        "Dec_US_chn": (dec * us * china).values,
        "Adv_Dec_US": (adv * dec * us).values,
        # alpha_ist / alpha_jst are the sectoral generalisation of the standard
        # multilateral resistance terms (they nest alpha_it / alpha_jt);
        # alpha_ij absorbs time-invariant bilateral trade costs
        "fe_ist": df.groupby(["i", "ISIC4c", "t"], observed=True).ngroup().astype("int32").values,
        "fe_jst": df.groupby(["j", "ISIC4c", "t"], observed=True).ngroup().astype("int32").values,
        "fe_ij": df.groupby(["i", "j"], observed=True).ngroup().astype("int32").values,
        # cluster at (i,s), the level at which policy is assigned and persists.
        # NB clustering at (i,s,t) is wrong: every US-bound observation is alone
        # in its own ist cell, so it provides no adjustment to the identifying
        # variation and understates SEs by roughly a factor of two.
        "cl_is": df.groupby(["i", "ISIC4c"], observed=True).ngroup().astype("int32").values,
    })
    del df
    gc.collect()
    return m


FORMULA = ("imports ~ DDD_dev + DDD_adv + IPxUS_dev + IPxUS_adv + IPxUS_chn "
           "+ Dec_US_chn + Adv_Dec_US | fe_ist + fe_jst + fe_ij")


def estimate(m, formula=FORMULA):
    fit = pf.fepois(formula, data=m, vcov={"CRV1": "cl_is"},
                    demeaner=pf.LsmrDemeaner(fixef_maxiter=2000),
                    lean=True, store_data=False, copy_data=False)
    t = fit.tidy()
    print(f"\nN = {fit._N:,}   cluster (i,s)\n{t.round(5).to_string()}")
    # DDD_dev is the DIFFERENTIAL between decoupling and non-decoupling sectors;
    # the level effect inside decoupling sectors is IPxUS_dev + DDD_dev.
    print(f"\nlevel effect in decoupling sectors (IPxUS_dev + DDD_dev): "
          f"{t.loc['IPxUS_dev','Estimate'] + t.loc['DDD_dev','Estimate']:+.4f}")
    return fit


def fe_ladder(df, measure="share_frac_policies", lag=3):
    """US-bound, developing ex-China. Shows the raw descriptive association is
    entirely BETWEEN countries: significant at L1, dead once alpha_i is added."""
    d = build_ladder_frame(df, measure, lag)
    for name, fes in [("L1 st", "fe_st"), ("L2 +i", "fe_st + fe_i"),
                      ("L3 +is", "fe_st + fe_is"), ("L4 +is+it", "fe_st + fe_is + fe_it")]:
        fit = pf.fepois(f"imports ~ IP_z + IPxDec | {fes}", data=d,
                        vcov={"CRV1": "fe_i"},
                        demeaner=pf.LsmrDemeaner(fixef_maxiter=2000))
        t = fit.tidy()
        print(f"{name:10s} IP={t.loc['IP_z','Estimate']:+.4f} (p={t.loc['IP_z','Pr(>|t|)']:.3f})"
              f"   IPxDec={t.loc['IPxDec','Estimate']:+.4f} (p={t.loc['IPxDec','Pr(>|t|)']:.3f})")


def build_ladder_frame(df, measure, lag):
    df[measure] = pd.to_numeric(df[measure], errors="coerce").fillna(0).astype("float32")
    lookup = (df[["i_int", "ISIC4c", "t_int", measure]]
              .drop_duplicates(subset=["i_int", "ISIC4c", "t_int"]))
    lookup = lookup.assign(t_int=lookup["t_int"] + lag).rename(columns={measure: "IP_lag"})
    d = df[df["US_trade"] == 1].merge(lookup, on=["i_int", "ISIC4c", "t_int"], how="left")
    adv = d["i_int"].isin(ADVANCED)
    d = d[(d["i_int"] != CHINA) & (~adv) & d["IP_lag"].notna()].copy()
    d["IP_z"] = (d["IP_lag"] / d["IP_lag"].std()).astype("float32")
    d["IPxDec"] = (d["IP_z"] * d["target"] * (d["t_int"] >= 2018)).astype("float32")
    d["fe_st"] = d.groupby(["ISIC4c", "t"], observed=True).ngroup().astype("int32")
    d["fe_i"] = d["i_int"]
    d["fe_is"] = d.groupby(["i", "ISIC4c"], observed=True).ngroup().astype("int32")
    d["fe_it"] = d.groupby(["i", "t"], observed=True).ngroup().astype("int32")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="trade_df_isic.dta")
    ap.add_argument("--lag", type=int, default=3)
    ap.add_argument("--measure", default="share_frac_policies",
                    help="share_frac_policies (targeting) is the headline; "
                         "frac_policies (volume) is the discriminating null")
    ap.add_argument("--restrict-destinations", action="store_true",
                    help="22 advanced markets instead of all 229; ~4x faster")
    ap.add_argument("--ladder", action="store_true", help="run the FE ladder instead")
    a = ap.parse_args()

    df = load(a.data)
    if a.ladder:
        fe_ladder(df, a.measure, a.lag)
    else:
        estimate(build(df, a.measure, a.lag, a.restrict_destinations))

# OUTSTANDING before submission
#   - Pre-trend test in THIS gravity spec (IP_z x target x year, base 2017).
#     The event study we ran was on the US-only DiD, not this specification.
#   - Exporter-level clustering as the conservative alternative to (i,s).
#   - GTA intervention-type breakdown, to identify WHICH kinds of targeting work
#     (the benchmark finds tax breaks positive, direct transfers negative --
#     they cancel in any pooled measure).
