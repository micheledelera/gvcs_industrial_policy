import pandas as pd
import gc
import time

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

# Corrected advanced-economy code list: 757 (not 756) for Switzerland, 579 (not 578)
# for Norway -- both confirmed against actual codes present in the data (COMTRADE
# uses non-ISO-numeric codes for these two, same quirk as USA=842).
ADVANCED_CODES = {
    36: "Australia", 40: "Austria", 56: "Belgium", 124: "Canada", 196: "Cyprus",
    203: "Czech Republic", 208: "Denmark", 233: "Estonia", 246: "Finland",
    251: "France", 276: "Germany", 300: "Greece", 344: "Hong Kong SAR",
    352: "Iceland", 372: "Ireland", 376: "Israel", 380: "Italy", 392: "Japan",
    410: "Korea, Rep.", 428: "Latvia", 440: "Lithuania", 442: "Luxembourg",
    446: "Macao SAR", 470: "Malta", 528: "Netherlands", 554: "New Zealand",
    579: "Norway", 620: "Portugal", 630: "Puerto Rico", 674: "San Marino",
    702: "Singapore", 703: "Slovak Republic", 705: "Slovenia", 724: "Spain",
    752: "Sweden", 757: "Switzerland", 490: "Taiwan (Other Asia, nes)",
    826: "United Kingdom", 842: "United States",
}

agg = pd.read_pickle("agg_for_estimation.pkl")
log(f"loaded agg: {agg.shape}")

old_adv_share = agg['Advanced_i'].mean()
agg['Advanced_i'] = agg['i'].astype('int32').isin(ADVANCED_CODES.keys()).astype('int8')
new_adv_share = agg['Advanced_i'].mean()
log(f"Advanced_i share of obs: {old_adv_share*100:.2f}% (old, buggy) -> {new_adv_share*100:.2f}% (corrected)")

for ip_var in ['n_policies', 'n_sub', 'frac_policies']:
    d = ip_var
    agg[f'{d}_x_US_dev'] = (agg[ip_var] * agg['US_trade'] * (1 - agg['Advanced_i'])).astype('float32')
    agg[f'{d}_x_US_adv'] = (agg[ip_var] * agg['US_trade'] * agg['Advanced_i']).astype('float32')
    agg[f'DDD_{d}_dev'] = (agg['target'] * agg[ip_var] * agg['US_trade'] * (1 - agg['Advanced_i'])).astype('float32')
    agg[f'DDD_{d}_adv'] = (agg['target'] * agg[ip_var] * agg['US_trade'] * agg['Advanced_i']).astype('float32')

agg.to_pickle("agg_for_estimation.pkl")
log("saved corrected agg_for_estimation.pkl")
