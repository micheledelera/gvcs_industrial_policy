#!/bin/bash
for M in n_policies frac_policies share_n_policies share_frac_policies; do
  echo "################################################################"
  echo "### MEASURE: $M"
  echo "################################################################"
  python3 fit_event_decile.py "$M" 2>&1 | grep -v Warning | grep -v "warn("
done
