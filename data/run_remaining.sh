#!/bin/bash
set -e
cd /home/user/gvcs_industrial_policy/data
echo "=== starting n_sub $(date) ==="
python3 fit_ddd_only.py n_sub > estimate_n_sub.log 2>&1
echo "=== n_sub done, starting frac_policies $(date) ==="
python3 fit_ddd_only.py frac_policies > estimate_frac_policies.log 2>&1
echo "=== frac_policies done $(date) ==="
