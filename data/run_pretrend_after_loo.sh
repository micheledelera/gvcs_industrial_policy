#!/bin/sh
# wait on the sweep by PID -- pgrep on the script name would match this wrapper too
while kill -0 1615 2>/dev/null; do sleep 60; done
exec python fit_pretrend.py > pretrend.log 2>&1
