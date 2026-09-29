#!/bin/bash
: "${COMP_BASE_ROOT:?set COMP_BASE_ROOT to the competitor-run workspace}"
S=$COMP_BASE_ROOT
W=$S/env/wid3r/bin/python; PV=$S/env/panovggt/bin/python; H=$(cd "$(dirname "$0")" && pwd); R=$S/runs
while ! grep -q CHAIN1_DONE $R/logs/chain1.status 2>/dev/null; do sleep 20; done
cd $S/competitors/PanoVGGT
$PV $H/predict_panovggt.py --list $R/lists/v11_blk_erp.json --out $R/panovggt/v11_blk_erp > $R/logs/panovggt_v11_blk_erp.log 2>&1
cd $S/competitors/Wid3R
for T in v11_blk_erp v11_blk_mixed; do
  $W $H/predict_wid3r.py --list $R/lists/$T.json --out $R/wid3r/${T}_told --mode told --save-points > $R/logs/wid3r_${T}.log 2>&1
done
$W $H/predict_wid3r.py --list $R/lists/v11_blk_mixed.json --out $R/wid3r/v11_blk_mixed_pinhole --mode pinhole --save-points > $R/logs/wid3r_v11_blk_mixed_pinhole.log 2>&1
echo "CHAIN2_DONE $(date)" >> $R/logs/chain2.status
