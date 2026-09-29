#!/bin/bash
: "${COMP_BASE_ROOT:?set COMP_BASE_ROOT to the competitor-run workspace}"
S=$COMP_BASE_ROOT
W=$S/env/wid3r/bin/python; PV=$S/env/panovggt/bin/python; H=$(cd "$(dirname "$0")" && pwd); R=$S/runs
mkdir -p $R/wid3r $R/panovggt $R/logs
cd $S/competitors/Wid3R
$W $H/predict_wid3r.py --cases-root $S/inputs/calib_cases --out $R/wid3r/calib_told --mode told > $R/logs/wid3r_calib_told.log 2>&1
$W $H/predict_wid3r.py --cases-root $S/inputs/calib_cases --out $R/wid3r/calib_pinhole --mode pinhole > $R/logs/wid3r_calib_pinhole.log 2>&1
$W $H/predict_wid3r.py --cases-root $S/inputs/heterogeneous_2d3ds --out $R/wid3r/het2d3ds_told --mode told > $R/logs/wid3r_het2d3ds_told.log 2>&1
$W $H/predict_wid3r.py --cases-root $S/inputs/heterogeneous_2d3ds --out $R/wid3r/het2d3ds_pinhole --mode pinhole > $R/logs/wid3r_het2d3ds_pinhole.log 2>&1
echo "WID3R_STAGE1_DONE $(date)" >> $R/logs/chain1.status
cd $S/competitors/PanoVGGT
$PV $H/predict_panovggt.py --list $R/lists/single_panoramas.json --out $R/panovggt/single_panoramas > $R/logs/panovggt_single_panoramas.log 2>&1
$PV $H/predict_panovggt.py --list $R/lists/blk_erp.json --out $R/panovggt/blk_erp > $R/logs/panovggt_blk_erp.log 2>&1
$PV $H/predict_panovggt.py --list $R/lists/panorama_tuples.json --out $R/panovggt/panorama_tuples > $R/logs/panovggt_panorama_tuples.log 2>&1
echo "PANOVGGT_DONE $(date)" >> $R/logs/chain1.status
cd $S/competitors/Wid3R
$W $H/predict_wid3r.py --list $R/lists/panorama_tuples.json --out $R/wid3r/panorama_tuples_told --mode told > $R/logs/wid3r_panorama_tuples.log 2>&1
$W $H/predict_wid3r.py --list $R/lists/single_panoramas.json --out $R/wid3r/single_panoramas_told --mode told --save-points > $R/logs/wid3r_single_panoramas.log 2>&1
for T in blk_erp blk_mixed blk_pinhole blk_fisheye; do
  $W $H/predict_wid3r.py --list $R/lists/$T.json --out $R/wid3r/${T}_told --mode told --save-points > $R/logs/wid3r_${T}.log 2>&1
done
$W $H/predict_wid3r.py --list $R/lists/blk_mixed.json --out $R/wid3r/blk_mixed_pinhole --mode pinhole --save-points > $R/logs/wid3r_blk_mixed_pinhole.log 2>&1
echo "CHAIN1_DONE $(date)" >> $R/logs/chain1.status
