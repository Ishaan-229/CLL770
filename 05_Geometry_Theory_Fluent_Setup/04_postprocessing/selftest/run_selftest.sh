#!/bin/bash
# Software self-test with SYNTHETIC data (not CFD results). Run from 04_postprocessing.
set -e
python3 selftest/make_synthetic_exports.py
S=selftest/synthetic_data
python3 validate_shear.py --zrakes $S/SYNTHETIC_Q1_zrakes.csv --yrakes $S/SYNTHETIC_Q1_yrakes_z20.csv \
  --wall $S/SYNTHETIC_Q1_wall_bottom.csv --hs2d ../02_analytical/output/hs2d_W10_Q1_centreline.csv \
  --Q 1 --tag SYNTHETIC_Q1 --out selftest/out --label SYNTHETIC
python3 oxygen_post.py "SYNTH 1 IMP=$S/SYNTHETIC_Q1_IMP_wall_bottom_o2.csv" "SYNTH 1 PDMS=$S/SYNTHETIC_Q1_PDMS_wall_bottom_o2.csv" --out selftest/out
echo "Self-test finished: expected dev_fit_bottom at S1 = +8.96 % (aspect-ratio correction built into the synthetic data)."
