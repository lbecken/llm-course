"""
Lab 08a — How documents become training rows.   Tier A (seconds, CPU)

    python labs/pretraining/08_packing.py
    python labs/pretraining/08_packing.py --median 1500 --sigma 1.2    # long documents

Replays three packing strategies on a synthetic distribution of document lengths
(log-normal: most documents short, a long tail) and reports, for several row lengths T:
utilization, tokens discarded by cropping, padding, and how many rows begin at a
document start. `bestfit_crop` is nanochat's pretraining loader, `bestfit_pad` its
fine-tuning loader, `stream` the classic approach it replaced.
Success criterion: you can explain every column, and why cropping grows with T relative
to padding shrinking.
"""

import argparse
import math
import random

from mlle.data.packing import simulate_packing

ap = argparse.ArgumentParser()
ap.add_argument("--median", type=float, default=400.0, help="median document length in tokens")
ap.add_argument("--sigma", type=float, default=1.0, help="log-normal shape: larger = heavier tail")
ap.add_argument("--rows", type=int, default=2000)
args = ap.parse_args()

rng = random.Random(0)
lengths = [max(2, int(rng.lognormvariate(math.log(args.median), args.sigma))) for _ in range(2_000_000)]
sample = sorted(lengths[:100000])
print(f"document lengths (tokens, incl. BOS): median {sample[50000]}, mean {sum(sample) / len(sample):.0f}, "
      f"90th percentile {sample[90000]}, 99th {sample[99000]}\n")

print(f"{'T':>6} {'strategy':>13} {'utilization':>12} {'cropped':>9} {'padding':>9} {'rows starting at a doc':>23}")
for T in (128, 512, 2048):
    for strategy in ("stream", "bestfit_crop", "bestfit_pad"):
        st = simulate_packing(lengths, T + 1, strategy, num_rows=args.rows)
        pad = st.pad_positions / (st.rows * st.row_capacity)
        print(f"{T:>6} {strategy:>13} {st.utilization:>11.1%} {st.cropped_fraction:>8.1%} {pad:>8.1%} "
              f"{st.rows_starting_with_document / st.rows:>22.1%}")
    print()
print("cropped  = share of consumed document tokens thrown away (the price of BOS alignment without padding)")
print("padding  = share of row positions carrying no training signal (the price of never cropping)")
