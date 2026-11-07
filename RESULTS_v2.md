# v2 results: sensitivity to the exploration-bounds factor

Setup: `inverse_lambda_gradient_at` with exploration bounds (p/f, p*f) per parameter, six factors
f in {1.3, 1.6, 2, 3, 5, 8}, 6-point C_OPTIMAL fit of degree 3, depolarising noise plus (second parameter)
measurement noise. Two runs: distances 3 and 5 (20 repeats per factor, `bounds_v2.jsonl`) and distances
3, 5, 7 (10 repeats per factor, `bounds_v2_d357.jsonl`). Relative spread = std / |mean| over repeats.

| f | d=3,5 OK | d=3,5 spread | d=3,5,7 OK | d=3,5,7 spread |
|---|---|---|---|---|
| 1.3 | 20/20 | 126% | 10/10 | 85% |
| 1.6 | 20/20 | 34% | 10/10 | 26% |
| 2.0 | 20/20 | 42% | 10/10 | 9% |
| 3.0 | 17/20 | 18% | 4/10 | 11% (survivors only) |
| 5.0 | 10/20 | 32% | 3/10 | 7% (survivors only) |
| 8.0 | 10/20 | 95% | 1/10 | n/a |

Plots: `sens_d35.png`, `sens_d357.png`.

## Findings (indicative)

- Narrow bounds (f = 1.3) give a very noisy gradient; the spread exceeds the value itself.
- Wide bounds (f >= 3) often make the fit fail (LinAlgError, ValueError), more often at d = 7.
- f around 2 succeeded in every run and had the tightest spread in these runs.
- Spreads at f = 3 and 5 for d = 3,5,7 come only from the few runs that finished, so they say little.
- The depolarising-gradient mean also moves with f, so the bounds choice shifts the estimate, not only its noise.
- The measurement-noise gradient was not resolved at these shot counts (its sign changes between runs).

## Caveats

Small repeat counts, capped shots per point, one circuit family. Treat as indicative, not conclusive.
Reproduce with `python bench_bounds_v2.py --repeats 10 --distances 3 5 7 --max-shots 3000000 --workers 4 --out bounds_v2_d357.jsonl`
and summarise with `python summarize_bounds_v2.py bounds_v2_d357.jsonl`.
