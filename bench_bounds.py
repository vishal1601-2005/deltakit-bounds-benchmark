"""Benchmark: how the choice of exploration bounds affects the estimate of the gradient of 1/Lambda.
Single parameter (uniform depolarising noise after Cliffords), rotated planar code, d in {3,5}.
"""
import sys, json, time, warnings, numpy as np, deltakit_circuit as sp
warnings.filterwarnings("ignore")
from deltakit_explorer.analysis.error_budget import (inverse_lambda_gradient_at, FittingParameters,
    SamplingParameters, DiscretisationStrategy)

def noise_model(circuit, params):
    p = float(params[0])
    c = circuit.copy() if hasattr(circuit, "copy") else circuit
    for prof in sp.after_clifford_depolarisation(p):
        c.apply_gate_noise(prof, sp.Circuit.LayerAdjacency.AFTER)
    return c

point = 1e-3            # = p/2 with p = 2e-3 (gradient evaluated at p/2, as in get_error_budget)
factors = [1.3, 1.6, 2.0, 3.0, 5.0, 8.0]
R = int(sys.argv[1]) if len(sys.argv) > 1 else 6
strategy = DiscretisationStrategy[sys.argv[2]] if len(sys.argv) > 2 else DiscretisationStrategy.C_OPTIMAL
out = open(f"./bounds_{strategy.name}.jsonl", "w")
for f in factors:
    bounds = [(point / f, point * f)]
    gs, ss, ts = [], [], []
    nfail, errs = 0, []
    for r in range(R):
        t = time.time()
        try:
          g, s = inverse_lambda_gradient_at(noise_model, np.array([point]), {3: [2, 4], 5: [2, 4]}, bounds,
            FittingParameters(num_points_per_parameters=6, discretisation_strategy=strategy, fitting_degree=3),
            SamplingParameters(max_shots=200_000, batch_size=20_000, lep_target_rse=0.05,
                               lep_computation_min_fails=20, max_workers=1))
          gs.append(float(g[0])); ss.append(float(s[0]))
        except Exception as e:
          nfail += 1; errs.append(type(e).__name__)
        ts.append(time.time() - t)
    rec = dict(failed_runs=nfail, errors=sorted(set(errs)), n_runs=R, factor=f, lo=bounds[0][0], hi=bounds[0][1], grads=gs, reported_std=ss, secs=ts)
    out.write(json.dumps(rec) + "\n"); out.flush()
    if len(gs) >= 2:
        print(f, "ok %d/%d  mean grad %.1f  empirical std %.1f  mean reported std %.1f  secs/run %.1f" %
          (len(gs), R, np.mean(gs), np.std(gs, ddof=1), np.mean(ss), np.mean(ts)), flush=True)
    else:
        print(f, "ok %d/%d  (too few successful runs)" % (len(gs), R), errs[:1], flush=True)
