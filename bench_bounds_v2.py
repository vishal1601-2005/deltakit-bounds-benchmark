"""Extended benchmark for Deltakit issue #216: how the choice of exploration bounds affects the gradient of 1/Lambda.

Two noise parameters (p_dep: depolarising noise after Clifford gates; p_meas: measurement flip), rotated planar code.
Bounds for each parameter are (point / f, point * f).  Runs tasks in parallel processes and resumes if interrupted.

Usage:  python bench_bounds_v2.py --repeats 12 --workers 4 --max-shots 1000000
"""
import argparse, json, os, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")

import deltakit_circuit as sp
from deltakit_explorer.analysis.error_budget import (inverse_lambda_gradient_at, FittingParameters,
                                                      SamplingParameters, DiscretisationStrategy)


def noise_model(circuit, params):
    """params = [p_dep, p_meas] (p_meas optional)."""
    c = circuit.copy() if hasattr(circuit, "copy") else circuit
    for prof in sp.after_clifford_depolarisation(float(params[0])):
        c.apply_gate_noise(prof, sp.Circuit.LayerAdjacency.AFTER)
    if len(params) > 1:
        c.replace_gates(sp.measurement_noise_profile(float(params[1])))
    return c


def run_task(task):
    factor, repeat, points, distances, max_shots, batch, rse, minfails = task
    points = np.array(points)
    bounds = [(float(p) / factor, float(p) * factor) for p in points]
    rounds = {int(d): [2, 4] for d in distances}
    t = time.time()
    try:
        g, s = inverse_lambda_gradient_at(
            noise_model, points, rounds, bounds,
            FittingParameters(num_points_per_parameters=6, discretisation_strategy=DiscretisationStrategy.C_OPTIMAL,
                              fitting_degree=3),
            SamplingParameters(max_shots=max_shots, batch_size=batch, lep_target_rse=rse,
                               lep_computation_min_fails=minfails, max_workers=1))
        return dict(factor=factor, repeat=repeat, grads=[float(x) for x in g], reported_std=[float(x) for x in s],
                    error=None, secs=time.time() - t)
    except Exception as e:  # keep failures: they are part of the result
        return dict(factor=factor, repeat=repeat, grads=None, reported_std=None, error=type(e).__name__,
                    secs=time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-shots", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=50_000)
    ap.add_argument("--rse", type=float, default=0.03)
    ap.add_argument("--min-fails", type=int, default=20)
    ap.add_argument("--factors", type=float, nargs="+", default=[1.3, 1.6, 2.0, 3.0, 5.0, 8.0])
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--points", type=float, nargs="+", default=[1e-3, 1e-3],
                    help="noise parameter values at which the gradient is evaluated (one per parameter)")
    ap.add_argument("--out", default="bounds_v2.jsonl")
    a = ap.parse_args()

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line); done.add((r["factor"], r["repeat"]))
    tasks = [(f, r, a.points, a.distances, a.max_shots, a.batch, a.rse, a.min_fails)
             for r in range(a.repeats) for f in a.factors if (f, r) not in done]
    print(f"{len(tasks)} tasks to run ({len(done)} already done), {a.workers} workers", flush=True)
    t0 = time.time(); n = 0
    with open(a.out, "a") as out, ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_task, t) for t in tasks]
        for fu in as_completed(futs):
            rec = fu.result(); out.write(json.dumps(rec) + "\n"); out.flush(); n += 1
            print(f"[{n}/{len(tasks)}] f={rec['factor']} rep={rec['repeat']} "
                  f"{'ok' if rec['error'] is None else rec['error']} {rec['secs']:.0f}s  elapsed {(time.time()-t0)/60:.1f} min",
                  flush=True)


if __name__ == "__main__":
    main()
