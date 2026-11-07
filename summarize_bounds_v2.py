"""Summarise bounds_v2.jsonl: table and figure.  Usage: python summarize_bounds_v2.py bounds_v2.jsonl"""
import sys, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "bounds_v2.jsonl"
rows = [json.loads(l) for l in open(path)]
factors = sorted({r["factor"] for r in rows})
nparam = max(len(r["grads"]) for r in rows if r["grads"]) if any(r["grads"] for r in rows) else 1
names = ["depolarising (p_dep)", "measurement flip (p_meas)"][:nparam]
print(f"{'f':>5} {'runs':>5} {'ok':>4} " + " ".join(f"{n[:12]:>12} mean   std/|mean|" for n in names))
table = {}
for f in factors:
    rs = [r for r in rows if r["factor"] == f]
    ok = [r for r in rs if r["error"] is None]
    line = f"{f:>5} {len(rs):>5} {len(ok):>4} "
    for k in range(nparam):
        g = np.array([r["grads"][k] for r in ok])
        rel = np.std(g, ddof=1) / abs(np.mean(g)) if len(g) >= 3 else np.nan
        table[(f, k)] = rel
        line += f"{(np.mean(g) if len(g) else float('nan')):>12.1f}   {rel:>10.2f}  "
    print(line)
errs = {}
for r in rows:
    if r["error"]: errs[r["error"]] = errs.get(r["error"], 0) + 1
print("failures by type:", errs or "none")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
for k in range(nparam):
    ax[0].plot(factors, [table[(f, k)] for f in factors], "o-", label=names[k])
ax[0].set_xscale("log"); ax[0].set_xlabel("bounds = (point / f, point * f), f"); ax[0].set_ylabel("spread of gradient estimate\n(std / |mean|, successful runs)")
ax[0].set_title("Noise in the estimate"); ax[0].legend(fontsize=8)
okfrac = [np.mean([r["error"] is None for r in rows if r["factor"] == f]) for f in factors]
ax[1].bar(range(len(factors)), okfrac, color="#d1242f"); ax[1].set_xticks(range(len(factors)), [str(f) for f in factors])
ax[1].set_ylim(0, 1); ax[1].set_xlabel("f"); ax[1].set_ylabel("fraction of runs that completed"); ax[1].set_title("Runs that did not fail")
fig.suptitle(f"Hand-picked bounds in get_error_budget ({len(rows)} runs, {len(set(r['repeat'] for r in rows))} repeats per width)", fontsize=9)
fig.savefig("bounds_sensitivity_v2.png", dpi=160)
print("wrote bounds_sensitivity_v2.png")
