import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows=[json.loads(l) for l in open("./bounds_C_OPTIMAL.jsonl")]
f=np.array([r["factor"] for r in rows]); ok=np.array([len(r["grads"])/r["n_runs"] for r in rows])
rel=np.array([ (np.std(r["grads"],ddof=1)/abs(np.mean(r["grads"]))) if len(r["grads"])>=2 else np.nan for r in rows])
fig,ax=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
ax[0].plot(f,rel,"o-",color="#1f6feb"); ax[0].set_xscale("log"); ax[0].set_xlabel("bounds = (p/2 / f, p/2 * f), f")
ax[0].set_ylabel("spread of gradient estimate\n(std / |mean|, repeated runs)"); ax[0].set_title("Noise in the estimate")
ax[1].bar(range(len(f)),ok,color="#d1242f"); ax[1].set_xticks(range(len(f)),[str(x) for x in f]); ax[1].set_ylim(0,1)
ax[1].set_xlabel("f"); ax[1].set_ylabel("fraction of runs that completed"); ax[1].set_title("Runs that did not crash")
fig.suptitle("Hand-picked bounds in get_error_budget: noisy when narrow, crash-prone when wide\n(rotated planar code d=3,5; 1 parameter; 6 repeats; small sampling budget)",fontsize=9)
fig.savefig("./bounds_sensitivity.png",dpi=160)
