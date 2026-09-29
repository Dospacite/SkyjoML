"""Print training progress: dev evaluation credit x n (parity = 1.0) against the run's eval opponent."""

import glob
import json
import sys

pattern = sys.argv[1] if len(sys.argv) > 1 else "artifacts/runs/ppo2p_s*"
for d in sorted(glob.glob(pattern)):
    rows = [json.loads(line) for line in open(f"{d}/metrics.jsonl")]
    if not rows:
        continue
    last = rows[-1]
    ev = []
    for r in rows:
        keys = [k for k in r if k.startswith("eval_") and not k.endswith("_se")]
        if keys:
            n = int(keys[0].split("_")[1][:-1])
            ev.append((r["update"], round(r[keys[0]] * n, 3)))
    print(f"{d.split('/')[-1]}: update {last['update']} decisions {last['decisions']:,} dps {last['dps']:.0f} "
          f"entropy {last['entropy']:.3f} ev {last['explained_var']:.3f}")
    print("   evals:", ev[-12:])
