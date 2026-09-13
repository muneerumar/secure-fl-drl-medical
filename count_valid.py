"""Count results produced by the *current* source, not just files on disk."""
import glob, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_one import source_hash

SRC = source_hash()
TARGETS = {"main": 25, "ablation": 21, "privacy": 18, "attacks": 21,
           "partition": 12}
total = 0
rows = []
for e, t in TARGETS.items():
    n = 0
    for f in glob.glob(f"results/bloodmnist/{e}/*.json"):
        try:
            if json.load(open(f)).get("source_hash") == SRC:
                n += 1
        except Exception:
            pass
    total += n
    rows.append((e, n, t))
if "--quiet" in sys.argv:
    print(total)
else:
    print(f"source {SRC}")
    for e, n, t in rows:
        print(f"  {e:<11} {n:>2} / {t}")
    print(f"  {'TOTAL':<11} {total:>2} / 97")
