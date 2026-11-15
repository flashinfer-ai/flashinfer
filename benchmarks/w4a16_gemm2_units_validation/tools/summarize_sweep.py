"""Summarize fc2_unit_sweep.py JSONL: per (shape, route tile), FC2 median us per unit size and
speedup vs one-tile units; fails loudly on any bitwise mismatch."""

import collections
import glob
import json
import math
import sys

rows = []
for path in sys.argv[1:]:
    for line in open(path):
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))

bad = [r for r in rows if r.get("bitwise_vs_1") is False]
errors = [r for r in rows if "error" in r]
print(f"records={len(rows)} bitwise_mismatch={len(bad)} errors={len(errors)}")
for r in bad:
    print("MISMATCH", r)
for r in errors[:20]:
    print("ERROR", r["label"], r["tokens"], r["route_tile"], r["arm"], r["error"][:160])

by = collections.defaultdict(dict)
meta = {}
for r in rows:
    if "us_median" not in r:
        continue
    key = (r["label"], r["route_tile"], r.get("cluster", "1x1"), r["tokens"])
    by[key][str(r["arm"])] = r["us_median"]
    meta[(r["label"], r["route_tile"], r.get("cluster", "1x1"), str(r["arm"]))] = (
        r.get("tiles"), r.get("trans_stages"), r.get("load_stages"))

labels = sorted({k[0] for k in by})
for label in labels:
    print(f"\n## {label}")
    for rt in sorted({k[1] for k in by if k[0] == label}):
        for cl in sorted({k[2] for k in by if k[0] == label and k[1] == rt}):
            keys = sorted(k for k in by if k[:3] == (label, rt, cl))
            arms = sorted({a for k in keys for a in by[k]}, key=lambda a: (a != "1", a != "auto", a))
            auto_meta = meta.get((label, rt, cl, "auto"))
            print(f"route_tile={rt} cluster={cl} auto(tiles,trans,load)={auto_meta}")
            print("tokens | " + " | ".join(f"{a}" for a in arms))
            geo = collections.defaultdict(list)
            for k in keys:
                base = by[k].get("1")
                cells = []
                for a in arms:
                    v = by[k].get(a)
                    if v is None:
                        cells.append("-")
                        continue
                    if base:
                        s = base / v - 1
                        if k[3] >= 4:
                            geo[a].append(math.log(base / v))
                        cells.append(f"{v:.2f} ({100*s:+.1f}%)" if a != "1" else f"{v:.2f}")
                    else:
                        cells.append(f"{v:.2f}")
                print(f"{k[3]:>6} | " + " | ".join(cells))
            print("geomean(t>=4) | " + " | ".join(
                f"{100*(math.exp(sum(geo[a])/len(geo[a]))-1):+.2f}%" if geo[a] else "-" for a in arms))
