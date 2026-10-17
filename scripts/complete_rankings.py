#!/usr/bin/env python3
"""CMP 170HX port: complete the static hot-expert ranking so QWEN38_HOT_ONLY can place every expert on the GPUs.

The shipped ranking lists only the experts seen in the author's traffic (409-501 of 512 per layer), which caps
QWEN38_HOT_ONLY below the full expert set. With 64 GB per card all experts fit, so each layer's list keeps its
order and gets the unranked experts appended in id order. Usage: complete_rankings.py <rankings.json> <num_experts>
"""
import json
import sys

path, n = sys.argv[1], int(sys.argv[2])
r = json.load(open(path))
for k, ids in r.items():
    if not k.isdigit():
        continue
    seen = {int(x) for x in ids}
    ids.extend(e for e in range(n) if e not in seen)
json.dump(r, open(path, "w"))
print(f"completed {sum(k.isdigit() for k in r)} layers to {n} experts each")
