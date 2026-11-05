"""Medians over seeds 11-13 of race_models.py: seconds, FOM ratio vs OpenMC
analog, and the worst |z| against OpenMC analog per contender."""
import json, math, statistics as st
runs = [json.load(open(f"race_seed{s}.json")) for s in (11, 12, 13)]
out = {}
for p in runs[0]:
    print(f"\n{p}")
    keys = list(runs[0][p][0]["quantities"])
    print(f"  {'contender':30s} {'median s':>9s} {'s range':>14s} {'balls/h':>7s} "
          + " ".join(f"{'FOMx ' + k[:10]:>16s}" for k in keys) + "  worst|z| per seed")
    for i, r0 in enumerate(runs[0][p]):
        secs = [r[p][i]["seconds"] for r in runs]
        fom = {k: st.median(r[p][i]["fom"][k] / r[p][0]["fom"][k] for r in runs) for k in keys}
        zs = []
        for r in runs:
            row, ref = r[p][i], r[p][0]
            zs.append(max(abs((row["quantities"][k][0] - ref["quantities"][k][0]) /
                              math.hypot(row["quantities"][k][1], ref["quantities"][k][1]))
                          for k in keys) if i else 0.0)
        b = r0.get("balls_per_history")
        print(f"  {r0['label'][:30]:30s} {st.median(secs):9.1f} {min(secs):6.1f}-{max(secs):6.1f} "
              f"{('%.2f' % b) if b is not None else '-':>7s} "
              + " ".join(f"{fom[k]:16.2f}" for k in keys) + "  " + " ".join(f"{z:.1f}" for z in zs))
        out.setdefault(p, []).append({"label": r0["label"], "median_s": st.median(secs),
                                      "seconds": secs, "fom_median": fom, "worst_z": zs})
json.dump(out, open("medians.json", "w"), indent=1)
