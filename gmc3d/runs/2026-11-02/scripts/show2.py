import json, sys
rows = json.load(open(sys.argv[1]))
by = {}
for r in rows:
    by[(r["prob"], r["tally"], r["be"])] = r
print("{:7s} {:12s} {:>7s} {:>7s} {:>7s} {:>6s} {:>6s} {:>6s}".format(
    "prob", "tally", "mc_s", "ball_s", "ratio", "fl_mc", "fl_bl", "balls"))
seen = []
for r in rows:
    k = (r["prob"], r["tally"])
    if k in seen:
        continue
    seen.append(k)
    m, b = by[(*k, "mc")], by[(*k, "oracle")]
    print("{:7s} {:12s} {:7.2f} {:7.2f} {:6.2f}x {:6.1f} {:6.1f} {:6.2f}".format(
        k[0], k[1], m["sec"], b["sec"], m["sec"] / b["sec"],
        m["fl"], b["fl"], b["balls"]))
