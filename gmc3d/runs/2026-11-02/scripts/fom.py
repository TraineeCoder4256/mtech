import json, sys
ref = json.load(open(f"{sys.argv[1]}/openref.json"))
rows = []
for l in open(f"{sys.argv[1]}/phase11.log"):
    if l.startswith("BR "):
        rows.append(json.loads(l[3:]))
best = {}
for r in rows:                       # keep the fastest round per config
    k = (r["tag"], r["prob"], r["tal"], r["be"])
    if k not in best or r["sec"] < best[k]["sec"]:
        best[k] = r
print("v1.6.1, each problem's own tallies on both sides, 500k histories")
print("{:7s} {:16s} {:>7s} {:>9s} {:>8s} {:>9s}".format(
    "prob", "contender", "sec", "vs OpenMC", "rel err", "FOM x"))
for p in ("sphere", "slab", "cask"):
    o = ref[p]
    re0 = o["se"] / o["leakage"]
    f0 = 1.0 / (re0 ** 2 * o["implicit_s"])
    print("{:7s} {:16s} {:7.2f} {:>9s} {:8.4f} {:9s}".format(
        p, "OpenMC implicit", o["implicit_s"], "-", re0 * 100, "1.00x"))
    for tag, lab in (("v161", "ours, plain MC"), ("v161", "ours + balls")):
        be = "mc" if "plain" in lab else "oracle"
        r = best[(tag, p, 1, be)]
        re1 = r["se"] / r["leak"]
        f1 = 1.0 / (re1 ** 2 * r["sec"])
        print("{:7s} {:16s} {:7.2f} {:8.2f}x {:8.4f} {:8.2f}x".format(
            "", lab, r["sec"], o["implicit_s"] / r["sec"], re1 * 100, f1 / f0))
print()
print("ball step vs our own plain MC (wall time), min of 2 rounds")
print("{:7s} {:>18s} {:>18s} {:>18s} {:>18s}".format(
    "prob", "v1.0 no tallies", "v1.0 own tallies", "v1.6.1 no tallies",
    "v1.6.1 own tallies"))
for p in ("sphere", "slab", "cask"):
    cells = []
    for tag in ("v100", "v161"):
        for tal in (0, 1):
            m, b = best[(tag, p, tal, "mc")], best[(tag, p, tal, "oracle")]
            cells.append((tal, tag, m["sec"], b["sec"], m["sec"] / b["sec"]))
    order = [(0, "v100"), (1, "v100"), (0, "v161"), (1, "v161")]
    out = []
    for key in order:
        c = next(c for c in cells if (c[0], c[1]) == key)
        out.append("{:5.2f}/{:5.2f} {:4.2f}x".format(c[2], c[3], c[4]))
    print("{:7s} {:>18s} {:>18s} {:>18s} {:>18s}".format(p, *out))
print()
print("what the ball run itself did, v1.0 -> v1.6.1 (seconds, x slower)")
for p in ("sphere", "slab", "cask"):
    for tal in (0, 1):
        a = best[("v100", p, tal, "oracle")]["sec"]
        b = best[("v161", p, tal, "oracle")]["sec"]
        am = best[("v100", p, tal, "mc")]["sec"]
        bm = best[("v161", p, tal, "mc")]["sec"]
        print("  {:7s} tallies {} balls {:5.2f} -> {:5.2f} ({:4.2f}x)   "
              "plain MC {:5.2f} -> {:5.2f} ({:4.2f}x)".format(
                  p, "on " if tal else "off", a, b, b / a, am, bm, bm / am))
