import sys, json
for l in open(sys.argv[1]):
    if not l.startswith("BR "):
        continue
    d = json.loads(l[3:])
    print("{} r{} {:7s} tal{} {:7s} {:6.2f}s adv {:6.2f} be {:5.2f} "
          "fl {:6.2f} balls {:5.2f} leak {:.5f} +- {:.2g}".format(
              d["tag"], d["rnd"], d["prob"], d["tal"], d["be"], d["sec"],
              d["adv"], d["bes"], d["fl"], d["balls"], d["leak"], d["se"]))
