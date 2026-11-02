"""Turn the logs into the tables the write-up needs."""
import re, sys, collections, statistics as st

S = "/tmp/claude-0/-home-claude-mtech/59c0a5da-f9a2-52ca-86e4-11e0c02c807a/scratchpad"
VERS = ["v1.0", "v1.1.0", "v1.1.3", "v1.2.0", "v1.3.0", "v1.4.0",
        "v1.4.1", "v1.5.0", "v1.6.0", "v1.6.1"]
PROB = ["sphere", "slab", "cask"]


def table(title, head, rows):
    print(f"\n### {title}\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join("---" for _ in head) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


# ---- pass 1: the problems' own (growing) tally sets -------------------
own = {}
for ln in open(f"{S}/sweep.log"):
    m = re.match(r"(\S+)\s+(\w+)\s+with tallies\s+([\d.]+) s\s+no tallies\s+"
                 r"([\d.]+) s\s+tallies cost\s+([\d.]+) s\s+locate\s+(\d+) ns"
                 r"\s+flights ([\d,]+)", ln)
    if m:
        own[(m.group(1), m.group(2))] = dict(
            with_t=float(m.group(3)), no_t=float(m.group(4)),
            locate=int(m.group(6)), flights=int(m.group(7).replace(",", "")))

# ---- pass 2: interleaved rounds, v1.0's tally set in every version ----
rnd = collections.defaultdict(list)
adv = collections.defaultdict(list)
wit = collections.defaultdict(list)
ntal = {}
try:
    for ln in open(f"{S}/phase2.log"):
        m = re.match(r"ROW (\S+) (\S+) (\w+) no_t ([\d.]+) with_t ([\d.]+) "
                     r"adv ([\d.]+) flights (\d+) leak ([\d.]+) ntally (\d+)",
                     ln)
        if m:
            k = (m.group(1), m.group(3))
            rnd[k].append(float(m.group(4)))
            wit[k].append(float(m.group(5)))
            adv[k].append(float(m.group(6)))
            ntal[k] = int(m.group(9))
except FileNotFoundError:
    pass

desc, src = {}, {}
try:
    for ln in open(f"{S}/phase2.log"):
        m = re.match(r"DESC (\S+) (\w+)\s+(\d+) ns", ln)
        if m:
            desc[(m.group(1), m.group(2))] = int(m.group(3))
        m = re.match(r"SRC (\S+) (\w+)\s+([\d.]+) ms", ln)
        if m:
            src[(m.group(1), m.group(2))] = float(m.group(3))
except FileNotFoundError:
    pass


def fmt(v, f="{:.2f}"):
    return f.format(v) if v is not None else "-"


print("## Plain MC, no tallies at all: seconds for 500,000 histories")
rows = []
for v in VERS:
    r = [v]
    for p in PROB:
        best = min(rnd[(v, p)]) if rnd.get((v, p)) else None
        r.append(fmt(best) if best else fmt(own.get((v, p), {}).get("no_t")))
    for p in PROB:
        b0 = min(rnd[("v1.0", p)]) if rnd.get(("v1.0", p)) else \
            own.get(("v1.0", p), {}).get("no_t")
        b = min(rnd[(v, p)]) if rnd.get((v, p)) else \
            own.get((v, p), {}).get("no_t")
        r.append(f"{b / b0:.2f}x" if b and b0 else "-")
    rows.append(r)
table("no tallies", ["version"] + PROB + [p + " vs v1.0" for p in PROB],
      rows)

print("\n## Where that time sits")
rows = []
for v in VERS:
    r = [v]
    for p in PROB:
        n = min(rnd[(v, p)]) if rnd.get((v, p)) else None
        a = min(adv[(v, p)]) if adv.get((v, p)) else None
        r.append(f"{a:.2f} / {n - a:.2f}" if n and a else "-")
    for p in PROB:
        r.append(str(desc.get((v, p), "-")))
    for p in PROB:
        r.append(f"{src[(v, p)]:.0f}" if (v, p) in src else "-")
    rows.append(r)
table("advance s / rest s, then ns per _descend, then source ms per batch",
      ["version"] + [p + " adv/rest" for p in PROB] +
      [p + " desc ns" for p in PROB] + [p + " src ms" for p in PROB], rows)

print("\n## The tally side")
rows = []
for v in VERS:
    r = [v]
    for p in PROB:
        o = own.get((v, p))
        r.append(f"{o['with_t'] - o['no_t']:.2f}" if o else "-")
    for p in PROB:
        n = min(rnd[(v, p)]) if rnd.get((v, p)) else None
        w = min(wit[(v, p)]) if wit.get((v, p)) else None
        r.append(f"{w - n:.2f}" if n and w else "-")
    for p in PROB:
        r.append(str(ntal.get((v, p), "-")))
    rows.append(r)
table("tally cost: the problem's own set, then v1.0's set, then how many "
      "tallies v1.0's set is",
      ["version"] + [p + " own" for p in PROB] +
      [p + " fixed" for p in PROB] + [p + " n" for p in PROB], rows)
