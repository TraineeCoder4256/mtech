"""The registry: model name -> Generator class.

Adding a model family = one new file in this folder + one line here.
run.py looks the name up in REGISTRY; nothing else needs to know the family
exists.

The two reference generators are not models.  They are the goalposts every
model sits between: `oracle` has zero model error (accuracy goalpost) and
`free` has zero model cost (cost goalpost).
"""
from generators.cfm import CFM
from generators.oracle import Oracle
from generators.free import Free

REGISTRY = {
    "cfm": CFM,
    "oracle": Oracle,
    "free": Free,
}
