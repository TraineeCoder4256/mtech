"""The registry: model name -> Generator class.

Adding a model family = one new file in this folder + one line here.
run.py looks the name up in REGISTRY; nothing else needs to know the family
exists.
"""
from generators.cfm import CFM

REGISTRY = {
    "cfm": CFM,
}
