"""Line up two Results (ours and a reference) bin by bin.

Used by: run.py, checks/validate.py.

For every tally bin and score, the difference in units of the combined
standard error,

    z = (a - b) / sqrt(se_a^2 + se_b^2),

should look like a standard normal draw if both codes are right.  So:

    chi2/dof   mean of z^2 over bins: about 1 (well above 1 = a real gap)
    max |z|    with N bins, expect about sqrt(2 ln N); 5+ is suspicious
    |z|>3      fraction of bins: about 0.3% if all is well

Skipped bins: those where both standard errors are zero (nothing scored,
e.g. voxels outside the geometry), and those where both values are below
NEGLIGIBLE times the tally's largest value.  The second rule is for
quantities that are zero in truth but not in floating point: OpenMC stores a
zero absorption cross section as about 1e-10 of the total, and a voxel of
void touched by a rounding-sized piece of a flight gets 1e-20.  Comparing
two such numbers measures rounding, not physics.
"""

import numpy as np

SCORE_NAMES = ("flux", "absorption")
NEGLIGIBLE = 1.0e-9


def z_scores(a_mean, a_se, b_mean, b_se):
    a_mean, a_se = np.atleast_1d(np.asarray(a_mean, float)), \
        np.atleast_1d(np.asarray(a_se, float))
    b_mean, b_se = np.atleast_1d(np.asarray(b_mean, float)), \
        np.atleast_1d(np.asarray(b_se, float))
    se = np.sqrt(a_se ** 2 + b_se ** 2)
    scale = max(np.max(np.abs(a_mean)), np.max(np.abs(b_mean)))
    tiny = np.maximum(np.abs(a_mean), np.abs(b_mean)) < NEGLIGIBLE * scale
    ok = (se > 0) & ~tiny
    z = np.zeros_like(a_mean)
    z[ok] = (a_mean[ok] - b_mean[ok]) / se[ok]
    return z, ok


def compare(ours, ref):
    """Returns a dict: per tally and score, the statistics above, plus the
    leakage comparison."""
    out = {}
    z, _ = z_scores(ours.leakage, ours.leakage_se, ref.leakage,
                    ref.leakage_se)
    out["leakage"] = {"ours": ours.leakage, "ours_se": ours.leakage_se,
                      "ref": ref.leakage, "ref_se": ref.leakage_se,
                      "z": float(z[0])}
    for name, t in ours.tallies.items():
        if name not in ref.tallies:
            continue
        r = ref.tallies[name]
        for k, score in enumerate(SCORE_NAMES):
            z, ok = z_scores(t["mean"][:, k], t["se"][:, k],
                             r["mean"][:, k], r["se"][:, k])
            zk = z[ok]
            entry = {"bins": int(ok.sum()),
                     "chi2_dof": float(np.mean(zk ** 2)) if zk.size else
                     float("nan"),
                     "max_abs_z": float(np.max(np.abs(zk))) if zk.size
                     else float("nan"),
                     "frac_over_3": float(np.mean(np.abs(zk) > 3))
                     if zk.size else float("nan")}
            if t["mean"].shape[0] == 1:
                entry.update(ours=float(t["mean"][0, k]),
                             ours_se=float(t["se"][0, k]),
                             ref=float(r["mean"][0, k]),
                             ref_se=float(r["se"][0, k]))
            out[f"{name}/{score}"] = entry
    return out


def report(cmp, title=""):
    """The comparison as aligned text lines."""
    lines = [title] if title else []
    lk = cmp["leakage"]
    lines.append(f"  leakage           ours {lk['ours']:.6g} +- "
                 f"{lk['ours_se']:.2g}   ref {lk['ref']:.6g} +- "
                 f"{lk['ref_se']:.2g}   z {lk['z']:+.2f}")
    for key, e in cmp.items():
        if key == "leakage":
            continue
        if "ours" in e:
            z = (e["ours"] - e["ref"]) / max(np.hypot(e["ours_se"],
                                                      e["ref_se"]), 1e-300)
            lines.append(f"  {key:17} ours {e['ours']:.6g} +- "
                         f"{e['ours_se']:.2g}   ref {e['ref']:.6g} +- "
                         f"{e['ref_se']:.2g}   z {z:+.2f}")
        else:
            lines.append(f"  {key:17} {e['bins']:6d} bins   chi2/dof "
                         f"{e['chi2_dof']:.3f}   max|z| "
                         f"{e['max_abs_z']:.2f}   |z|>3 "
                         f"{100 * e['frac_over_3']:.2f}%")
    return "\n".join(lines)
