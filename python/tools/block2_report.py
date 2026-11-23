#!/usr/bin/env python
"""Block 2 report: figures and a results table from one run folder.

    python tools/block2_report.py output/block2/<run folder> [--min-rt-ms 250] [--all]

Writes into the run folder (which stays on the lab PC; output/ is never committed):

* ``results_std.csv`` -- per condition x place type: shown, seen, right, wrong, miss and
  their %, right of seen with a one-sided binomial test against 50 %, the same for
  answered-but-unseen trials (forced choice), catch trials, false presses and d'
  (log-linear correction, Hautus 1995);
* ``block2_greymap.png`` -- per condition, a square per spot: black = seen 0 of n, white
  = all seen (as in Sanchez-Lopez et al. 2017); red edge = not seen in Block 1, dashed =
  natural blind spot;
* ``block2_accuracy.png`` -- direction right (horizontal vs vertical) per condition and
  place type: % of answered (with the binomial test) and % of all shown.

Rules: a trial whose first press came sooner than ``--min-rt-ms`` after onset is left out
(an anticipation, or a press carried over from the trial before -- runs before the 17 Nov
fix had these). Only trials that count (eye on the +) are used, unless ``--all``. The
shown rate, not the asked-for one, goes in the labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
from typing import Dict, List

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from specvis.procedure.block2 import binomial_p  # noqa: E402

KINDS = [("at threshold", "At Block 1 threshold", "#2ca02c", "o"),
         ("not seen in Block 1", "Not seen in Block 1 (100%)", "#d62728", "s"),
         ("blind spot", "Natural blind spot (100%)", "black", "^")]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(folder: pathlib.Path, min_rt_ms: float, all_trials: bool):
    meta = json.loads((folder / "session.json").read_text())
    with open(folder / "trials.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    kept, dropped = [], dict(anticipation=0, fixation=0, no_answer=0)
    for r in rows:
        rt = _num(r.get("reaction_ms_from_onset"))
        if (rt is not None and rt < min_rt_ms) or "ANTICIPATION" in r.get("note", ""):
            dropped["anticipation"] += 1
            continue
        if "NO_ANSWER" in r.get("note", ""):
            dropped["no_answer"] += 1
            continue
        if not all_trials and r.get("counts") == "0":
            dropped["fixation"] += 1
            continue
        r["seen"] = int(r.get("seen", "") or (1 if r.get("answer") else 0))
        r["correct"] = int(r.get("correct") or 0)
        kept.append(r)
    labels = {}
    for rate in meta.get("rates", []):
        hz = float(rate["requested_hz"])
        labels[hz] = ("static" if hz == 0 else
                      f"{hz:g} Hz\n({float(rate['shown_hz']):.2f})")
    return meta, kept, dropped, labels


def _d_prime(hits: int, n: int, fas: int, m: int):
    if not n or not m:
        return ""
    z = statistics.NormalDist().inv_cdf
    return round(z((hits + 0.5) / (n + 1)) - z((fas + 0.5) / (m + 1)), 2)


def table(kept: List[dict], labels: Dict[float, str]) -> List[dict]:
    out = []
    for hz, label in labels.items():
        rs = [r for r in kept if float(r["requested_hz"]) == hz]
        catches = [r for r in rs if r["place"] == "catch"]
        fp = sum(r["seen"] for r in catches)
        patches = [r for r in rs if r["place"] != "catch"]
        for kind, name in [(None, "all")] + [(k, k) for k, _, _, _ in KINDS]:
            ps = [r for r in patches if kind is None or r["kind"] == kind]
            if not ps:
                continue
            seen = [r for r in ps if r["seen"]]
            right = sum(r["correct"] for r in seen)
            unseen = [r for r in ps if not r["seen"] and r.get("answer")]
            right_un = sum(r["correct"] for r in unseen)
            n = len(ps)

            def pct(a, b):
                return round(100.0 * a / b, 1) if b else ""

            def p(a, b):
                v = binomial_p(a, b)
                return v if v == "" else float(f"{v:.3g}")
            out.append(dict(
                condition=label.replace("\n", " "), places=name, shown=n,
                seen=len(seen), seen_pct=pct(len(seen), n), right=right,
                right_pct=pct(right, n), wrong=len(seen) - right,
                wrong_pct=pct(len(seen) - right, n), miss=n - len(seen),
                miss_pct=pct(n - len(seen), n), right_pct_of_seen=pct(right, len(seen)),
                p_seen_vs_50=p(right, len(seen)), unseen_answered=len(unseen),
                right_unseen=right_un, right_pct_of_unseen=pct(right_un, len(unseen)),
                p_unseen_vs_50=p(right_un, len(unseen)),
                catch=len(catches) if kind is None else "",
                false_presses=fp if kind is None else "",
                d_prime=_d_prime(len(seen), n, fp, len(catches)) if kind is None else ""))
    return out


def greymap(kept, labels, meta, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    places = {}
    for r in kept:
        if r["place"] != "catch":
            places.setdefault(r["place"], (float(r["x_deg"]), float(r["y_deg"]), r["kind"]))
    fig, axes = plt.subplots(1, len(labels), figsize=(5.5 * len(labels), 3.6), squeeze=False)
    for ax, (hz, label) in zip(axes[0], labels.items()):
        ax.set_facecolor("#6fa8dc")
        for name, (x, y, kind) in places.items():
            rs = [r for r in kept if r["place"] == name and float(r["requested_hz"]) == hz]
            if not rs:
                continue
            frac = sum(r["seen"] for r in rs) / len(rs)
            ax.add_patch(Rectangle((x - 3, y - 3), 6, 6, facecolor=str(frac),
                                   edgecolor="#1f3b57", lw=0.8))
            if kind != "at threshold":
                ax.add_patch(Rectangle((x - 2.7, y - 2.7), 5.4, 5.4, fill=False,
                                       edgecolor="red" if kind.startswith("not") else "black",
                                       lw=2.5, ls="-" if kind.startswith("not") else "--"))
            ax.text(x, y, f"{sum(r['seen'] for r in rs)}/{len(rs)}", ha="center",
                    va="center", fontsize=9, weight="bold",
                    color="black" if frac > 0.5 else "white")
        ax.plot(0, 0, "+", color="yellow", ms=14, mew=3)
        ax.set_xlim(-25, 25)
        ax.set_ylim(-13, 13)
        ax.set_aspect("equal")
        ax.set_title(label.replace("\n", " "))
        ax.set_xlabel("degrees")
    axes[0][0].set_ylabel("degrees")
    eye = (meta.get("settings") or {}).get("eye", "")
    fig.suptitle(f"Block 2, {'right' if eye == 'R' else 'left'} eye: seen per place "
                 "(black = 0 of n, white = all; red edge = not seen in Block 1, "
                 "dashed = natural blind spot)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def accuracy(rows, labels, meta, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [label.replace("\n", " ") for label in labels.values()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for kind, label, colour, marker in KINDS:
        sel = {r["condition"]: r for r in rows if r["places"] == kind}
        xs = [i for i, n in enumerate(names) if n in sel]
        if not xs:
            continue
        a1.plot(xs, [sel[names[i]]["right_pct_of_seen"] or float("nan") for i in xs],
                marker=marker, color=colour, label=label, lw=2, ms=9)
        a2.plot(xs, [sel[names[i]]["right_pct"] for i in xs], marker=marker, color=colour,
                lw=2, ms=9)
    for ax, title in ((a1, "Direction right, % of seen (binomial vs 50%)"),
                      (a2, "Direction right, % of all shown (no press = not right)")):
        ax.axhline(50, ls="--", color="grey")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(list(labels.values()))
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=10)
    a1.set_ylabel("% right (horizontal vs vertical)")
    a1.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(folder, min_rt_ms: float = 250.0, all_trials: bool = False) -> pathlib.Path:
    folder = pathlib.Path(folder)
    meta, kept, dropped, labels = load(folder, min_rt_ms, all_trials)
    rows = table(kept, labels)
    with open(folder / "results_std.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (folder / "results_std_rules.json").write_text(json.dumps(dict(
        min_rt_ms=min_rt_ms, trials_used="all" if all_trials else "eye on the + only",
        trials_kept=len(kept), left_out=dropped), indent=2))
    greymap(kept, labels, meta, folder / "block2_greymap.png")
    accuracy(rows, labels, meta, folder / "block2_accuracy.png")
    return folder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--min-rt-ms", type=float, default=250.0,
                    help="leave out trials pressed sooner than this after onset")
    ap.add_argument("--all", action="store_true",
                    help="use every trial, also those with the eye off the +")
    args = ap.parse_args()
    folder = report(args.folder, args.min_rt_ms, args.all)
    rules = json.loads((folder / "results_std_rules.json").read_text())
    print(f"{rules['trials_kept']} trials used; left out: {rules['left_out']}")
    with open(folder / "results_std.csv", newline="") as f:
        for r in csv.DictReader(f):
            print(f"{r['condition']:16s} {r['places']:22s} seen {r['seen']}/{r['shown']}"
                  f"  right {r['right_pct_of_seen']}% of seen (p {r['p_seen_vs_50']})")
    print(f"Written to {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
