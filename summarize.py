"""
Summarizes results.csv (from bfl_bench.py) into paper-ready tables and one figure.

Outputs (in the --outdir folder):
  table_accuracy_<dataset>.csv   mean +- std test accuracy over seeds, defense x (attack, ratio)
  table_detection_<dataset>.csv  detection recall / false-positive rate of the filtering defenses
  fig_accuracy_<dataset>.png     grouped bar chart of test accuracy
  summary.txt                    run counts and a check that every cell has all seeds

Usage:
  python summarize.py                 # reads results.csv
  python summarize.py --inp results.csv --outdir summary
"""
import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["fedavg", "krum", "multikrum", "median", "trimmed", "fltrust", "cosine", "committee"]
NAMES = {"fedavg": "FedAvg (no defense)", "krum": "Krum", "multikrum": "Multi-Krum",
         "median": "Median", "trimmed": "Trimmed mean", "fltrust": "FLTrust",
         "cosine": "Cosine filter", "committee": "Committee + reputation"}
ATT_ORDER = ["none", "labelflip", "signflip", "scaling", "gaussian", "alie"]


def col_label(att, r):
    return "no attack" if att == "none" else f"{att} {int(round(r * 100))}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results.csv")
    ap.add_argument("--outdir", default="summary")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.inp)
    lines = [f"rows: {len(df)}"]

    for ds, d in df.groupby("dataset"):
        cols = sorted({(t, r) for t, r in zip(d.attack, d.mal_ratio)},
                      key=lambda x: (ATT_ORDER.index(x[0]), x[1]))
        acc = pd.DataFrame(index=[NAMES[x] for x in ORDER if x in set(d.defense)])
        det = pd.DataFrame(index=acc.index)
        plot_mean = pd.DataFrame(index=acc.index)
        for att, r in cols:
            sub = d[(d.attack == att) & (d.mal_ratio == r)]
            g = sub.groupby("defense")
            m, s, n = g.test_acc.mean(), g.test_acc.std().fillna(0), g.test_acc.count()
            lab = col_label(att, r)
            acc[lab] = [f"{100 * m[x]:.1f} ± {100 * s[x]:.1f}" if x in m else "" for x in ORDER if x in set(d.defense)]
            plot_mean[lab] = [100 * m.get(x, float("nan")) for x in ORDER if x in set(d.defense)]
            for x in ORDER:
                if x in n and n[x] != n.max():
                    lines.append(f"WARNING {ds} {lab} {x}: only {n[x]} seeds")
            if att != "none":
                rec = g.det_recall.mean(); fpr = g.det_fpr.mean()
                det[lab] = [f"{100 * rec[x]:.0f} / {100 * fpr[x]:.0f}"
                            if x in rec and pd.notna(rec[x]) else "n/a"
                            for x in ORDER if x in set(d.defense)]
        acc.to_csv(os.path.join(a.outdir, f"table_accuracy_{ds}.csv"), encoding="utf-8-sig")
        det.to_csv(os.path.join(a.outdir, f"table_detection_{ds}.csv"), encoding="utf-8-sig")
        lines.append(f"{ds}: {d.seed.nunique()} seeds, {len(d)} runs, mean run time {d.run_s.mean():.0f}s")

        ax = plot_mean.T.plot(kind="bar", figsize=(11, 4), width=0.85)
        ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 100); ax.set_xlabel("")
        ax.legend(fontsize=7, ncol=4, loc="lower left"); ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=20, ha="right"); plt.tight_layout()
        plt.savefig(os.path.join(a.outdir, f"fig_accuracy_{ds}.png"), dpi=300); plt.close()

    with open(os.path.join(a.outdir, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("saved to", a.outdir)


if __name__ == "__main__":
    main()
