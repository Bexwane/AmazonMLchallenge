"""Render reports/FULL_DENSITY_BASELINE.md, EXPERIMENT_COMPARISON.md and MEMORY_REPORT.md from measured
outputs (nothing here is typed in by hand).

  python scripts/make_reports.py --baseline W/experiments/E003-fullblock/blocking_full.json \
      --probe PROBE_OUT/experiments.json [--final W/experiments/E007-fullblock/blocking_full.json] --out reports
"""
import argparse
import json
from pathlib import Path


def _f(x, nd=4):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _peak(p):
    if not p:
        return "n/a"
    if isinstance(p, dict):
        return f"{p.get('self')} GiB (children {p.get('children')})"
    return f"{p} GiB"


def baseline_md(b):
    L = ["# Full-Density Blocking Baseline (all train Source-1, no sampling)", "",
         f"Blocker: `{b['blocker']}`", "", "| metric | value |", "|---|---|"]
    keys = ["pairs", "pairs_per_s1", "pair_recall", "recall_India", "recall_US", "recall_S2", "recall_S3",
            "recall@1", "recall@3", "recall@5", "recall@10", "recall@20", "recall@30", "oracle_macro_f05"]
    L += [f"| {k} | {_f(b.get(k))} |" for k in keys if k in b]
    L += [f"| blocking runtime (min) | {_f(b.get('block_min'), 1)}{' (cached)' if b.get('from_cache') else ''} |",
          f"| total runtime incl. diagnosis (min) | {_f(b.get('total_min'), 1)} |",
          f"| peak RAM after blocking | {_peak(b.get('peak_gib_after_blocking'))} |",
          f"| peak RAM total | {_peak(b.get('peak_gib'))} |", "",
          "## Present vs absent", "", "| bucket | true pairs | share | " +
          " | ".join(k for k in b["presence"][0] if k.startswith("share_")) + " |",
          "|---|---|---|" + "---|" * sum(k.startswith("share_") for k in b["presence"][0])]
    for r in b["presence"]:
        L.append(f"| {r['bucket']} | {r['true_pairs']:,} | {r['share']:.4f} | " +
                 " | ".join(f"{v:.4f}" for k, v in r.items() if k.startswith("share_")) + " |")
    L += ["", f"Absent pairs with no surviving key (name_cos + addr_cos = 0): {b['absent_no_key']:.2%}.",
          f"Full-pool rank quantiles of scored-but-absent pairs (sample {b['rank_sample']:,}): {b['absent_rank_quantiles']}"]
    return "\n".join(L)


def comparison_md(base, probe, final):
    n = probe["n_s1_full"]
    L = ["# Experiment Comparison (blocking only; no model CV during E004-E007)", "",
         "Probe rows: sampled train Source-1 queries per country searched against the FULL country pool "
         "(exact full-density recall, sampling error about 0.3 pt; candidate counts scaled to all "
         f"{n:,} train S1). Recall is measured on the deduplicated UNION; no RRF, no truncation.", "",
         f"Selection rule: a channel joins if it adds >= {probe['selection']['min_gain']:.3f} recall and >= "
         f"{probe['selection']['min_eff']:.4f} recall per extra candidate per S1, within "
         f"{probe['selection']['max_pairs_per_s1']:g} candidates per S1.", "",
         "| Experiment | Spec | Candidates (est.) | Pairs/S1 | Pair Recall | India | US | Incremental Recall | CV F0.5 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    if base:
        L.append(f"| E003 (all S1, measured) | base | {base['pairs']:,} | {base['pairs_per_s1']:.1f} | "
                 f"{base['pair_recall']:.4f} | {base.get('recall_India', float('nan')):.4f} | "
                 f"{base.get('recall_US', float('nan')):.4f} | baseline | not run |")
    for e in probe["experiments"]:
        pc = e["recall_by_country"]
        L.append(f"| {e['experiment']} | `{e['spec']}` | {e['candidates_est']:,.0f} | {e['pairs_per_s1']:.1f} | "
                 f"{e['pair_recall']:.4f} | {pc.get('India', float('nan')):.4f} | {pc.get('US', float('nan')):.4f} | "
                 f"{e['incremental_recall']:+.4f} | not run |")
    if final:
        L.append(f"| E007 (all S1, measured) | `{final['blocker']}` | {final['pairs']:,} | {final['pairs_per_s1']:.1f} | "
                 f"{final['pair_recall']:.4f} | {final.get('recall_India', float('nan')):.4f} | "
                 f"{final.get('recall_US', float('nan')):.4f} | "
                 f"{final['pair_recall'] - base['pair_recall'] if base else float('nan'):+.4f} | not run |")
    L += ["", "## Greedy steps", ""]
    for e in probe["experiments"]:
        if e["steps"]:
            L += [f"**{e['experiment']}**", "", "| add | recall | gain | extra pairs/S1 | recall per extra pair |",
                  "|---|---:|---:|---:|---:|"]
            L += [f"| {s['add']} | {s['recall']:.4f} | {s['gain']:+.4f} | {s['extra_pairs_per_s1']:.1f} | "
                  f"{s['recall_per_extra_pair']:.5f} |" for s in e["steps"]]
            L.append("")
    L += ["## Channels alone (per country)", "",
          "| country | channel | recall | recall@10 | recall@45 | gain over base | new pairs/q | cand. increase % | sec (probe) |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c, chans in probe["channels"].items():
        for name, r in chans.items():
            L.append(f"| {c} | {r['spec']} | {r['recall']:.4f} | {r['recall@10']:.4f} | {r['recall@45']:.4f} | "
                     f"{_f(r.get('gain_over_base'))} | {_f(r.get('new_pairs_per_q'), 1)} | "
                     f"{_f(r.get('candidate_increase_pct'), 1)} | {r['sec']} |")
    L += ["", "## Base misses: failure mode share and recovery rate by channel", ""]
    for c, modes in (probe.get("base_miss_modes") or {}).items():
        if not modes:
            continue
        rec = probe["recovery_by_mode"][c]
        chans = list(rec)
        L += [f"**{c}**", "", "| mode | share of base misses | " + " | ".join(x[4:] for x in chans) + " |",
              "|---|---:|" + "---:|" * len(chans)]
        for m, s in modes.items():
            L.append(f"| {m} | {s:.3f} | " + " | ".join(_f(rec[x].get(m), 3) for x in chans) + " |")
        L.append("")
    dense = {c: d for c, d in probe.get("dense_info", {}).items() if d}
    if dense:
        L += ["## Multilingual retrieval details", "", "```", json.dumps(dense, indent=1), "```"]
    return "\n".join(L)


def memory_md(base, probe, final):
    L = ["# Memory Report (Kaggle limit 30 GiB)", "", "| run | peak RAM | runtime | note |", "|---|---|---|---|"]
    if base:
        L.append(f"| E003 all-S1 blocking eval | {_peak(base.get('peak_gib'))} | {base.get('total_min')} min | "
                 f"blocking {base.get('block_min')} min |")
    if final:
        L.append(f"| E007 all-S1 blocking eval | {_peak(final.get('peak_gib'))} | {final.get('total_min')} min | "
                 f"blocking {final.get('block_min')} min |")
    for c, p in probe.get("peak_gib", {}).items():
        L.append(f"| probe {c} (one process) | {_peak(p)} | {probe['sec_total'][c] / 60:.1f} min | sampled queries, all channels |")
    L += ["", "## Per channel (probe, RSS after the channel ran, cumulative within the process)", "",
          "| country | channel | RSS after (GiB) | peak so far |", "|---|---|---:|---|"]
    for c, chans in probe["channels"].items():
        for name, r in chans.items():
            L.append(f"| {c} | {r['spec']} | {r['rss_gib_after']} | {_peak(r['peak_gib'])} |")
    L += ["", "Rules kept: no N x M similarity matrix (sparse top-k per query chunk); dense embeddings only for the "
          "native-script subset in fp16 on the GPU; one process per country; parquet caches."]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--final")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text()) if p and Path(p).exists() else None
    base, probe, final = load(a.baseline), load(a.probe), load(a.final)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if base:
        (out / "FULL_DENSITY_BASELINE.md").write_text(baseline_md(base), encoding="utf-8")
    (out / "EXPERIMENT_COMPARISON.md").write_text(comparison_md(base, probe, final), encoding="utf-8")
    (out / "MEMORY_REPORT.md").write_text(memory_md(base, probe, final), encoding="utf-8")
    print("wrote", sorted(p.name for p in out.glob("*.md")))


if __name__ == "__main__":
    main()
