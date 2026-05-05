"""Generate retrieval charts and judge score table from eval results."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

RESULTS_PATH = "eval/results_v2_final.json"
OUT_DIR = Path("eval/figures")
OUT_DIR.mkdir(exist_ok=True)

with open(RESULTS_PATH) as f:
    data = json.load(f)

summary = data["summary"]
modes = ["text", "protein", "hybrid"]
colors = {"text": "#4C72B0", "protein": "#DD8452", "hybrid": "#55A868"}

# ── 1. Recall@K bar chart ────────────────────────────────────────────────────
ks = [1, 3, 5, 10]
recall_keys = [f"recall_at_{k}" for k in ks]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(ks))
width = 0.25

for i, mode in enumerate(modes):
    vals = [summary["per_mode"][mode][k] for k in recall_keys]
    bars = ax.bar(x + i * width, vals, width, label=mode.capitalize(),
                  color=colors[mode], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x + width)
ax.set_xticklabels([f"R@{k}" for k in ks], fontsize=11)
ax.set_ylabel("Recall", fontsize=11)
ax.set_title("Recall@K by Retrieval Mode", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.08)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "recall_at_k.png", dpi=150)
plt.close()

# ── 2. Ranking metrics grouped bar chart (MRR, MAP, NDCG@5, NDCG@10) ────────
rank_metrics = ["mrr", "map", "ndcg_at_5", "ndcg_at_10"]
rank_labels  = ["MRR", "MAP", "NDCG@5", "NDCG@10"]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(rank_metrics))

for i, mode in enumerate(modes):
    vals = [summary["per_mode"][mode][k] for k in rank_metrics]
    bars = ax.bar(x + i * width, vals, width, label=mode.capitalize(),
                  color=colors[mode], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x + width)
ax.set_xticklabels(rank_labels, fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Ranking Metrics by Retrieval Mode", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.08)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "ranking_metrics.png", dpi=150)
plt.close()

# ── 3. Radar chart — all retrieval metrics per mode ──────────────────────────
radar_keys   = ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
                "mrr", "map", "ndcg_at_5", "ndcg_at_10"]
radar_labels = ["R@1", "R@3", "R@5", "R@10", "MRR", "MAP", "NDCG@5", "NDCG@10"]
N = len(radar_keys)
angles = [n / N * 2 * np.pi for n in range(N)] + [0]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for mode in modes:
    vals = [summary["per_mode"][mode][k] for k in radar_keys]
    vals += [vals[0]]
    ax.plot(angles, vals, color=colors[mode], linewidth=2, label=mode.capitalize())
    ax.fill(angles, vals, color=colors[mode], alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
ax.set_title("Retrieval Performance Radar", fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "radar.png", dpi=150)
plt.close()

# ── 4. Recall@K — hybrid-critical questions only ─────────────────────────────
with open("eval/questions_v2.json") as f:
    questions = json.load(f)
hybrid_ids = {q["id"] for q in questions if q["retrieval_bias"] == "hybrid"}

hybrid_recall = {mode: {k: [] for k in recall_keys} for mode in modes}
for r in data["results"]:
    if r["id"] not in hybrid_ids:
        continue
    for mode in modes:
        m = r["modes"][mode]
        for k in recall_keys:
            hybrid_recall[mode][k].append(m[k])

hybrid_ks = [1, 3, 10]
hybrid_recall_keys = [f"recall_at_{k}" for k in hybrid_ks]

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(hybrid_ks))

for i, mode in enumerate(modes):
    vals = [np.mean(hybrid_recall[mode][k]) for k in hybrid_recall_keys]
    bars = ax.bar(x + i * width, vals, width, label=mode.capitalize(),
                  color=colors[mode], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels([f"R@{k}" for k in hybrid_ks], fontsize=11)
ax.set_ylabel("Recall", fontsize=11)
ax.set_title("Recall@K on Hybrid-Critical Questions (n=12)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.12)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "recall_hybrid_critical.png", dpi=150)
plt.close()

print("Saved figures to eval/figures/:")
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name}")
