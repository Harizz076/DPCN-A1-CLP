#!/usr/bin/env python3
"""
generate_plots.py: Renders the six Method 1 figures from report_data.json.

Pure visualization -- no network computation happens here. Run
process_data.py first to (re)generate report_data.json.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "report_data.json"
FIGURES_DIR = SCRIPT_DIR / "figures"

# Color palette for plotting & visualise different categories.
REST_COLOR = "#F5C85A"
CORE_COLOR = "#8E1B1B"
CATEGORY_COLORS = {"T": "#1f77b4", "E": "#ff7f0e", "S": "#2ca02c", "V": "#9467bd", "None": "#b5b3ac"}
CATEGORY_FULL_NAMES = {"T": "Technology", "E": "Education", "S": "Ethics and Society", "V": "Environment"}


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


# ============================================================================
# Figure 1: Percolation and phase transition (2-panel)
# ============================================================================

def plot_percolation(data):
    sweep = data["percolation_sweep"]
    tau = [row["tau"] for row in sweep]
    edges = [row["edges"] for row in sweep]
    giant = [row["giant_component_size"] for row in sweep]
    clustering = [row["avg_clustering"] for row in sweep]
    modularity = [row["modularity"] for row in sweep]
    tau_mark = data["metadata"]["tau"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1b = ax1.twinx()
    l1, = ax1.plot(tau, edges, marker="o", color="#1f77b4", label="Active Edges |E|")
    l2, = ax1b.plot(tau, giant, marker="s", ls="--", color="#2ca02c", label="Giant Component Size")
    ax1.set_xlabel("Correlation Threshold ($\\tau$)", fontweight="bold")
    ax1.set_ylabel("Active Edges Count", color="#1f77b4", fontweight="bold")
    ax1b.set_ylabel(f"Giant Component Size (Max {data['metadata']['n_complete_case']})", color="#2ca02c", fontweight="bold")
    ax1.set_title("Network Connectivity vs. Threshold", fontweight="bold")
    ax1.axvline(tau_mark, ls=":", color="#d62728", lw=1.5)
    ax1.legend(handles=[l1, l2], loc="upper right")
    ax1.grid(alpha=0.3)

    l3, = ax2.plot(tau, clustering, marker="^", color="#9467bd", label="Clustering Coefficient (C)")
    l4, = ax2.plot(tau, modularity, marker="D", color="#ff7f0e", label="Modularity (Q)")
    ax2.set_xlabel("Correlation Threshold ($\\tau$)", fontweight="bold")
    ax2.set_ylabel("Metric Value [0, 1]", fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.set_title("Clustering (C) and Modularity (Q)", fontweight="bold")
    ax2.axvline(tau_mark, ls=":", color="#d62728", lw=1.5)
    ax2.legend(handles=[l3, l4], loc="upper right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "percolation_analysis.png", dpi=150)
    plt.close(fig)


# ============================================================================
# Figures 2-3: Dominant leaning pie chart + category means bar chart
# ============================================================================

def plot_leaning_pie(data):
    dist = data["category_leaning"]["leaning_distribution"]
    order = [c for c in ["V", "S", "T", "E", "None"] if dist.get(c)]
    counts = [dist[c] for c in order]
    labels = [CATEGORY_FULL_NAMES.get(c, "No clear leaning") for c in order]
    colors = [CATEGORY_COLORS[c] for c in order]
    explode = [0.12 if c in ("T", "E") else 0 for c in order]

    def autopct(pct):
        val = int(round(pct * sum(counts) / 100.0))
        return f"{val} ({pct:.0f}%)"

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.pie(counts, explode=explode, labels=labels, colors=colors, autopct=autopct,
           pctdistance=0.78, labeldistance=1.08, startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5},
           textprops={"fontweight": "bold", "fontsize": 11})
    ax.set_title(f"Dominant Leaning Across {data['metadata']['n_complete_case']} Respondents", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "leaning_pie.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_category_means_bar(data):
    means = data["category_leaning"]["population_means"]
    order = sorted(means, key=lambda c: -means[c])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar([CATEGORY_FULL_NAMES[c] for c in order], [means[c] for c in order],
                  color=[CATEGORY_COLORS[c] for c in order], edgecolor="#333333")
    for bar, c in zip(bars, order):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{means[c]:.2f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 5)
    ax.axhline(3, color="gray", ls=":", lw=1, label="Neutral (3)")
    ax.set_ylabel("Mean response across respondents (1-5)", fontweight="bold")
    ax.set_title("Population-level Mean Agreement by Category", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "category_means_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Figure 4: Distribution of pairwise correlations (positive vs. negative)
# ============================================================================

def plot_correlation_distribution(data):
    neg_stats = data["negative_correlations"]
    vals = neg_stats["all_pairwise_r"]
    pos = [v for v in vals if v >= 0]
    neg = [v for v in vals if v < 0]
    # Label counts come from the unrounded stats, not from re-filtering the
    # (4-decimal-rounded) array above, since rounding can flip a handful of
    # near-zero values across the >=0 boundary.
    n_negative = neg_stats["n_negative"]
    n_positive = neg_stats["n_total"] - n_negative

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pos, bins=30, color=REST_COLOR, alpha=0.95, label=f"Positive (n={n_positive})")
    ax.hist(neg, bins=10, color=CORE_COLOR, alpha=0.95, label=f"Negative (n={n_negative})")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Pairwise Pearson correlation $r$", fontweight="bold")
    ax.set_ylabel("Number of respondent pairs", fontweight="bold")
    ax.set_title(f"Distribution of All {neg_stats['n_total']:,} Pairwise Respondent Correlations", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Figure 5: Observed modularity vs. degree-preserving null model
# ============================================================================

def plot_community_null_model(data):
    null_model = data["community_detection"]["null_model"]
    q_obs = data["community_detection"]["observed_modularity"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(null_model["trial_values"], bins=12, color=REST_COLOR, edgecolor="white",
            label="null model (weight-shuffled)")
    ax.axvline(q_obs, color=CORE_COLOR, lw=2, label=f"observed Q = {q_obs:.3f}")
    ax.set_xlabel("Modularity Q")
    ax.set_ylabel("Null-model trials")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "community_null_model.png", dpi=150)
    plt.close(fig)


# ============================================================================
# Figure 6: Louvain communities and their overlap with the deepest k-core shell
# ============================================================================

def plot_community_core_overlap(data):
    overlap = data["k_core"]["community_overlap"][:-1]  # drop the "Outside" row; not plotted here
    labels = [f"Community {i + 1}\n(n={row['size']})" for i, row in enumerate(overlap)]
    sizes = [row["size"] for row in overlap]
    core_counts = [row["in_core"] for row in overlap]
    rest_counts = [s - c for s, c in zip(sizes, core_counts)]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(len(overlap))
    ax.bar(x, core_counts, color=CORE_COLOR, label=f"In deepest k-core shell (core = {data['k_core']['max_core_number']})")
    ax.bar(x, rest_counts, bottom=core_counts, color=REST_COLOR, label="Rest of community")
    for i, (c, s) in enumerate(zip(core_counts, sizes)):
        ax.text(i, s + 0.6, f"{c}/{s}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Respondents", fontweight="bold")
    ax.set_title("Louvain Communities and Overlap with the Deepest k-core Shell", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "community_core_overlap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES_DIR.mkdir(exist_ok=True)
    data = load_data()

    plot_percolation(data)
    plot_leaning_pie(data)
    plot_category_means_bar(data)
    plot_correlation_distribution(data)
    plot_community_null_model(data)
    plot_community_core_overlap(data)

    print(f"Saved 6 figures to {FIGURES_DIR}")
