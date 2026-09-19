#!/usr/bin/env python3
"""
process_data.py: Method 1 (People As Nodes) data pipeline for Survey_Results_UC.csv.

Produces two outputs from one shared loading/cleaning step:

1. report_data.json -- complete-case respondents only (zero missing
   answers), a single Pearson correlation metric, and one fixed threshold
   (tau = 0.40) chosen via the percolation sweep below. Consumed by
   generate_plots.py.

2. interactive_data.json -- a richer, exploratory dataset: every respondent
   who answered at least part of the survey, pairwise-complete correlations
   (each edge uses only the questions both respondents answered), and four
   similarity metrics, so the 3D viewer can let a user explore alternative
   thresholds and metrics interactively. Embedded into
   respondent_network_3d.html.
"""

import json
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity

# ============================================================================
# Paths and constants
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent.parent / "data" / "Survey_Results_UC.csv"
REPORT_DATA_OUT = SCRIPT_DIR / "report_data.json"
INTERACTIVE_DATA_OUT = SCRIPT_DIR / "interactive_data.json"
HTML_PATH = SCRIPT_DIR / "respondent_network_3d.html"

LIKERT_MAP = {
    "Strongly Disagree": 1, "Disagree": 2, "Neutral": 3,
    "Agree": 4, "Strongly Agree": 5, "No Comments": 3,
}
CATEGORIES = ["T", "E", "S", "V"]
CATEGORY_NAMES = {"T": "Technology", "E": "Education", "S": "Ethics and Society", "V": "Environment"}

# Colors for the interactive 3D page 
CATEGORY_COLORS_3D = {"T": "#4C72B0", "E": "#DD8452", "S": "#E8C547", "V": "#8172B2"}

TAU = 0.40                     # operating threshold
LEANING_MARGIN = 0.10          # min gap between top two category means to assign a leaning
NULL_MODEL_TRIALS = 30
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================================
# Loading and cleaning
# ============================================================================

def load_survey(csv_path):
    """Read the raw CSV and identify the 60 question columns by category."""
    df = pd.read_csv(csv_path)
    id_col = df.columns[0]
    q_cols = [c for c in df.columns if c != id_col]
    categories = {c: c.split(".")[0][0] for c in q_cols}
    return df, id_col, q_cols, categories


def split_respondents(df, id_col, q_cols):
    """
    Split respondents by completeness:
      - drop fully blank rows
      - "not fully blank" = every respondent who answered at least one question
      - "complete case" = every respondent who answered all 60 questions
    Returns (not_fully_blank_df, complete_df, excluded_dropout_info).
    """
    fully_blank = df[q_cols].isna().all(axis=1)
    not_blank = df[~fully_blank].reset_index(drop=True)

    n_missing = not_blank[q_cols].isna().sum(axis=1)
    complete = not_blank[n_missing == 0].reset_index(drop=True)
    excluded = not_blank[n_missing > 0].reset_index(drop=True)

    dropout_info = []
    for _, row in excluded.iterrows():
        answered = [q for q in q_cols if pd.notna(row[q])]
        last_answered_cat = answered[-1].split(".")[0][0] if answered else None
        dropout_info.append({
            "id": int(row[id_col]),
            "n_answered": len(answered),
            "last_category_answered": last_answered_cat,
        })

    return not_blank, complete, dropout_info


def score_matrix(df, q_cols):
    """Map Likert text to 1-5 scores. Returns (scores[N,60], no_comments_mask[N,60])."""
    raw = df[q_cols].to_numpy()
    to_score = np.vectorize(lambda v: LIKERT_MAP.get(v, np.nan), otypes=[float])
    scores = to_score(raw)
    no_comments_mask = (raw == "No Comments")
    return scores, no_comments_mask


# ============================================================================
# Complete-case dataset (single Pearson metric, tau = 0.40)
# ============================================================================

def compute_no_comments_stats(no_comments_mask):
    per_respondent = no_comments_mask.sum(axis=1)
    return {
        "respondents_with_any": int((per_respondent > 0).sum()),
        "total_cells": int(no_comments_mask.sum()),
        "total_possible_cells": int(no_comments_mask.size),
    }

def compute_raw_vs_centered(scores, tau):
    """Compare raw (uncentered) cosine similarity against centered Pearson correlation."""
    n = scores.shape[0]
    pearson = np.corrcoef(scores)

    norms = np.linalg.norm(scores, axis=1, keepdims=True)
    cosine = (scores @ scores.T) / (norms @ norms.T)

    iu = np.triu_indices(n, 1)
    pear_vals, cos_vals = pearson[iu], cosine[iu]

    def sparsify_point(vals):
        """Smallest threshold (in 0.01 steps) at which fewer than 5% of pairs remain."""
        for t in np.arange(vals.max(), 0, -0.01):
            if (vals >= t).mean() < 0.05:
                return round(float(t), 2), int((vals >= t).sum())
        return None, None
    raw_sparsify_tau, raw_sparsify_edges = sparsify_point(cos_vals)

    return {
        "n_pairs": int(len(pear_vals)),
        "raw_cosine": {
            "mean": round(float(cos_vals.mean()), 3),
            "min": round(float(cos_vals.min()), 3),
            "max": round(float(cos_vals.max()), 3),
            "edges_at_tau": int((cos_vals >= tau).sum()),
            "sparsify_threshold": raw_sparsify_tau,
            "edges_at_sparsify_threshold": raw_sparsify_edges,
        },
        "pearson_centered": {
            "mean": round(float(pear_vals.mean()), 3),
            "min": round(float(pear_vals.min()), 3),
            "max": round(float(pear_vals.max()), 3),
            "edges_at_tau": int((pear_vals >= tau).sum()),
        },
    }


def build_graph(ids, pearson, tau, positive_only=False):
    G = nx.Graph()
    G.add_nodes_from(ids)
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            w = pearson[i, j]
            if w >= tau and (not positive_only or w > 0):
                G.add_edge(ids[i], ids[j], weight=float(w))
    return G


def compute_summary_metrics(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    components = list(nx.connected_components(G))
    giant = max(components, key=len)
    giant_g = G.subgraph(giant)
    isolated = [node for node in G.nodes() if G.degree(node) == 0]

    return {
        "nodes": n, "edges": m,
        "density": round(nx.density(G), 4),
        "avg_degree": round(2 * m / n, 2),
        "avg_weighted_degree": round(sum(d for _, d in G.degree(weight="weight")) / n, 2),
        "giant_component_size": len(giant),
        "isolated_nodes": sorted(int(x) for x in isolated),
        "diameter": nx.diameter(giant_g) if len(giant) > 1 else 0,
        "avg_path_length": round(nx.average_shortest_path_length(giant_g), 3) if len(giant) > 1 else 0.0,
        "avg_clustering": round(nx.average_clustering(G), 4),
    }


def compute_percolation_sweep(ids, pearson, taus):
    rows = []
    for tau in taus:
        G = build_graph(ids, pearson, tau)
        giant = max(nx.connected_components(G), key=len)
        try:
            comms = louvain_communities(G, weight="weight", seed=RANDOM_SEED)
            q = modularity(G, comms, weight="weight")
        except Exception:
            q = 0.0
        rows.append({
            "tau": round(float(tau), 2),
            "edges": G.number_of_edges(),
            "giant_component_size": len(giant),
            "avg_clustering": round(nx.average_clustering(G), 4),
            "modularity": round(q, 4),
        })
    return rows


def compute_category_leaning(ids, scores, q_cols, categories):
    cat_idx = {c: [i for i, q in enumerate(q_cols) if categories[q] == c] for c in CATEGORIES}
    cat_means = {c: scores[:, cat_idx[c]].mean(axis=1) for c in CATEGORIES}
    cat_means_df = pd.DataFrame(cat_means)

    def dominant(row):
        ranked = row.sort_values(ascending=False)
        if ranked.iloc[0] - ranked.iloc[1] < LEANING_MARGIN:
            return "None"
        return ranked.index[0]

    leaning = cat_means_df.apply(dominant, axis=1)
    population_means = {c: round(float(cat_means_df[c].mean()), 3) for c in CATEGORIES}
    distribution = {c: int((leaning == c).sum()) for c in CATEGORIES}
    distribution["None"] = int((leaning == "None").sum())

    return {
        "population_means": population_means,
        "leaning_distribution": distribution,
        "per_respondent_leaning": dict(zip((int(i) for i in ids), leaning.tolist())),
    }


def compute_leaning_cohesion(ids, pearson, leaning_map, tau):
    within, across = [], []
    id_list = list(ids)
    n = len(id_list)
    for i in range(n):
        for j in range(i + 1, n):
            w = pearson[i, j]
            if w < tau:
                continue
            la, lb = leaning_map[int(id_list[i])], leaning_map[int(id_list[j])]
            if la == "None" or lb == "None":
                continue
            (within if la == lb else across).append(w)
    return {
        "within_leaning_mean_r": round(float(np.mean(within)), 3) if within else None,
        "within_leaning_n": len(within),
        "across_leaning_mean_r": round(float(np.mean(across)), 3) if across else None,
        "across_leaning_n": len(across),
    }


def compute_centralities(G, overall_mean, leaning_map):
    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight=None)
    try:
        eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in G.nodes()}

    ids = list(G.nodes())
    deg_vals = np.array([degree[i] for i in ids])
    eig_vals = np.array([eigenvector[i] for i in ids])
    eig_degree_corr = round(float(np.corrcoef(deg_vals, eig_vals)[0, 1]), 3)

    def top_n(d, n=3):
        return [{"id": int(k), "value": round(float(v), 4)} for k, v in
                sorted(d.items(), key=lambda kv: -kv[1])[:n]]

    # Find the pair with the largest degree-rank vs weighted-degree-rank divergence,
    # as a concrete illustration that "many connections" != "strong connections".
    deg_rank = {i: r for r, i in enumerate(sorted(degree, key=lambda k: -degree[k]))}
    wdeg_rank = {i: r for r, i in enumerate(sorted(weighted_degree, key=lambda k: -weighted_degree[k]))}
    biggest_divergence_id = max(ids, key=lambda i: deg_rank[i] - wdeg_rank[i])
    peer_id = min(ids, key=lambda i: abs(degree[i] - degree[biggest_divergence_id])) \
        if len(ids) > 1 else biggest_divergence_id

    return {
        "degree": {int(k): v for k, v in degree.items()},
        "weighted_degree": {int(k): round(v, 2) for k, v in weighted_degree.items()},
        "betweenness": {int(k): round(v, 4) for k, v in betweenness.items()},
        "eigenvector": {int(k): round(v, 4) for k, v in eigenvector.items()},
        "top_degree": top_n(degree),
        "top_betweenness": top_n(betweenness),
        "eigenvector_vs_degree_correlation": eig_degree_corr,
        "degree_weighted_degree_divergence_example": {
            "id": int(biggest_divergence_id), "peer_id": int(peer_id),
        },
    }


def compute_negative_correlations(ids, pearson):
    n = len(ids)
    iu = np.triu_indices(n, 1)
    vals = pearson[iu]
    pos, neg = vals[vals > 0], vals[vals < 0]

    pairs = [(ids[iu[0][k]], ids[iu[1][k]], vals[k]) for k in range(len(vals))]
    most_negative = sorted(pairs, key=lambda p: p[2])[:5]
    neg_partners = [int(a) for a, b, w in most_negative] + [int(b) for a, b, w in most_negative]
    most_frequent_id, most_frequent_count = pd.Series(neg_partners).value_counts().index[0], \
        pd.Series(neg_partners).value_counts().iloc[0]

    return {
        "n_negative": int(len(neg)), "n_total": int(len(vals)),
        "pct_negative": round(100 * len(neg) / len(vals), 1),
        "mean_positive_r": round(float(pos.mean()), 3) if len(pos) else None,
        "mean_abs_negative_r": round(float(np.abs(neg).mean()), 3) if len(neg) else None,
        "most_negative_pairs": [
            {"a": int(a), "b": int(b), "r": round(float(w), 3)} for a, b, w in most_negative
        ],
        "most_frequent_negative_partner": {"id": int(most_frequent_id), "count": int(most_frequent_count)},
        # Full distribution, kept for generate_plots.py's histogram (Figure 4).
        "all_pairwise_r": [round(float(v), 4) for v in vals],
    }


def compute_community_detection(G, ids, pearson, tau):
    comms = louvain_communities(G, weight="weight", seed=RANDOM_SEED)
    q_obs = modularity(G, comms, weight="weight")

    null_qs = []
    edges_list = list(G.edges(data="weight"))
    for t in range(NULL_MODEL_TRIALS):
        shuffled_w = rng.permutation([w for _, _, w in edges_list])
        Gp = nx.Graph()
        Gp.add_nodes_from(G.nodes())
        for (a, b, _), w in zip(edges_list, shuffled_w):
            Gp.add_edge(a, b, weight=w)
        c_null = louvain_communities(Gp, weight="weight", seed=RANDOM_SEED + t)
        null_qs.append(modularity(Gp, c_null, weight="weight"))
    null_qs = np.array(null_qs)
    percentile = round(float((null_qs < q_obs).mean() * 100), 0)

    comms_sorted = sorted(comms, key=len, reverse=True)
    substantial = [c for c in comms_sorted if len(c) > 1]

    return {
        "communities": [sorted(int(x) for x in c) for c in comms_sorted],
        "substantial_sizes": [len(c) for c in substantial],
        "observed_modularity": round(float(q_obs), 4),
        "null_model": {
            "trials": NULL_MODEL_TRIALS,
            "mean": round(float(null_qs.mean()), 4),
            "std": round(float(null_qs.std()), 4),
            "percentile": percentile,
            "significant": bool(percentile >= 95),
            "trial_values": [round(float(v), 4) for v in null_qs],
        },
        "_substantial_raw": substantial,  # consumed by k-core step, stripped before writing JSON
    }


def compute_k_core(G, ids, substantial_communities):
    G_pos = nx.Graph()
    G_pos.add_nodes_from(G.nodes())
    for u, v, w in G.edges(data="weight"):
        if w > 0:
            G_pos.add_edge(u, v)

    core = nx.core_number(G_pos)
    max_core = max(core.values())
    core_members = {n for n, c in core.items() if c == max_core}

    all_ids = set(G.nodes())
    grouped = set().union(*substantial_communities) if substantial_communities else set()
    outside = all_ids - grouped

    overlap_by_community = [
        {"size": len(c), "in_core": len(set(c) & core_members)} for c in substantial_communities
    ]
    overlap_by_community.append({"size": len(outside), "in_core": len(outside & core_members)})

    return {
        "max_core_number": max_core,
        "core_size": len(core_members),
        "core_members": sorted(int(x) for x in core_members),
        "community_overlap": overlap_by_community,
    }


def build_network_dataset(complete_df, id_col, q_cols, categories, dropout_info):
    ids = complete_df[id_col].astype(int).tolist()
    scores, no_comments_mask = score_matrix(complete_df, q_cols)
    overall_mean = scores.mean(axis=1)

    pearson = np.corrcoef(scores)
    G = build_graph(ids, pearson, TAU, positive_only=True)

    leaning = compute_category_leaning(ids, scores, q_cols, categories)
    community = compute_community_detection(G, ids, pearson, TAU)
    substantial_raw = community.pop("_substantial_raw")

    return {
        "metadata": {
            "n_total_respondents": len(dropout_info) + len(complete_df),
            "n_complete_case": len(complete_df),
            "n_excluded": len(dropout_info),
            "excluded_dropout_pattern": dropout_info,
            "no_comments": compute_no_comments_stats(no_comments_mask),
            "tau": TAU,
            "leaning_margin": LEANING_MARGIN,
        },
        "raw_vs_centered": compute_raw_vs_centered(scores, TAU),
        "summary_metrics": compute_summary_metrics(G),
        "percolation_sweep": compute_percolation_sweep(ids, pearson, np.arange(0.05, 0.60, 0.05)),
        "category_leaning": leaning,
        "leaning_cohesion": compute_leaning_cohesion(ids, pearson, leaning["per_respondent_leaning"], TAU),
        "centralities": compute_centralities(G, overall_mean, leaning["per_respondent_leaning"]),
        "negative_correlations": compute_negative_correlations(ids, pearson),
        "community_detection": community,
        "k_core": compute_k_core(G, ids, substantial_raw),
    }


# ============================================================================
# Interactive dataset (all non-blank respondents, pairwise-complete, 4 metrics)
# ============================================================================

def pairwise_similarity(scores, method):
    """
    Edge list using only the columns both respondents actually answered.
    `scores` may contain NaN for unanswered questions.
    """
    n = scores.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            mask = ~np.isnan(scores[i]) & ~np.isnan(scores[j])
            overlap = int(mask.sum())
            if overlap < 5:
                continue
            x, y = scores[i, mask], scores[j, mask]

            if method == "cosine_raw":
                w = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
            elif method == "cosine_centered":
                xc, yc = x - x.mean(), y - y.mean()
                denom = np.linalg.norm(xc) * np.linalg.norm(yc)
                w = float(xc @ yc / denom) if denom > 1e-9 else 0.0
            elif method == "spearman":
                rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
                rxc, ryc = rx - rx.mean(), ry - ry.mean()
                denom = np.linalg.norm(rxc) * np.linalg.norm(ryc)
                w = float(rxc @ ryc / denom) if denom > 1e-9 else 0.0
            elif method == "percent_agreement":
                w = float((np.abs(x - y) <= 1).mean())
            else:
                raise ValueError(f"unknown method {method}")

            edges.append({"a": i, "b": j, "w": round(w, 4), "n": overlap})
    return edges


def build_interactive_dataset(not_blank_df, id_col, q_cols, categories):
    ids = not_blank_df[id_col].astype(int).tolist()
    scores, _ = score_matrix(not_blank_df, q_cols)
    n_valid = (~np.isnan(scores)).sum(axis=1)
    n_missing = np.isnan(scores).sum(axis=1)

    cat_idx = {c: [i for i, q in enumerate(q_cols) if categories[q] == c] for c in CATEGORIES}
    cat_means = {c: np.nanmean(scores[:, cat_idx[c]], axis=1) for c in CATEGORIES}

    def safe_round(x, digits=3):
        return round(float(x), digits) if np.isfinite(x) else None

    nodes = []
    for i, rid in enumerate(ids):
        row_cat_means = {c: cat_means[c][i] for c in CATEGORIES}
        answered_cats = {c: v for c, v in row_cat_means.items() if np.isfinite(v)}
        if len(answered_cats) >= 2:
            ranked = sorted(answered_cats, key=lambda c: -answered_cats[c])
            dominant = ranked[0] if answered_cats[ranked[0]] - answered_cats[ranked[1]] >= LEANING_MARGIN \
                else "No clear dominant"
        else:
            dominant = "No clear dominant"
        nodes.append({
            "id": int(rid),
            "overall_mean": safe_round(np.nanmean(scores[i])),
            **{f"{c}_mean": safe_round(row_cat_means[c]) for c in CATEGORIES},
            "n_valid_responses": int(n_valid[i]),
            "n_missing": int(n_missing[i]),
            "dominant_category": dominant,
        })

    id_edges = {}
    for method in ("cosine_raw", "cosine_centered", "spearman", "percent_agreement"):
        raw_edges = pairwise_similarity(scores, method)
        id_edges[method] = [{"a": ids[e["a"]], "b": ids[e["b"]], "w": e["w"], "n": e["n"]} for e in raw_edges]

    return {
        "nodes": nodes,
        "edges": id_edges,
        "methods": ["cosine_raw", "cosine_centered", "spearman", "percent_agreement"],
        "defaultMethod": "cosine_centered",
        "defaultMode": "threshold",
        "defaultThreshold": TAU,
        "defaultPercentile": 10,
        "defaultKnnK": 5,
        "categoryColors": CATEGORY_COLORS_3D,
        "isolatedColor": "#b5b3ac",
        "positiveEdge": {"color": "#2E7D32", "dash": "solid"},
        "negativeEdge": {"color": "#C62828", "dash": "dash"},
        "randomSeed": RANDOM_SEED,
    }


# ============================================================================
# HTML patching: swap the 3D page's embedded dataset for our own
# ============================================================================

def patch_html_dataset(html_path, interactive_data):
    """Replace the page's `const DATA = {...};` block with freshly computed data."""
    html = html_path.read_text(encoding="utf-8")

    start = html.index("const DATA = ")
    end = html.index("\nconst state = {", start)
    if start == -1 or end == -1:
        raise RuntimeError("Could not locate the DATA block in respondent_network_3d.html")

    new_block = "const DATA = " + json.dumps(interactive_data) + ";"
    html = html[:start] + new_block + html[end:]
    html_path.write_text(html, encoding="utf-8")


# ============================================================================
# Entry point
# ============================================================================

def main():
    df, id_col, q_cols, categories = load_survey(CSV_PATH)
    not_blank_df, complete_df, dropout_info = split_respondents(df, id_col, q_cols)

    print(f"Total rows: {len(df)}")
    print(f"Not fully blank: {len(not_blank_df)}  |  Complete-case: {len(complete_df)}")

    network_data = build_network_dataset(complete_df, id_col, q_cols, categories, dropout_info)
    REPORT_DATA_OUT.write_text(json.dumps(network_data, indent=2))
    print(f"Wrote {REPORT_DATA_OUT} ({REPORT_DATA_OUT.stat().st_size / 1024:.1f} KB)")

    interactive_data = build_interactive_dataset(not_blank_df, id_col, q_cols, categories)
    INTERACTIVE_DATA_OUT.write_text(json.dumps(interactive_data, indent=2))
    print(f"Wrote {INTERACTIVE_DATA_OUT} ({INTERACTIVE_DATA_OUT.stat().st_size / 1024:.1f} KB)")

    if HTML_PATH.exists():
        patch_html_dataset(HTML_PATH, interactive_data)
        print(f"Patched dataset embedded in {HTML_PATH.name}")
    else:
        print(f"Skipped HTML patch: {HTML_PATH.name} not found (copy it in before running).")


if __name__ == "__main__":
    main()
