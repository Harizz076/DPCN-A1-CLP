#!/usr/bin/env python3
"""
process_data.py: Process Survey_Results_UC.csv according to opinion network specifications.

Filtering Rules:
1. Ignore rows where there are NO entries across all questions.
2. For the entire graph (all 4 categories: T, E, S, V), ignore rows that are partially filled (requires all 60 questions filled).
3. For category-wise graphs, include rows where THAT specific category is completely full:
   - Technology (T01-T15)
   - Education (E01-E15)
   - Ethics/Society (S01-S15)
   - Environment (V01-V15)

Outputs:
- Generates web application JSON data with:
  - Questions metadata (ID, category, title, text, full distribution)
  - Full-graph similarity matrices (Pearson correlation, Agreement score, Cosine similarity)
  - Category-specific similarity matrices computed on their respective full cohorts
  - Graph topological properties (density, degree distribution, centralities, modularity, components)
"""

import csv
import json
import math
import os
from collections import Counter, defaultdict

# Likert mapping
LIKERT_MAP = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'No Comments': 3,
    'Agree': 4,
    'Strongly Agree': 5
}

CATEGORY_NAMES = {
    'T': 'Technology',
    'E': 'Education',
    'S': 'Ethics & Society',
    'V': 'Environment'
}

CATEGORY_COLORS = {
    'T': '#0284c7',  # Crisp Sky Blue
    'E': '#d97706',  # Warm Amber
    'S': '#7c3aed',  # Royal Violet
    'V': '#059669'   # Emerald Green
}

def load_and_clean_survey(csv_path):
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        headers = next(reader)
        raw_rows = list(reader)

    # Identify question columns
    question_cols = []
    col_by_cat = {'T': [], 'E': [], 'S': [], 'V': []}

    for col_idx, h in enumerate(headers):
        clean_h = h.strip('\"').strip()
        if clean_h.startswith(('T', 'E', 'S', 'V')) and (len(clean_h) >= 3 and clean_h[1:3].isdigit()):
            cat = clean_h[0]
            q_id = clean_h.split('.')[0].strip()
            # Split title and text
            parts = clean_h.split('.', 1)
            q_text = parts[1].strip() if len(parts) > 1 else clean_h
            
            q_info = {
                'index': len(question_cols),
                'col_idx': col_idx,
                'id': q_id,
                'cat': cat,
                'cat_name': CATEGORY_NAMES[cat],
                'color': CATEGORY_COLORS[cat],
                'text': q_text,
                'full_header': clean_h
            }
            question_cols.append(q_info)
            col_by_cat[cat].append(q_info)

    # Filter rows
    all_col_indices = [q['col_idx'] for q in question_cols]

    # 1. Ignore completely empty rows
    valid_respondent_rows = []
    empty_row_count = 0
    for r in raw_rows:
        has_any = any(r[c].strip() != '' for c in all_col_indices)
        if has_any:
            valid_respondent_rows.append(r)
        else:
            empty_row_count += 1

    # 2. Entire graph: only rows where all 60 questions are filled
    full_graph_rows = []
    for r in valid_respondent_rows:
        if all(r[c].strip() != '' for c in all_col_indices):
            full_graph_rows.append(r)

    # 3. Category-wise rows
    cat_rows = {}
    for cat, q_list in col_by_cat.items():
        cat_col_indices = [q['col_idx'] for q in q_list]
        c_rows = []
        for r in valid_respondent_rows:
            if all(r[c].strip() != '' for c in cat_col_indices):
                c_rows.append(r)
        cat_rows[cat] = c_rows

    return question_cols, col_by_cat, valid_respondent_rows, full_graph_rows, cat_rows, empty_row_count


def compute_question_stats(q_list, rows):
    """Compute distribution, mean, standard deviation for questions."""
    stats = {}
    n = len(rows)
    for q in q_list:
        c_idx = q['col_idx']
        dist = Counter()
        scores = []
        for r in rows:
            val_raw = r[c_idx].strip()
            score = LIKERT_MAP.get(val_raw, 3)
            dist[val_raw] += 1
            scores.append(score)

        mean_val = sum(scores) / n if n > 0 else 0
        variance = sum((s - mean_val) ** 2 for s in scores) / n if n > 0 else 0
        std_val = math.sqrt(variance)

        stats[q['id']] = {
            'mean': round(mean_val, 3),
            'std': round(std_val, 3),
            'variance': round(variance, 3),
            'distribution': {
                'Strongly Disagree': dist['Strongly Disagree'],
                'Disagree': dist['Disagree'],
                'Neutral': dist['Neutral'],
                'Agree': dist['Agree'],
                'Strongly Agree': dist['Strongly Agree'],
                'No Comments': dist['No Comments']
            },
            'n_respondents': n
        }
    return stats


def compute_similarity_matrices(q_list, rows):
    """
    Compute Pearson Correlation, Mean Agreement, and Cosine Similarity
    between all pairs of questions in q_list over the provided rows.
    """
    n = len(rows)
    num_q = len(q_list)
    
    # Pre-extract numerical vectors
    vectors = []
    means = []
    stds = []
    norms = []

    for q in q_list:
        c_idx = q['col_idx']
        v = [LIKERT_MAP.get(r[c_idx].strip(), 3) for r in rows]
        m = sum(v) / n
        var = sum((x - m) ** 2 for x in v) / n
        s = math.sqrt(var)
        norm = math.sqrt(sum(x * x for x in v))

        vectors.append(v)
        means.append(m)
        stds.append(s)
        norms.append(norm)

    pearson_matrix = [[0.0] * num_q for _ in range(num_q)]
    agreement_matrix = [[0.0] * num_q for _ in range(num_q)]
    cosine_matrix = [[0.0] * num_q for _ in range(num_q)]

    for i in range(num_q):
        pearson_matrix[i][i] = 1.0
        agreement_matrix[i][i] = 1.0
        cosine_matrix[i][i] = 1.0

        for j in range(i + 1, num_q):
            # Pearson
            if stds[i] > 1e-9 and stds[j] > 1e-9:
                cov = sum((vectors[i][k] - means[i]) * (vectors[j][k] - means[j]) for k in range(n)) / n
                r = cov / (stds[i] * stds[j])
            else:
                r = 0.0

            # Agreement: 1 - mean(|x_i - x_j|) / 4
            mean_diff = sum(abs(vectors[i][k] - vectors[j][k]) for k in range(n)) / n
            agree = 1.0 - (mean_diff / 4.0)

            # Cosine
            if norms[i] > 1e-9 and norms[j] > 1e-9:
                dot = sum(vectors[i][k] * vectors[j][k] for k in range(n))
                cos = dot / (norms[i] * norms[j])
            else:
                cos = 0.0

            pearson_matrix[i][j] = round(r, 4)
            pearson_matrix[j][i] = round(r, 4)

            agreement_matrix[i][j] = round(agree, 4)
            agreement_matrix[j][i] = round(agree, 4)

            cosine_matrix[i][j] = round(cos, 4)
            cosine_matrix[j][i] = round(cos, 4)

    # Edge list (upper triangle)
    edge_list = []
    for i in range(num_q):
        for j in range(i + 1, num_q):
            edge_list.append({
                'source': q_list[i]['id'],
                'target': q_list[j]['id'],
                'source_idx': i,
                'target_idx': j,
                'pearson': pearson_matrix[i][j],
                'agreement': agreement_matrix[i][j],
                'cosine': cosine_matrix[i][j],
                'same_cat': q_list[i]['cat'] == q_list[j]['cat']
            })

    return {
        'pearson': pearson_matrix,
        'agreement': agreement_matrix,
        'cosine': cosine_matrix,
        'edges': edge_list
    }


def compute_network_topology(nodes, edges, metric_name='pearson', threshold=0.35):
    """
    Compute graph topological properties for a given metric and threshold.
    """
    import networkx as nx

    G = nx.Graph()
    for node in nodes:
        G.add_node(node['id'], **node)

    for e in edges:
        weight = e[metric_name]
        if weight >= threshold:
            G.add_edge(e['source'], e['target'], weight=weight)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)

    # Connected components
    components = list(nx.connected_components(G))
    num_components = len(components)
    giant_comp = max(components, key=len) if components else set()
    giant_size = len(giant_comp)

    # Metrics on giant component or entire graph if connected
    if giant_size > 1:
        giant_subgraph = G.subgraph(giant_comp)
        avg_path_length = nx.average_shortest_path_length(giant_subgraph)
        diameter = nx.diameter(giant_subgraph)
    else:
        avg_path_length = 0.0
        diameter = 0

    avg_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)

    # Centralities
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    closeness_centrality = nx.closeness_centrality(G)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigenvector_centrality = {n: 0.0 for n in G.nodes()}

    # Communities via greedy modularity
    try:
        communities = nx.community.greedy_modularity_communities(G)
        modularity = nx.community.modularity(G, communities)
        comm_map = {}
        for c_idx, comm in enumerate(communities):
            for node_id in comm:
                comm_map[node_id] = c_idx
    except:
        modularity = 0.0
        comm_map = {n: 0 for n in G.nodes()}

    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / n_nodes if n_nodes > 0 else 0

    node_metrics = {}
    for node in nodes:
        nid = node['id']
        node_metrics[nid] = {
            'degree': G.degree(nid),
            'degree_centrality': round(degree_centrality.get(nid, 0), 4),
            'betweenness': round(betweenness_centrality.get(nid, 0), 4),
            'closeness': round(closeness_centrality.get(nid, 0), 4),
            'eigenvector': round(eigenvector_centrality.get(nid, 0), 4),
            'community': comm_map.get(nid, 0)
        }

    return {
        'threshold': threshold,
        'metric': metric_name,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': round(density, 4),
        'avg_degree': round(avg_degree, 2),
        'num_components': num_components,
        'giant_component_size': giant_size,
        'avg_path_length': round(avg_path_length, 3),
        'diameter': diameter,
        'avg_clustering': round(avg_clustering, 4),
        'transitivity': round(transitivity, 4),
        'modularity': round(modularity, 4),
        'node_metrics': node_metrics
    }


def main():
    csv_path = 'Survey_Results_UC.csv'
    q_cols, col_by_cat, valid_rows, full_rows, cat_rows, empty_rows = load_and_clean_survey(csv_path)

    print(f"Data Summary:")
    print(f"  Total valid respondents (excluding {empty_rows} empty rows): {len(valid_rows)}")
    print(f"  Complete respondents across ALL 60 questions: {len(full_rows)}")
    for c in ['T', 'E', 'S', 'V']:
        print(f"  Complete respondents for Category {c} ({CATEGORY_NAMES[c]}): {len(cat_rows[c])}")

    # Compute Question Stats
    full_stats = compute_question_stats(q_cols, full_rows)
    cat_stats = {
        c: compute_question_stats(col_by_cat[c], cat_rows[c]) for c in ['T', 'E', 'S', 'V']
    }

    # Compute Matrices
    full_matrices = compute_similarity_matrices(q_cols, full_rows)
    cat_matrices = {
        c: compute_similarity_matrices(col_by_cat[c], cat_rows[c]) for c in ['T', 'E', 'S', 'V']
    }

    # Precompute topological properties for key thresholds
    thresholds = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    topology_presets = {}
    for th in thresholds:
        topology_presets[str(th)] = compute_network_topology(q_cols, full_matrices['edges'], 'pearson', th)

    # Package output bundle
    output_data = {
        'metadata': {
            'total_rows_in_csv': len(valid_rows) + empty_rows,
            'empty_rows_ignored': empty_rows,
            'full_graph_respondents': len(full_rows),
            'category_respondents': {c: len(cat_rows[c]) for c in ['T', 'E', 'S', 'V']},
            'categories': CATEGORY_NAMES,
            'category_colors': CATEGORY_COLORS,
            'total_questions': len(q_cols)
        },
        'questions': [
            {
                'id': q['id'],
                'text': q['text'],
                'cat': q['cat'],
                'cat_name': q['cat_name'],
                'color': q['color'],
                'stats_full': full_stats[q['id']],
                'stats_cat': cat_stats[q['cat']][q['id']]
            }
            for q in q_cols
        ],
        'full_graph': {
            'n_respondents': len(full_rows),
            'edges': full_matrices['edges'],
            'pearson_matrix': full_matrices['pearson'],
            'agreement_matrix': full_matrices['agreement'],
            'cosine_matrix': full_matrices['cosine'],
            'topology_presets': topology_presets
        },
        'category_graphs': {
            c: {
                'cat': c,
                'cat_name': CATEGORY_NAMES[c],
                'n_respondents': len(cat_rows[c]),
                'questions': [q['id'] for q in col_by_cat[c]],
                'edges': cat_matrices[c]['edges'],
                'pearson_matrix': cat_matrices[c]['pearson'],
                'agreement_matrix': cat_matrices[c]['agreement'],
                'cosine_matrix': cat_matrices[c]['cosine'],
                'topology_default': compute_network_topology(col_by_cat[c], cat_matrices[c]['edges'], 'pearson', 0.35)
            }
            for c in ['T', 'E', 'S', 'V']
        }
    }

    os.makedirs('public', exist_ok=True)
    out_file = os.path.join('public', 'survey_network_data.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully generated dataset export at {out_file} ({os.path.getsize(out_file)/1024:.1f} KB)")


if __name__ == '__main__':
    main()
