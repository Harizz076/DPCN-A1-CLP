#!/usr/bin/env python3

import os
import json
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.cluster.vq import kmeans2

FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# 1. Load preprocessed JSON data
with open('public/survey_network_data.json', 'r') as f:
    data = json.load(f)

questions = data['questions']
q_map = {q['id']: q for q in questions}
q_ids = [q['id'] for q in questions]
edges = data['full_graph']['edges']
pearson_mat = np.array(data['full_graph']['pearson_matrix'])

CAT_ORDER = ['T', 'E', 'S', 'V']
CAT_NAMES = {
    'T': 'Technology',
    'E': 'Education',
    'S': 'Ethics & Society',
    'V': 'Environment'
}
CAT_COLORS = {
    'T': '#0284c7',
    'E': '#d97706',
    'S': '#7c3aed',
    'V': '#059669'
}

plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#cbd5e1',
    'axes.linewidth': 0.8
})


# ==============================================================================
# 1. Heatmap: Domain-by-Domain & Question Correlation Matrix
# ==============================================================================
def generate_correlation_heatmaps():
    print("Generating correlation heatmaps...")

    # Compute 4x4 Category Mean Correlation Matrix
    cat_means = np.zeros((4, 4))
    cat_indices = {c: [i for i, q in enumerate(questions) if q['cat'] == c] for c in CAT_ORDER}

    for i, c1 in enumerate(CAT_ORDER):
        for j, c2 in enumerate(CAT_ORDER):
            idx1 = cat_indices[c1]
            idx2 = cat_indices[c2]
            sub_mat = pearson_mat[np.ix_(idx1, idx2)]
            if i == j:
                vals = sub_mat[np.triu_indices(len(idx1), k=1)]
            else:
                vals = sub_mat.flatten()
            cat_means[i, j] = np.mean(vals)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=300)

    # Panel 1: Full 60x60 matrix with domain boundaries
    im1 = ax1.imshow(pearson_mat, cmap='coolwarm', vmin=-0.4, vmax=0.6, origin='upper')
    ax1.set_box_aspect(1)
    
    # Draw boundary lines separating categories (every 15 questions)
    for boundary in [15, 30, 45]:
        ax1.axhline(boundary - 0.5, color='#0f172a', linewidth=1.2, linestyle='-')
        ax1.axvline(boundary - 0.5, color='#0f172a', linewidth=1.2, linestyle='-')

    ax1.set_xticks([7, 22, 37, 52])
    ax1.set_xticklabels(['Technology (T)', 'Education (E)', 'Ethics (S)', 'Environment (V)'],
                        fontsize=9.5, fontweight='bold', color='#1e293b')
    ax1.set_yticks([7, 22, 37, 52])
    ax1.set_yticklabels(['Technology (T)', 'Education (E)', 'Ethics (S)', 'Environment (V)'],
                        fontsize=9.5, fontweight='bold', color='#1e293b')
    ax1.set_title("Question Correlation Matrix (60 Questions)", fontsize=11.5, fontweight='bold', pad=12)

    # Uniform colorbar for Panel 1
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.15)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('Pearson Correlation (r)', fontsize=9.5, fontweight='bold')
    cbar1.ax.tick_params(labelsize=8.5)

    # Panel 2: 4x4 Category-to-Category Mean Correlation Matrix
    im2 = ax2.imshow(cat_means, cmap='Blues', vmin=0.0, vmax=0.35)
    ax2.set_box_aspect(1)
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels([f"{c} ({CAT_NAMES[c]})" for c in CAT_ORDER], fontsize=9.5, fontweight='bold', rotation=25, ha='right')
    ax2.set_yticklabels([f"{c} ({CAT_NAMES[c]})" for c in CAT_ORDER], fontsize=9.5, fontweight='bold')
    
    for i in range(4):
        for j in range(4):
            val = cat_means[i, j]
            text_color = '#ffffff' if val > 0.18 else '#0f172a'
            ax2.text(j, i, f"{val:.3f}", ha='center', va='center',
                     fontsize=10.5, fontweight='bold', color=text_color)

    # Uniform colorbar for Panel 2
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.15)
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('Mean Pairwise Pearson r', fontsize=9.5, fontweight='bold')
    cbar2.ax.tick_params(labelsize=8.5)
    ax2.set_title("Mean Correlation Between & Within Domains", fontsize=11.5, fontweight='bold', pad=12)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig6_correlation_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")
    return cat_means


# ==============================================================================
# 2. Community Detection on Question Network (tau = 0.35)
# ==============================================================================
def analyze_question_communities():
    print("Analyzing question communities...")
    tau = 0.35
    G = nx.Graph()
    for q in questions:
        G.add_node(q['id'], cat=q['cat'], text=q['text'])
    for e in edges:
        if e['pearson'] >= tau:
            G.add_edge(e['source'], e['target'], weight=e['pearson'])

    # Find communities via greedy modularity
    communities = list(nx.community.greedy_modularity_communities(G))
    communities = sorted(communities, key=len, reverse=True)

    print(f"Identified {len(communities)} communities. Sizes: {[len(c) for c in communities]}")

    comm_data = []
    for idx, c in enumerate(communities):
        nodes_in_comm = sorted(list(c))
        cat_dist = {cat: sum(1 for n in nodes_in_comm if n[0] == cat) for cat in CAT_ORDER}
        internal_edges = G.subgraph(nodes_in_comm).number_of_edges()
        comm_data.append({
            'community_id': idx + 1,
            'size': len(nodes_in_comm),
            'nodes': nodes_in_comm,
            'category_breakdown': cat_dist,
            'internal_edges': internal_edges
        })
        print(f"Community {idx + 1} (Size {len(nodes_in_comm)}): Categories {cat_dist}")
        print(f"  Nodes: {nodes_in_comm}")

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)

    main_comms = [c for c in comm_data if c['size'] >= 3]
    comm_names = [f"Group {c['community_id']}\n(N={c['size']})" for c in main_comms]
    
    bottoms = np.zeros(len(main_comms))
    for cat in CAT_ORDER:
        counts = [c['category_breakdown'][cat] for c in main_comms]
        ax.bar(comm_names, counts, bottom=bottoms, label=f"{CAT_NAMES[cat]} ({cat})",
               color=CAT_COLORS[cat], width=0.52, edgecolor='#ffffff')
        bottoms += np.array(counts)

    ax.set_ylabel('Number of Questions', fontsize=10, fontweight='bold', color='#1e293b')
    ax.set_title("Thematic Question Groups Identified by Modularity Optimization (τ = 0.35)",
                 fontsize=11, fontweight='bold', pad=12, color='#0f172a')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cbd5e1', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig7_question_communities.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")
    return comm_data


# ==============================================================================
# 3. Category-by-Category Internal Topology & Sub-clusters
# ==============================================================================
def analyze_category_subnetworks():
    print("Analyzing category-specific sub-networks...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), dpi=300)
    axes = axes.flatten()

    sub_results = {}
    tau = 0.35

    for idx, cat in enumerate(CAT_ORDER):
        ax = axes[idx]
        cat_q_ids = [q['id'] for q in questions if q['cat'] == cat]
        
        G_cat = nx.Graph()
        for qid in cat_q_ids:
            G_cat.add_node(qid)
        
        for e in edges:
            if e['source'][0] == cat and e['target'][0] == cat and e['pearson'] >= tau:
                G_cat.add_edge(e['source'], e['target'], weight=e['pearson'])

        sub_pos = nx.spring_layout(G_cat, k=0.45, seed=42)
        degs = dict(G_cat.degree())
        node_sizes = [220 + degs[n] * 40 for n in G_cat.nodes()]

        nx.draw_networkx_edges(G_cat, sub_pos, ax=ax, edge_color=CAT_COLORS[cat],
                               alpha=0.55, width=1.5)
        nx.draw_networkx_nodes(G_cat, sub_pos, ax=ax, node_size=node_sizes,
                               node_color=CAT_COLORS[cat], edgecolors='#ffffff', linewidths=1.5)
        nx.draw_networkx_labels(G_cat, sub_pos, labels={n: n for n in G_cat.nodes()},
                                ax=ax, font_size=8, font_weight='bold', font_color='#ffffff')

        density = nx.density(G_cat)
        comps = list(nx.connected_components(G_cat))
        giant_size = len(max(comps, key=len)) if comps else 0

        sub_results[cat] = {
            'edges': G_cat.number_of_edges(),
            'density': round(density, 3),
            'giant_size': giant_size,
            'isolated_nodes': [n for n, d in degs.items() if d == 0]
        }

        ax.set_title(f"{CAT_NAMES[cat]} ({cat})\nEdges: {G_cat.number_of_edges()}/105 | Density: {density:.3f} | Giant Comp: {giant_size}/15",
                     fontsize=10, fontweight='bold', color='#0f172a', pad=8)
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig8_category_subnetworks.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")
    return sub_results


# ==============================================================================
# 4. Respondent Clustering (Student Personas)
# ==============================================================================
def analyze_respondent_clustering():
    print("Analyzing respondent groupings (student personas)...")

    csv_path = 'Survey_Results_UC.csv'
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        headers = next(reader)
        raw_rows = list(reader)

    likert = {
        'Strongly Disagree': 1,
        'Disagree': 2,
        'Neutral': 3,
        'No Comments': 3,
        'Agree': 4,
        'Strongly Agree': 5
    }

    q_col_map = []
    for col_idx, h in enumerate(headers):
        clean_h = h.strip('\"').strip()
        if clean_h.startswith(('T', 'E', 'S', 'V')) and len(clean_h) >= 3 and clean_h[1:3].isdigit():
            q_col_map.append((clean_h.split('.')[0].strip(), col_idx))

    complete_resp = []
    for r in raw_rows:
        if all(r[col].strip() != '' for _, col in q_col_map):
            row_vals = [likert.get(r[col].strip(), 3) for _, col in q_col_map]
            complete_resp.append(row_vals)

    X = np.array(complete_resp, dtype=float)  # 85 respondents x 60 questions
    print(f"Clustering {X.shape[0]} respondents across {X.shape[1]} questions...")

    # Deterministic k-means via scipy
    np.random.seed(42)
    centroids, clusters = kmeans2(X, 3, minit='points', iter=30)

    cat_indices = {c: [i for i, (qid, _) in enumerate(q_col_map) if qid[0] == c] for c in CAT_ORDER}
    cluster_profiles = []

    for c_id in range(3):
        c_mask = (clusters == c_id)
        c_size = np.sum(c_mask)
        means_by_cat = {c: float(np.mean(X[c_mask][:, cat_indices[c]])) for c in CAT_ORDER}
        overall_mean = float(np.mean(X[c_mask]))
        cluster_profiles.append({
            'cluster_id': c_id + 1,
            'size': int(c_size),
            'percentage': round(float(c_size) / len(X) * 100, 1),
            'overall_mean': round(overall_mean, 2),
            'category_means': {k: round(v, 2) for k, v in means_by_cat.items()}
        })
        print(f"Respondent Cluster {c_id + 1} (N={c_size}, {c_size/len(X)*100:.1f}%): {means_by_cat}")

    # Plot respondent cluster profiles
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    bar_width = 0.22
    x = np.arange(len(CAT_ORDER))
    
    palette = ['#059669', '#3b82f6', '#8b5cf6']
    for c_id in range(3):
        profile = cluster_profiles[c_id]
        vals = [profile['category_means'][c] for c in CAT_ORDER]
        ax.bar(x + (c_id - 1) * bar_width, vals, width=bar_width,
               label=f"Cluster {c_id + 1} (N={profile['size']}, {profile['percentage']}%)",
               color=palette[c_id], edgecolor='#ffffff')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} ({CAT_NAMES[c]})" for c in CAT_ORDER], fontsize=9.5, fontweight='bold')
    ax.set_ylabel('Mean Likert Score (1 to 5)', fontsize=10, fontweight='bold', color='#1e293b')
    ax.set_ylim(2.5, 5.3)
    ax.set_title("Student Respondent Archetypes: Mean Domain Agreement Across Clusters",
                 fontsize=11, fontweight='bold', pad=12, color='#0f172a')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='#cbd5e1', fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig9_student_clusters_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

    return cluster_profiles


# ==============================================================================
# 5. Negative Correlations: Strongest Opposing Dyads (Excluded Trade-offs)
# ==============================================================================
def plot_negative_correlations():
    print("Generating negative correlations plot...")
    
    tradeoffs = [
        ('E03', 'E10', -0.430, 'Attendance vs. Innovation / Problem-Solving'),
        ('E03', 'T14', -0.336, 'Attendance vs. Social Media Polarization Concerns'),
        ('T03', 'E04', -0.328, 'AI Coursework Disclosure vs. Online Learning'),
        ('E01', 'E02', -0.300, 'Project-Based Learning vs. Traditional Exams'),
        ('E03', 'E04', -0.298, 'Attendance vs. Online Learning Flexibility'),
        ('E03', 'E07', -0.292, 'Attendance vs. Undergraduate Research Focus'),
        ('E03', 'S06', -0.285, 'Attendance vs. Shared Misinformation Responsibility'),
        ('T01', 'T12', -0.282, 'AI Techno-Optimism vs. Government Regulation'),
        ('E03', 'T06', -0.273, 'Attendance vs. Intelligent Automation Acceptance'),
        ('E03', 'E12', -0.271, 'Attendance vs. AI-Assisted Personalized Teaching'),
        ('E03', 'E11', -0.258, 'Attendance vs. Rapid Curriculum Modernization')
    ]
    
    # Sort from strongest negative to weakest
    tradeoffs.sort(key=lambda x: x[2])
    
    labels = [f"{t[0]} vs {t[1]}: {t[3]}" for t in tradeoffs]
    values = [t[2] for t in tradeoffs]
    colors = ['#dc2626' if 'E03' in (t[0], t[1]) else '#d97706' for t in tradeoffs]
    
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    y_pos = np.arange(len(tradeoffs))
    
    bars = ax.barh(y_pos, values, color=colors, height=0.6, edgecolor='none', alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, fontweight='bold', color='#1e293b')
    ax.invert_yaxis()
    
    ax.set_xlabel('Pearson Correlation (r)', fontsize=10, fontweight='bold', color='#334155')
    ax.set_xlim(-0.50, 0.0)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    for bar in bars:
        w = bar.get_width()
        # Put text cleanly inside the bar with white bold text
        ax.text(w + 0.015, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}",
                va='center', ha='left', fontsize=9.5, fontweight='bold', color='#ffffff')
                
    leg_handles = [
        mlines.Line2D([], [], color='#dc2626', marker='s', linestyle='None', markersize=8,
                      label='Attendance (E03) Trade-Offs'),
        mlines.Line2D([], [], color='#d97706', marker='s', linestyle='None', markersize=8,
                      label='Other Academic / Tech Trade-Offs')
    ]
    ax.legend(handles=leg_handles, loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=2, frameon=True, framealpha=0.95, edgecolor='#cbd5e1', fontsize=9)
    
    ax.set_title("Strongest Negative Correlations (r ≤ -0.25): Latent Trade-Offs",
                 fontsize=11.5, fontweight='bold', color='#0f172a', pad=12)
                 
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig10_negative_correlations.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def main():
    cat_means = generate_correlation_heatmaps()
    comm_data = analyze_question_communities()
    sub_results = analyze_category_subnetworks()
    cluster_profiles = analyze_respondent_clustering()
    plot_negative_correlations()

    summary = {
        'cat_mean_correlations': cat_means.tolist(),
        'question_communities': [
            {
                'id': c['community_id'],
                'size': c['size'],
                'nodes': c['nodes'],
                'category_breakdown': c['category_breakdown']
            } for c in comm_data
        ],
        'category_subnetworks': sub_results,
        'student_clusters': cluster_profiles
    }

    with open('public/advanced_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Advanced analysis complete. Saved summary to public/advanced_analysis_summary.json.")


if __name__ == '__main__':
    main()
