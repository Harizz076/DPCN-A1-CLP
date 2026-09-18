#!/usr/bin/env python3
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import numpy as np

FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load data
with open('public/survey_network_data.json') as f:
    data = json.load(f)

q_list = data['questions']
questions = {q['id']: q for q in q_list}
edges = data['full_graph']['edges']

CAT_COLORS = {
    'T': '#0284c7',  # Sky Blue
    'E': '#d97706',  # Amber
    'S': '#7c3aed',  # Violet
    'V': '#059669'   # Emerald
}

CAT_NAMES = {
    'T': 'Technology',
    'E': 'Education',
    'S': 'Ethics & Society',
    'V': 'Environment'
}

plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#cbd5e1',
    'axes.linewidth': 0.8,
    'grid.color': '#f1f5f9',
    'grid.linewidth': 0.8
})


def plot_fig1_network():
    tau = 0.35
    G = nx.Graph()
    for q in q_list:
        G.add_node(q['id'], cat=q['cat'])

    active_edges = []
    for e in edges:
        if e['pearson'] >= tau:
            G.add_edge(e['source'], e['target'], weight=e['pearson'])
            active_edges.append(e)

    fig, ax = plt.subplots(figsize=(13, 11), dpi=300)

    # 1. Cluster-aware layout: place category centers in 4 distinct quadrants
    # V (Environment, top-right), S (Ethics, top-left), E (Education, bottom-left), T (Tech, bottom-right)
    centers = {
        'S': np.array([-1.5,  1.3]),
        'V': np.array([ 1.5,  1.3]),
        'E': np.array([-1.5, -1.3]),
        'T': np.array([ 1.5, -1.3])
    }

    rng = np.random.default_rng(42)
    pos = {}

    # Position nodes with a tight internal spring layout per category around its center
    for cat, center in centers.items():
        cat_nodes = [n for n in G.nodes() if n[0] == cat]
        sub = G.subgraph(cat_nodes)
        sub_pos = nx.spring_layout(sub, k=0.35, iterations=80, seed=42)
        for n, p in sub_pos.items():
            pos[n] = center + p * 0.95

    # Run a few global relaxation steps to untangle cross-cluster edges while keeping cluster gravity
    for step in range(30):
        for u, v in G.edges():
            delta = pos[v] - pos[u]
            dist = np.linalg.norm(delta) or 0.01
            force = 0.015 * (dist - 1.2) * (delta / dist)
            pos[u] += force
            pos[v] -= force
        # Pull each node back toward its category center
        for n in G.nodes():
            pos[n] += 0.04 * (centers[n[0]] - pos[n])

    # 2. Draw subtle cluster background regions with category labels
    cat_boxes = {
        'S': (-2.6, 0.1, 2.3, 2.3, 'Ethics & Society (S)', CAT_COLORS['S']),
        'V': ( 0.3, 0.1, 2.4, 2.3, 'Environment (V)', CAT_COLORS['V']),
        'E': (-2.6, -2.5, 2.3, 2.3, 'Education (E)', CAT_COLORS['E']),
        'T': ( 0.3, -2.5, 2.4, 2.3, 'Technology (T)', CAT_COLORS['T'])
    }
    for cat, (bx, by, bw, bh, title, col) in cat_boxes.items():
        rect = plt.Rectangle((bx, by), bw, bh, facecolor=col, alpha=0.04,
                             edgecolor=col, linestyle='--', linewidth=1.0, zorder=0)
        ax.add_patch(rect)
        ax.text(bx + bw/2, by + bh - 0.18, title, fontsize=11, fontweight='bold',
                color=col, ha='center', va='top', zorder=1)

    # 3. Draw Edges: cross-category vs internal category
    # Cross-category edges: thin, neutral slate, low opacity
    for u, v, d in G.edges(data=True):
        if u[0] != v[0]:
            w = d['weight']
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color='#94a3b8', alpha=0.35, linewidth=0.9, linestyle='-', zorder=2)

    # Internal edges: colored by category, bolder
    for u, v, d in G.edges(data=True):
        if u[0] == v[0]:
            w = d['weight']
            col = CAT_COLORS[u[0]]
            width = 1.2 + (w - 0.35) * 3.5
            alpha = min(0.85, max(0.4, w))
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=col, alpha=alpha, linewidth=width, zorder=3)

    # 4. Draw Nodes with clear sizing by degree
    degrees = dict(G.degree())
    node_sizes = [260 + degrees[n] * 28 for n in G.nodes()]
    node_colors = [CAT_COLORS[n[0]] for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, edgecolors='#ffffff', linewidths=1.8)

    # Highlight bridge (T12) and top hubs (S09, V09, E10) with crisp dark outline
    key_nodes = ['T12', 'S09', 'V09', 'E10', 'E03']
    key_sizes = [node_sizes[list(G.nodes()).index(n)] for n in key_nodes]
    key_colors = [CAT_COLORS[n[0]] for n in key_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=key_nodes, ax=ax, node_size=key_sizes,
                           node_color=key_colors, edgecolors='#0f172a', linewidths=2.5)

    # Node labels
    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8.5,
                            font_weight='bold', font_color='#ffffff')

    # Annotation callouts for T12 bridge and E03 outlier
    ax.annotate("T12 (AI Regulation)\nMain Cross-Domain Bridge",
                xy=(pos['T12'][0], pos['T12'][1]),
                xytext=(pos['T12'][0] - 0.75, pos['T12'][1] + 0.35),
                fontsize=8.5, fontweight='bold', color='#0f172a',
                arrowprops=dict(arrowstyle='->', color='#0f172a', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff', edgecolor='#cbd5e1', alpha=0.9),
                zorder=7)

    ax.annotate("E03 (Attendance)\nIsolated Outlier",
                xy=(pos['E03'][0], pos['E03'][1]),
                xytext=(pos['E03'][0] + 0.35, pos['E03'][1] - 0.35),
                fontsize=8.5, fontweight='bold', color='#0f172a',
                arrowprops=dict(arrowstyle='->', color='#0f172a', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff', edgecolor='#cbd5e1', alpha=0.9),
                zorder=7)

    # Clean bottom legend
    legend_handles = [
        mlines.Line2D([], [], color=CAT_COLORS[c], marker='o', markersize=9,
                      linestyle='None', label=f"{CAT_NAMES[c]} ({c})")
        for c in ['T', 'E', 'S', 'V']
    ]
    legend_handles.extend([
        mlines.Line2D([], [], color='#64748b', linewidth=2.0, label='Internal Domain Edge'),
        mlines.Line2D([], [], color='#94a3b8', linewidth=1.0, linestyle='-', label='Cross-Domain Edge (r ≥ 0.35)')
    ])
    ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=6, frameon=True, framealpha=0.95, edgecolor='#cbd5e1', fontsize=8.5)

    ax.set_title("Question-Similarity Opinion Network (r ≥ 0.35)\nGrouped by Survey Category",
                 fontsize=13, fontweight='bold', pad=12, color='#0f172a')
    ax.axis('off')
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig1_opinion_network.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_fig2_percolation():
    tau_vals = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    edge_counts = []
    giant_sizes = []
    clust_coeffs = []
    modularities = []

    for th in tau_vals:
        G = nx.Graph()
        for q in q_list:
            G.add_node(q['id'])
        for e in edges:
            if e['pearson'] >= th:
                G.add_edge(e['source'], e['target'])

        comps = list(nx.connected_components(G))
        giant = max(comps, key=len) if comps else set()
        edge_counts.append(G.number_of_edges())
        giant_sizes.append(len(giant))
        clust_coeffs.append(nx.average_clustering(G))
        try:
            c = nx.community.greedy_modularity_communities(G)
            m = nx.community.modularity(G, c)
        except:
            m = 0.0
        modularities.append(m)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Panel 1: Edges & Giant Component
    color_e = '#0284c7'
    color_g = '#059669'

    line1 = ax1.plot(tau_vals, edge_counts, marker='o', color=color_e, linewidth=2, label='Active Edges |E|')
    ax1.set_xlabel('Correlation Threshold (τ)', fontsize=10, fontweight='bold', color='#334155')
    ax1.set_ylabel('Active Edges Count', fontsize=10, fontweight='bold', color=color_e)
    ax1.tick_params(axis='y', labelcolor=color_e)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(tau_vals, giant_sizes, marker='s', color=color_g, linewidth=2, linestyle='--', label='Giant Component Size')
    ax1_twin.set_ylabel('Giant Component Size (Max 60)', fontsize=10, fontweight='bold', color=color_g)
    ax1_twin.tick_params(axis='y', labelcolor=color_g)
    ax1_twin.set_ylim(0, 65)

    # Highlight tau = 0.35 (line only, no label)
    ax1.axvline(x=0.35, color='#ef4444', linestyle=':', linewidth=1.8)
    ax1.set_title('Network Connectivity vs. Threshold', fontsize=11, fontweight='bold', color='#0f172a', pad=10)

    # Legend for Panel 1 on the top-right (completely empty region, no overlap)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cbd5e1', fontsize=8.5)

    # Panel 2: Clustering & Modularity
    color_c = '#7c3aed'
    color_m = '#d97706'

    line_c = ax2.plot(tau_vals, clust_coeffs, marker='^', color=color_c, linewidth=2, label='Clustering Coefficient (C)')
    line_m = ax2.plot(tau_vals, modularities, marker='d', color=color_m, linewidth=2, label='Modularity (Q)')
    ax2.axvline(x=0.35, color='#ef4444', linestyle=':', linewidth=1.8)

    ax2.set_xlabel('Correlation Threshold (τ)', fontsize=10, fontweight='bold', color='#334155')
    ax2.set_ylabel('Metric Value [0, 1]', fontsize=10, fontweight='bold', color='#334155')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title('Clustering (C) and Modularity (Q)', fontsize=11, fontweight='bold', color='#0f172a', pad=10)

    # Legend in the upper-right corner like the first plot, with extra y headroom so it sits above the data
    ax2.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cbd5e1', fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig2_percolation_analysis.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_fig3_cohesion():
    categories = ['Environment (V)', 'Ethics & Society (S)', 'Education (E)', 'Cross-Domain', 'Technology (T)']
    densities = [0.457, 0.229, 0.114, 0.078, 0.038]
    edge_counts = [48, 24, 12, 105, 4]
    possible = [105, 105, 105, 1350, 105]
    colors = ['#059669', '#7c3aed', '#d97706', '#64748b', '#0284c7']

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    y_pos = np.arange(len(categories))

    bars = ax.barh(y_pos, densities, color=colors, height=0.55, edgecolor='none')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=9.5, fontweight='bold', color='#1e293b')
    ax.invert_yaxis()
    ax.set_xlabel('Connection Density (Edges / Possible Pairs)', fontsize=10, fontweight='bold', color='#334155')
    ax.set_xlim(0, 0.55)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.008, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}",
                va='center', fontsize=9.5, fontweight='bold', color='#0f172a')

    ax.set_title("Domain Cohesion: Internal vs. Cross-Domain Density",
                 fontsize=12, fontweight='bold', color='#0f172a', pad=12)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig3_domain_cohesion.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_fig4_centrality():
    tau = 0.35
    G = nx.Graph()
    for q in q_list:
        G.add_node(q['id'], cat=q['cat'])
    for e in edges:
        if e['pearson'] >= tau:
            G.add_edge(e['source'], e['target'])

    deg = dict(G.degree())
    btw = nx.betweenness_centrality(G)

    deg_thresh = 10
    btw_thresh = 0.035

    # Key nodes to call out. Everything else is plotted as a neutral
    # "background" point so the handful of nodes that matter for the
    # narrative actually stand out.
    key_nodes = {
        'T12': 'T12 (AI Regulation)',
        'S13': 'S13',
        'E10': 'E10 (Innovation)',
        'S14': 'S14 (Ethics)',
        'V09': 'V09 (Reuse)',
        'S09': 'S09 (Dialogue)',
        'V02': 'V02',
        'E03': 'E03 (Attendance)',
    }

    fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=300)

    # 1. Background points: every question NOT called out is plotted in a
    # single neutral grey so it reads as "not significant" and doesn't
    # compete visually with the highlighted nodes.
    bg_x = [deg[q['id']] for q in q_list if q['id'] not in key_nodes]
    bg_y = [btw[q['id']] for q in q_list if q['id'] not in key_nodes]
    ax.scatter(bg_x, bg_y, color='#9ca3af', s=55, alpha=0.55,
               edgecolors='#ffffff', linewidths=0.6, zorder=2, label='_nolegend_')

    # Quadrant dividing guidelines
    ax.axvline(x=deg_thresh, color='#cbd5e1', linestyle='--', linewidth=1.0, zorder=1)
    ax.axhline(y=btw_thresh, color='#cbd5e1', linestyle='--', linewidth=1.0, zorder=1)

    # Give the axes some breathing room around the data so corner points
    # (e.g. an isolated node sitting at exactly 0,0) never sit flush against
    # a quadrant tag or the plot border. Extra padding at the bottom in
    # particular, since that's where a peripheral node tends to sit right on
    # top of the "Peripheral" quadrant tag otherwise.
    xmax = max(deg.values())
    ymax = max(btw.values())
    x_pad = max(xmax * 0.12, 0.8)
    y_pad = max(ymax * 0.12, 0.01)
    ax.set_xlim(-x_pad, xmax + x_pad)
    ax.set_ylim(-y_pad * 1.8, ymax + y_pad * 1.6)

    # Short quadrant tags anchored to axes fractions (0-1), not data
    # coordinates, so they always sit inset from the true corners regardless
    # of axis limits and never collide with data points or each other. Full
    # descriptions live in the caption below the plot instead of cluttering
    # the plot itself. The bottom two are pulled down close to the axis
    # floor (rather than sitting right where a low-degree/low-betweenness
    # point usually lands) now that the extra bottom padding above gives
    # them room to breathe.
    quadrant_labels = [
        (0.03, 0.95, 'Bridge Nodes', 'left', 'top'),
        (0.97, 0.95, 'Consensus Hubs', 'right', 'top'),
        (0.03, 0.015, 'Peripheral', 'left', 'bottom'),
        (0.97, 0.015, 'Intra-Domain Hubs', 'right', 'bottom'),
    ]
    for x, y, txt, ha, va in quadrant_labels:
        ax.text(x, y, txt, transform=ax.transAxes, fontsize=8.5, color='#94a3b8',
                fontweight='bold', style='italic', ha=ha, va=va, zorder=2)

    # 2. Highlighted points: color-coded by category, numbered, with a dark
    # outline so they pop out of the grey background.
    texts = []
    for i, (qid, label) in enumerate(key_nodes.items(), start=1):
        if qid not in deg:
            continue
        x, y = deg[qid], btw[qid]
        col = CAT_COLORS[qid[0]]
        ax.scatter(x, y, s=140, color=col, edgecolors='#0f172a', linewidths=1.6,
                   alpha=0.95, zorder=4)
        # Dark text with a white halo stays legible whether adjustText leaves
        # it sitting on top of the colored marker or nudges it off to the
        # side onto the plain white/grey background to dodge an overlap.
        t = ax.text(x, y, str(i), fontsize=9, fontweight='bold',
                     color='#0f172a', ha='center', va='center', zorder=6)
        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground='#ffffff')])
        texts.append(t)

    # Automatically resolve label overlaps (numbers only now, so the plot
    # itself stays uncluttered) and draw a thin leader line from each number
    # back to its actual point. NOTE: these are adjustText >= 1.1 parameter
    # names (expand / force_text / force_static / force_explode / max_move).
    # Older adjustText releases used expand_points/expand_text/force_points
    # instead -- check your installed version if labels don't move.
    adjust_text(
        texts, ax=ax,
        expand=(1.8, 2.1),
        force_text=(1.0, 1.2),
        force_static=(0.5, 0.6),
        force_explode=(0.5, 0.7),
        max_move=(80, 80),
        arrowprops=dict(arrowstyle='-', color='#0f172a', lw=0.8, alpha=0.6,
                         shrinkA=6, shrinkB=6),
    )

    ax.set_xlabel('Degree Centrality (Number of Correlated Neighbors at r ≥ 0.35)', fontsize=10, fontweight='bold', color='#334155')
    ax.set_ylabel('Betweenness Centrality', fontsize=10, fontweight='bold', color='#334155')
    ax.grid(True, linestyle='--', alpha=0.35, zorder=0)

    # 3. The question each number refers to moves entirely off the plot into
    # a legend on the right, so the plot area itself only ever shows small
    # numbered markers. Legend swatches are colored the same as their point
    # (category color) so the legend also does the job of a color key.
    legend_handles = [
        mlines.Line2D([], [], marker='o', color=CAT_COLORS[qid[0]], linestyle='None',
                      markersize=9, markeredgecolor='#0f172a', markeredgewidth=1.0,
                      label=f"{i}. {label}")
        for i, (qid, label) in enumerate(key_nodes.items(), start=1) if qid in deg
    ]
    legend_handles.append(
        mlines.Line2D([], [], marker='o', color='#9ca3af', linestyle='None',
                      markersize=8, label='Other questions (not significant)')
    )
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=True, framealpha=0.95, edgecolor='#cbd5e1', fontsize=9,
              title='Question', title_fontsize=9.5, borderaxespad=0.0)

    ax.set_title("Degree Centrality vs. Betweenness Centrality (τ = 0.35)\nIdentifying Opinion Hubs and the T12 Bridge Node",
                 fontsize=12, fontweight='bold', color='#0f172a', pad=14)

    # One-line caption spelling out what each quadrant tag means, instead of
    # cramming the full description onto the plot itself.
    fig.text(0.42, -0.02,
              f"Quadrants split at degree = {deg_thresh}, betweenness = {btw_thresh}: "
              "Bridge Nodes connect domains without being locally popular; Consensus Hubs do both; "
              "Intra-Domain Hubs are popular within one domain; Peripheral nodes are weakly connected.",
              ha='center', va='top', fontsize=8, color='#64748b', wrap=True)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig4_centrality_quadrant.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_fig5_consensus():
    means = [q['stats_full']['mean'] for q in q_list]
    vars_ = [q['stats_full']['variance'] for q in q_list]
    cats = [q['cat'] for q in q_list]
    colors = [CAT_COLORS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Shaded regions
    ax.axvspan(4.3, 5.05, ymin=0.0, ymax=0.35, color='#10b981', alpha=0.1, label='High Consensus Zone')
    ax.axvspan(1.8, 3.8, ymin=0.65, ymax=1.0, color='#ef4444', alpha=0.08, label='High Polarization / Controversy')

    ax.scatter(means, vars_, c=colors, s=60, alpha=0.85, edgecolors='#ffffff', linewidths=0.8)

    callouts = {
        'E15': (8, 0, 'E15 (Continuous Learning)'),
        'V13': (8, 0, 'V13 (Corporate Environmental Liability)'),
        'S05': (8, 0, 'S05 (Personal Data Control)'),
        'E03': (8, 0, 'E03 (Compulsory Attendance: Mean 2.02)'),
        'E04': (8, 0, 'E04 (Online Learning: Var 1.37)'),
        'E02': (8, 0, 'E02 (Traditional Exams: Mean 2.81)'),
        'T08': (8, 0, 'T08 (AI Medical Diagnosis: Var 1.21)')
    }

    for q in q_list:
        qid = q['id']
        if qid in callouts:
            dx, dy, text = callouts[qid]
            m = q['stats_full']['mean']
            v = q['stats_full']['variance']
            ax.scatter(m, v, s=95, facecolors='none', edgecolors='#0f172a', linewidths=1.5, zorder=5)
            ax.annotate(text, (m, v), xytext=(m + 0.05, v + 0.02),
                        fontsize=8.5, fontweight='bold', color='#0f172a')

    ax.set_xlabel('Mean Agreement Score (1: Strongly Disagree → 5: Strongly Agree)', fontsize=10, fontweight='bold', color='#334155')
    ax.set_ylabel('Opinion Variance (σ²)', fontsize=10, fontweight='bold', color='#334155')
    ax.set_xlim(1.8, 5.05)
    ax.set_ylim(0.15, 1.55)
    ax.grid(True, linestyle='--', alpha=0.5)

    legend_handles = [
        mlines.Line2D([], [], color='white', marker='o', markersize=8,
                      markerfacecolor=CAT_COLORS[c], markeredgecolor='white',
                      label=CAT_NAMES[c])
        for c in ['T', 'E', 'S', 'V']
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=True, fontsize=8.5)

    ax.set_title("Student Opinion Distribution: Consensus vs. Polarization Spectrum",
                 fontsize=12, fontweight='bold', color='#0f172a', pad=12)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'fig5_consensus_vs_polarization.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    plot_fig1_network()
    plot_fig2_percolation()
    plot_fig3_cohesion()
    plot_fig4_centrality()
    plot_fig5_consensus()
    print("All 5 clean plots generated successfully.")