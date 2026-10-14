# -*- coding: utf-8 -*-
"""
test_harness/plotting.py
~~~~~~~~~~~~~~~~~~~~~~~~
Auto-generates circuit visualization plots from saved network state.
Pure matplotlib — no external dependencies beyond what GeneNet already uses.
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_circuit_graph(W, slot_winners, confidences, output_path,
                       threshold=0.001):
    """Draw the gene regulatory network as a directed graph.

    Nodes are placed in a circle, labeled with protein names.
    Edges are colored green (activation) or red (repression),
    with width proportional to |weight|.

    Parameters
    ----------
    W : numpy.ndarray
        Weight matrix (S x S). W[i][j] = influence of gene j on gene i.
    slot_winners : list of str
        Protein name assigned to each slot.
    confidences : list of float
        DARTS confidence (%) for each slot.
    output_path : str
        Where to save the PNG.
    threshold : float
        Minimum |weight| to draw an edge.
    """
    S = W.shape[0]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect('equal')
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.axis('off')
    ax.set_title('Gene Regulatory Network', fontsize=16, fontweight='bold',
                 pad=20)

    # --- Node positions in a circle ---
    radius = 1.2
    angles = [2.0 * math.pi * i / S - math.pi / 2 for i in range(S)]
    positions = [(radius * math.cos(a), radius * math.sin(a)) for a in angles]

    # --- Draw edges first (so nodes are on top) ---
    max_w = max(np.max(np.abs(W)), 1e-6)

    for i in range(S):
        for j in range(S):
            w = W[i][j]
            if abs(w) < threshold:
                continue
            if i == j:
                continue  # skip self-loops for clarity

            x_src, y_src = positions[j]
            x_dst, y_dst = positions[i]

            # Shorten arrow to not overlap with node circles
            dx = x_dst - x_src
            dy = y_dst - y_src
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-6:
                continue
            node_r = 0.18
            # Start and end points offset by node radius
            x0 = x_src + (dx / dist) * node_r
            y0 = y_src + (dy / dist) * node_r
            x1 = x_dst - (dx / dist) * (node_r + 0.06)
            y1 = y_dst - (dy / dist) * (node_r + 0.06)

            color = '#e74c3c'  # Todas as conexões neste modelo são repressões
            lw = 1.0 + 3.0 * (abs(w) / max_w)
            alpha = 0.4 + 0.6 * (abs(w) / max_w)

            ax.annotate(
                '',
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='-|>,head_width=0.3,head_length=0.15',
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    connectionstyle='arc3,rad=0.15'
                )
            )
            # Weight label at midpoint
            mx = (x0 + x1) / 2 + 0.08
            my = (y0 + y1) / 2 + 0.08
            ax.text(mx, my, '{:.2f}'.format(w), fontsize=7, color=color,
                    ha='center', va='center', alpha=0.8,
                    fontweight='bold')

    # --- Draw nodes ---
    for i in range(S):
        x, y = positions[i]
        circle = plt.Circle((x, y), 0.18, color='#3498db', ec='#2c3e50',
                             lw=2, zorder=10)
        ax.add_patch(circle)

        # Protein name inside node
        name = slot_winners[i] if i < len(slot_winners) else 'Slot {}'.format(i)
        ax.text(x, y + 0.02, name, fontsize=9, fontweight='bold',
                ha='center', va='center', color='white', zorder=11)

        # Slot number and confidence below node
        conf = confidences[i] if i < len(confidences) else 0.0
        label = 'Slot {} ({:.0f}%)'.format(i, conf)
        ax.text(x, y - 0.30, label, fontsize=8, ha='center', va='center',
                color='#2c3e50')

    rep_patch = mpatches.FancyArrow(0, 0, 0.1, 0, width=0.02,
                                     color='#e74c3c')
    ax.legend(
        [mpatches.Patch(color='#e74c3c')],
        [u'Repressao (-)'],
        loc='lower right', fontsize=10, framealpha=0.9
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_wiring_heatmap(W, slot_winners, output_path):
    """Draw a heatmap of the wiring matrix W with protein labels.

    Parameters
    ----------
    W : numpy.ndarray
        Weight matrix (S x S).
    slot_winners : list of str
        Protein name for each slot.
    output_path : str
        Where to save the PNG.
    """
    S = W.shape[0]

    fig, ax = plt.subplots(figsize=(7, 6))

    vmax = max(np.max(np.abs(W)), 1e-6)
    im = ax.imshow(W, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   interpolation='nearest')

    labels = []
    for i in range(S):
        name = slot_winners[i] if i < len(slot_winners) else 'Slot {}'.format(i)
        labels.append('{} (S{})'.format(name, i))

    ax.set_xticks(range(S))
    ax.set_yticks(range(S))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate each cell with weight value
    for i in range(S):
        for j in range(S):
            val = W[i][j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, '{:.2f}'.format(val), ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    ax.set_title('Wiring Matrix (W)\nW[i][j] = influence of gene j on gene i',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Source gene (j)', fontsize=10)
    ax.set_ylabel('Target gene (i)', fontsize=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight', fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_circuit_plots(run_dir, network_state, circuit_design):
    """Generate all circuit visualization plots for a run.

    Called automatically by TestRun.save().

    Parameters
    ----------
    run_dir : str
        Path to the run directory.
    network_state : dict
        Dictionary with numpy arrays (from network.parameters).
    circuit_design : dict or None
        Circuit design dict with 'slots' key.

    Returns
    -------
    list of str
        Paths to generated plot files.
    """
    W = network_state.get('W', None)
    if W is None:
        return []
    
    # Apply softplus to convert weights to true biological force for plotting
    W = np.log(1.0 + np.exp(W))

    # Extract slot winners and confidences
    if circuit_design and 'slots' in circuit_design:
        slot_winners = [s['winner'] for s in circuit_design['slots']]
        confidences = [s['confidence_pct'] for s in circuit_design['slots']]
    else:
        slot_winners = ['Slot {}'.format(i) for i in range(W.shape[0])]
        confidences = [0.0] * W.shape[0]

    generated = []

    # 1. Network graph
    graph_path = os.path.join(run_dir, 'circuit_graph.png')
    plot_circuit_graph(W, slot_winners, confidences, graph_path)
    generated.append(graph_path)

    # 2. Wiring heatmap
    heatmap_path = os.path.join(run_dir, 'wiring_heatmap.png')
    plot_wiring_heatmap(W, slot_winners, heatmap_path)
    generated.append(heatmap_path)

    return generated
