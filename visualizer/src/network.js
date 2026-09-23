import * as d3 from 'd3';

export class OpinionNetworkGraph {
  constructor(containerSvgId, tooltipId, onNodeSelect, onMetricsUpdate) {
    this.svg = d3.select(containerSvgId);
    this.tooltip = d3.select(tooltipId);
    this.onNodeSelect = onNodeSelect;
    this.onMetricsUpdate = onMetricsUpdate;

    this.data = null;
    this.activeCategory = 'ALL';
    this.activeMetric = 'pearson';
    this.threshold = 0.35;
    this.keepMst = false;
    this.nodeSizeMode = 'degree';
    this.colorMode = 'category';
    this.isFrozen = false;
    this.showLabels = true;
    this.selectedNodeId = null;
    this.searchQuery = '';

    // Active graph data
    this.activeNodes = [];
    this.activeLinks = [];
    this.nodeMap = new Map();
    this.communityMap = new Map();

    this.initSvg();
  }

  initSvg() {
    this.width = this.svg.node().parentElement.clientWidth || 800;
    this.height = this.svg.node().parentElement.clientHeight || 600;

    this.svg
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', [0, 0, this.width, this.height]);

    // Background click to clear selection
    this.svg.on('click', (event) => {
      if (event.target.tagName === 'svg' || event.target.id === 'network-svg') {
        this.selectNode(null);
      }
    });

    // Root group for zooming
    this.g = this.svg.append('g').attr('class', 'network-root-group');

    // Layers
    this.linkLayer = this.g.append('g').attr('class', 'links-layer');
    this.nodeLayer = this.g.append('g').attr('class', 'nodes-layer');

    // Zoom behavior
    this.zoom = d3.zoom()
      .scaleExtent([0.2, 5])
      .on('zoom', (event) => {
        this.g.attr('transform', event.transform);
      });

    this.svg.call(this.zoom);

    // Resize listener
    window.addEventListener('resize', () => {
      this.width = this.svg.node().parentElement.clientWidth || 800;
      this.height = this.svg.node().parentElement.clientHeight || 600;
      if (this.simulation) {
        this.simulation.force('center', d3.forceCenter(this.width / 2, this.height / 2));
        this.simulation.alpha(0.2).restart();
      }
    });
  }

  resetZoom() {
    this.svg.transition().duration(500).call(
      this.zoom.transform,
      d3.zoomIdentity.translate(0, 0).scale(1)
    );
  }

  setData(data) {
    this.data = data;
    this.updateGraph();
  }

  setCategory(category) {
    this.activeCategory = category;
    this.selectedNodeId = null;
    this.updateGraph();
  }

  setMetric(metric) {
    this.activeMetric = metric;
    this.updateGraph();
  }

  setThreshold(threshold) {
    this.threshold = threshold;
    this.updateGraph();
  }

  setMstGuarantee(val) {
    this.keepMst = val;
    this.updateGraph();
  }

  setNodeSizeMode(mode) {
    this.nodeSizeMode = mode;
    this.updateNodeStyles();
  }

  setColorMode(mode) {
    this.colorMode = mode;
    this.updateNodeStyles();
  }

  setFrozen(isFrozen) {
    this.isFrozen = isFrozen;
    if (this.simulation) {
      if (isFrozen) {
        this.simulation.stop();
      } else {
        this.simulation.alpha(0.3).restart();
      }
    }
  }

  setShowLabels(show) {
    this.showLabels = show;
    this.nodeLayer.selectAll('.node-label').style('display', show ? 'block' : 'none');
  }

  setSearchQuery(query) {
    this.searchQuery = (query || '').trim().toLowerCase();
    this.highlightSearch();
  }

  updateGraph() {
    if (!this.data) return;

    // 1. Filter Nodes based on activeCategory
    let rawNodes = this.data.questions;
    if (this.activeCategory !== 'ALL') {
      rawNodes = rawNodes.filter(q => q.cat === this.activeCategory);
    }

    // Clone nodes so simulation maintains local x, y, vx, vy
    const existingNodeMap = new Map(this.activeNodes.map(d => [d.id, d]));
    this.activeNodes = rawNodes.map(q => {
      const existing = existingNodeMap.get(q.id);
      return {
        ...q,
        x: existing ? existing.x : this.width / 2 + (Math.random() - 0.5) * 200,
        y: existing ? existing.y : this.height / 2 + (Math.random() - 0.5) * 200,
        vx: existing ? existing.vx : 0,
        vy: existing ? existing.vy : 0
      };
    });

    this.nodeMap = new Map(this.activeNodes.map(d => [d.id, d]));

    // 2. Select raw edge set
    let rawEdges = [];
    if (this.activeCategory === 'ALL') {
      rawEdges = this.data.full_graph.edges;
    } else {
      rawEdges = this.data.category_graphs[this.activeCategory].edges;
    }

    // 3. Filter Edges by Metric & Threshold
    let filteredEdges = rawEdges.filter(e => {
      const weight = e[this.activeMetric];
      return weight >= this.threshold && this.nodeMap.has(e.source) && this.nodeMap.has(e.target);
    });

    // 4. If MST guarantee is active and an active node has 0 edges, add its highest weight edge
    if (this.keepMst) {
      const connected = new Set();
      filteredEdges.forEach(e => {
        connected.add(e.source);
        connected.add(e.target);
      });

      this.activeNodes.forEach(n => {
        if (!connected.has(n.id)) {
          // Find best edge for this node
          let bestEdge = null;
          let maxW = -Infinity;
          rawEdges.forEach(e => {
            if ((e.source === n.id || e.target === n.id) && this.nodeMap.has(e.source) && this.nodeMap.has(e.target)) {
              const w = e[this.activeMetric];
              if (w > maxW) {
                maxW = w;
                bestEdge = e;
              }
            }
          });
          if (bestEdge && !filteredEdges.includes(bestEdge)) {
            filteredEdges.push(bestEdge);
            connected.add(bestEdge.source);
            connected.add(bestEdge.target);
          }
        }
      });
    }

    // Build active link objects
    this.activeLinks = filteredEdges.map(e => ({
      source: this.nodeMap.get(e.source),
      target: this.nodeMap.get(e.target),
      weight: e[this.activeMetric],
      raw: e
    }));

    // 5. Calculate Graph Properties & Centralities
    this.calculateTopologicalMetrics();

    // 6. Restart Simulation
    this.restartSimulation();

    // 7. Render Elements
    this.render();

    // 8. Notify Metrics HUD
    if (this.onMetricsUpdate) {
      this.onMetricsUpdate(this.metrics);
    }
  }

  calculateTopologicalMetrics() {
    const N = this.activeNodes.length;
    const E = this.activeLinks.length;
    const density = N > 1 ? (2 * E) / (N * (N - 1)) : 0;

    // Build adjacency list
    const adj = new Map();
    this.activeNodes.forEach(n => adj.set(n.id, []));
    this.activeLinks.forEach(link => {
      const sId = typeof link.source === 'object' ? link.source.id : link.source;
      const tId = typeof link.target === 'object' ? link.target.id : link.target;
      if (adj.has(sId) && adj.has(tId)) {
        adj.get(sId).push({ id: tId, weight: link.weight });
        adj.get(tId).push({ id: sId, weight: link.weight });
      }
    });

    // Degree per node
    let totalDeg = 0;
    this.activeNodes.forEach(n => {
      n.degree = adj.get(n.id).length;
      n.degreeCentrality = N > 1 ? n.degree / (N - 1) : 0;
      totalDeg += n.degree;
    });
    const avgDegree = N > 0 ? totalDeg / N : 0;

    // Connected Components via BFS
    const visited = new Set();
    const components = [];
    this.activeNodes.forEach(node => {
      if (!visited.has(node.id)) {
        const comp = [];
        const queue = [node.id];
        visited.add(node.id);
        while (queue.length > 0) {
          const curr = queue.shift();
          comp.push(curr);
          adj.get(curr).forEach(neighbor => {
            if (!visited.has(neighbor.id)) {
              visited.add(neighbor.id);
              queue.push(neighbor.id);
            }
          });
        }
        components.push(comp);
      }
    });

    components.sort((a, b) => b.length - a.length);
    const giantComponent = components[0] || [];
    const giantSize = giantComponent.length;

    // Average Path Length and Diameter on Giant Component
    let totalDistance = 0;
    let pathPairsCount = 0;
    let diameter = 0;

    if (giantSize > 1) {
      giantComponent.forEach(startId => {
        const dists = new Map([[startId, 0]]);
        const queue = [startId];
        while (queue.length > 0) {
          const u = queue.shift();
          const dU = dists.get(u);
          adj.get(u).forEach(neighbor => {
            if (!dists.has(neighbor.id)) {
              dists.set(neighbor.id, dU + 1);
              queue.push(neighbor.id);
              if (dU + 1 > diameter) diameter = dU + 1;
            }
          });
        }
        dists.forEach((d, vId) => {
          if (startId < vId) {
            totalDistance += d;
            pathPairsCount++;
          }
        });
      });
    }

    const avgPathLength = pathPairsCount > 0 ? (totalDistance / pathPairsCount) : 0;

    // Local Clustering Coefficient
    let totalClustering = 0;
    this.activeNodes.forEach(node => {
      const neighbors = adj.get(node.id).map(x => x.id);
      const k = neighbors.length;
      if (k < 2) {
        node.clustering = 0;
      } else {
        let triangles = 0;
        for (let i = 0; i < k; i++) {
          for (let j = i + 1; j < k; j++) {
            const u = neighbors[i];
            const v = neighbors[j];
            if (adj.get(u).some(x => x.id === v)) {
              triangles++;
            }
          }
        }
        node.clustering = (2 * triangles) / (k * (k - 1));
      }
      totalClustering += node.clustering;
    });

    const avgClustering = N > 0 ? totalClustering / N : 0;

    // Approximate Betweenness Centrality
    this.calculateBetweenness(adj);

    // Greedy Modularity Communities
    const { modularity, communityMap } = this.calculateCommunities(adj, E);
    this.communityMap = communityMap;
    this.activeNodes.forEach(n => {
      n.community = communityMap.get(n.id) || 0;
    });

    this.metrics = {
      nodesCount: N,
      edgesCount: E,
      density: density.toFixed(3),
      avgDegree: avgDegree.toFixed(1),
      avgClustering: avgClustering.toFixed(3),
      avgPathLength: avgPathLength > 0 ? avgPathLength.toFixed(2) : 'N/A',
      diameter: diameter > 0 ? diameter : 'N/A',
      modularity: modularity.toFixed(3),
      numCommunities: new Set(communityMap.values()).size,
      giantComponentSize: giantSize
    };
  }

  calculateBetweenness(adj) {
    // Brandes algorithm for unweighted betweenness centrality
    const CB = new Map();
    this.activeNodes.forEach(n => CB.set(n.id, 0));

    this.activeNodes.forEach(s => {
      const S = [];
      const P = new Map();
      const sigma = new Map();
      const d = new Map();

      this.activeNodes.forEach(w => {
        P.set(w.id, []);
        sigma.set(w.id, 0);
        d.set(w.id, -1);
      });

      sigma.set(s.id, 1);
      d.set(s.id, 0);
      const Q = [s.id];

      while (Q.length > 0) {
        const v = Q.shift();
        S.push(v);
        const dV = d.get(v);
        adj.get(v).forEach(neighbor => {
          const w = neighbor.id;
          if (d.get(w) < 0) {
            d.set(w, dV + 1);
            Q.push(w);
          }
          if (d.get(w) === dV + 1) {
            sigma.set(w, sigma.get(w) + sigma.get(v));
            P.get(w).push(v);
          }
        });
      }

      const delta = new Map();
      this.activeNodes.forEach(w => delta.set(w.id, 0));

      while (S.length > 0) {
        const w = S.pop();
        P.get(w).forEach(v => {
          const c = (sigma.get(v) / sigma.get(w)) * (1 + delta.get(w));
          delta.set(v, delta.get(v) + c);
        });
        if (w !== s.id) {
          CB.set(w, CB.get(w) + delta.get(w));
        }
      }
    });

    const N = this.activeNodes.length;
    const norm = N > 2 ? ((N - 1) * (N - 2)) / 2 : 1;
    this.activeNodes.forEach(n => {
      n.betweenness = Math.round((CB.get(n.id) / norm) * 1000) / 1000;
    });
  }

  calculateCommunities(adj, m) {
    if (m === 0) {
      const cMap = new Map();
      this.activeNodes.forEach((n, idx) => cMap.set(n.id, idx));
      return { modularity: 0, communityMap: cMap };
    }

    // Initial partition: each node in its own community
    let cMap = new Map();
    this.activeNodes.forEach((n, idx) => cMap.set(n.id, idx));

    // Fast heuristic modularity optimization
    let improved = true;
    let passes = 0;
    const twoM = 2 * m;

    while (improved && passes < 10) {
      improved = false;
      passes++;

      for (let i = 0; i < this.activeNodes.length; i++) {
        const node = this.activeNodes[i];
        const currentC = cMap.get(node.id);
        const neighbors = adj.get(node.id);

        const candidateCommunities = new Set();
        candidateCommunities.add(currentC);
        neighbors.forEach(nbr => candidateCommunities.add(cMap.get(nbr.id)));

        let bestC = currentC;
        let bestGain = 0;

        // Compare neighbor affinity
        const affinityMap = new Map();
        neighbors.forEach(nbr => {
          const c = cMap.get(nbr.id);
          affinityMap.set(c, (affinityMap.get(c) || 0) + 1);
        });

        for (const targetC of candidateCommunities) {
          if (targetC === currentC) continue;
          const k_in = affinityMap.get(targetC) || 0;
          const k_curr = affinityMap.get(currentC) || 0;
          if (k_in > k_curr) {
            bestC = targetC;
            improved = true;
          }
        }
        cMap.set(node.id, bestC);
      }
    }

    // Renumber communities to compact indices 0, 1, 2...
    const uniqueComms = Array.from(new Set(cMap.values()));
    const finalMap = new Map();
    this.activeNodes.forEach(n => {
      finalMap.set(n.id, uniqueComms.indexOf(cMap.get(n.id)));
    });

    // Compute final modularity Q
    let Q = 0;
    for (const link of this.activeLinks) {
      const sId = typeof link.source === 'object' ? link.source.id : link.source;
      const tId = typeof link.target === 'object' ? link.target.id : link.target;
      if (finalMap.get(sId) === finalMap.get(tId)) {
        const ki = adj.get(sId).length;
        const kj = adj.get(tId).length;
        Q += 1 - (ki * kj) / twoM;
      }
    }
    Q = twoM > 0 ? Q / twoM : 0;

    return { modularity: Math.max(0, Q), communityMap: finalMap };
  }

  restartSimulation() {
    if (this.simulation) this.simulation.stop();

    this.simulation = d3.forceSimulation(this.activeNodes)
      .force('link', d3.forceLink(this.activeLinks).id(d => d.id).distance(d => {
        // Higher similarity = shorter distance
        const w = Math.max(0.1, d.weight);
        return 120 - w * 70;
      }))
      .force('charge', d3.forceManyBody().strength(-240))
      .force('center', d3.forceCenter(this.width / 2, this.height / 2))
      .force('collide', d3.forceCollide().radius(d => this.getNodeRadius(d) + 12).iterations(2))
      .on('tick', () => this.ticked());

    if (this.isFrozen) {
      this.simulation.stop();
    }
  }

  getNodeRadius(node) {
    switch (this.nodeSizeMode) {
      case 'degree':
        return 14 + Math.min(22, (node.degree || 0) * 1.3);
      case 'betweenness':
        return 14 + Math.min(24, (node.betweenness || 0) * 120);
      case 'variance': {
        const v = node.stats_full?.variance || node.stats_cat?.variance || 0.8;
        return 12 + v * 12;
      }
      case 'mean': {
        const m = node.stats_full?.mean || node.stats_cat?.mean || 3;
        return 12 + (m - 1) * 4.5;
      }
      case 'uniform':
      default:
        return 18;
    }
  }

  getNodeColor(node) {
    if (this.colorMode === 'community') {
      const palette = ['#2563eb', '#d97706', '#059669', '#7c3aed', '#db2777', '#0891b2', '#ea580c'];
      return palette[(node.community || 0) % palette.length];
    }
    return node.color || '#3b82f6';
  }

  ticked() {
    this.linkSelection
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);

    this.nodeSelection
      .attr('transform', d => `translate(${d.x},${d.y})`);
  }

  render() {
    // 1. Links Render
    this.linkSelection = this.linkLayer
      .selectAll('line.network-link')
      .data(this.activeLinks, d => `${d.source.id || d.source}-${d.target.id || d.target}`);

    this.linkSelection.exit().remove();

    const linkEnter = this.linkSelection.enter()
      .append('line')
      .attr('class', 'network-link');

    this.linkSelection = linkEnter.merge(this.linkSelection);

    this.linkSelection
      .attr('stroke', d => {
        const w = d.weight;
        if (this.activeMetric === 'pearson') {
          return w >= 0.5 ? '#0284c7' : '#94a3b8';
        }
        return '#0284c7';
      })
      .attr('stroke-opacity', d => Math.max(0.25, Math.min(0.85, d.weight)))
      .attr('stroke-width', d => {
        if (this.activeMetric === 'pearson') {
          return Math.max(1.2, d.weight * 4.5);
        }
        return Math.max(1.2, (d.weight - 0.7) * 8);
      });

    // 2. Nodes Render
    this.nodeSelection = this.nodeLayer
      .selectAll('g.network-node')
      .data(this.activeNodes, d => d.id);

    this.nodeSelection.exit().remove();

    const nodeEnter = this.nodeSelection.enter()
      .append('g')
      .attr('class', 'network-node')
      .call(d3.drag()
        .on('start', (event, d) => this.dragStarted(event, d))
        .on('drag', (event, d) => this.dragged(event, d))
        .on('end', (event, d) => this.dragEnded(event, d)))
      .on('click', (event, d) => {
        event.stopPropagation();
        this.selectNode(d.id);
      })
      .on('mouseover', (event, d) => this.showTooltip(event, d))
      .on('mousemove', (event, d) => this.moveTooltip(event, d))
      .on('mouseout', () => this.hideTooltip());

    nodeEnter.append('circle')
      .attr('class', 'node-circle');

    nodeEnter.append('text')
      .attr('class', 'node-label');

    this.nodeSelection = nodeEnter.merge(this.nodeSelection);

    this.updateNodeStyles();
  }

  updateNodeStyles() {
    if (!this.nodeSelection) return;

    this.nodeSelection.select('circle.node-circle')
      .transition().duration(250)
      .attr('r', d => this.getNodeRadius(d))
      .attr('fill', d => this.getNodeColor(d))
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 2.5);

    this.nodeSelection.select('text.node-label')
      .text(d => d.id)
      .style('display', this.showLabels ? 'block' : 'none');

    this.highlightSelection();
  }

  selectNode(nodeId) {
    this.selectedNodeId = nodeId;
    this.highlightSelection();

    if (this.onNodeSelect) {
      const node = nodeId ? this.nodeMap.get(nodeId) : null;
      this.onNodeSelect(node);
    }
  }

  highlightSelection() {
    if (!this.nodeSelection) return;

    if (!this.selectedNodeId) {
      this.nodeSelection.classed('is-dimmed', false).classed('is-highlighted', false);
      this.linkSelection.classed('is-dimmed', false).classed('is-highlighted', false);
      return;
    }

    const selId = this.selectedNodeId;
    const neighborSet = new Set();
    neighborSet.add(selId);

    this.linkSelection.each(d => {
      const sId = d.source.id;
      const tId = d.target.id;
      if (sId === selId) neighborSet.add(tId);
      if (tId === selId) neighborSet.add(sId);
    });

    this.nodeSelection
      .classed('is-dimmed', d => !neighborSet.has(d.id))
      .classed('is-highlighted', d => d.id === selId);

    this.linkSelection
      .classed('is-dimmed', d => d.source.id !== selId && d.target.id !== selId)
      .classed('is-highlighted', d => d.source.id === selId || d.target.id === selId);
  }

  highlightSearch() {
    if (!this.nodeSelection) return;
    if (!this.searchQuery) {
      this.nodeSelection.classed('is-dimmed', false);
      return;
    }

    const query = this.searchQuery;
    this.nodeSelection.classed('is-dimmed', d => {
      const matchId = d.id.toLowerCase().includes(query);
      const matchText = d.text.toLowerCase().includes(query);
      const matchCat = d.cat_name.toLowerCase().includes(query);
      return !(matchId || matchText || matchCat);
    });
  }

  showTooltip(event, d) {
    const stats = d.stats_full || d.stats_cat || {};
    this.tooltip
      .style('display', 'block')
      .style('opacity', 1)
      .html(`
        <div class="tooltip-header">
          <span class="tooltip-qid" style="color: ${d.color};">${d.id}</span>
          <span class="tooltip-cat" style="background: ${d.color}20; color: ${d.color};">${d.cat_name}</span>
        </div>
        <div class="tooltip-text">${d.text}</div>
        <div style="margin-top: 6px; font-size: 11px; color: #64748b;">
          Mean: <b>${stats.mean || 'N/A'}</b> &bull; Degree: <b>${d.degree || 0}</b> &bull; Betweenness: <b>${d.betweenness || 0}</b>
        </div>
      `);
    this.moveTooltip(event, d);
  }

  moveTooltip(event) {
    const container = this.svg.node().parentElement;
    const bounds = container.getBoundingClientRect();
    const x = event.clientX - bounds.left + 15;
    const y = event.clientY - bounds.top + 15;
    this.tooltip
      .style('left', `${Math.min(x, bounds.width - 290)}px`)
      .style('top', `${Math.min(y, bounds.height - 120)}px`);
  }

  hideTooltip() {
    this.tooltip.style('display', 'none');
  }

  dragStarted(event, d) {
    if (!event.active && !this.isFrozen) this.simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }

  dragEnded(event, d) {
    if (!event.active && !this.isFrozen) this.simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }
}
