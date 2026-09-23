import * as d3 from 'd3';
import { OpinionNetworkGraph } from './network.js';

let appData = null;
let graph = null;
let currentView = 'graph'; // 'graph' or 'heatmap'

// Threshold ranges per metric
const METRIC_CONFIGS = {
  pearson: {
    min: 0.10,
    max: 0.70,
    step: 0.01,
    default: 0.35,
    format: (v) => `≥ ${Number(v).toFixed(2)}`,
    ticks: [0.20, 0.30, 0.40, 0.50, 0.60],
    hint: 'Pearson correlation (r > 0) indicates shared opinion variance and alignment.'
  },
  agreement: {
    min: 0.60,
    max: 0.95,
    step: 0.01,
    default: 0.80,
    format: (v) => `≥ ${(Number(v) * 100).toFixed(0)}%`,
    ticks: [0.65, 0.75, 0.80, 0.85, 0.90],
    hint: 'Mean Response Agreement: fraction of alignment between 0% (opposite) and 100% (identical).'
  },
  cosine: {
    min: 0.88,
    max: 0.99,
    step: 0.005,
    default: 0.95,
    format: (v) => `≥ ${Number(v).toFixed(3)}`,
    ticks: [0.90, 0.92, 0.95, 0.97, 0.99],
    hint: 'Cosine similarity of Likert response vectors.'
  }
};

async function initApp() {
  try {
    const res = await fetch('./survey_network_data.json');
    appData = await res.json();
    console.log('Survey network data loaded:', appData);

    // Initialize Network Graph
    graph = new OpinionNetworkGraph(
      '#network-svg',
      '#graph-tooltip',
      handleNodeSelect,
      handleMetricsUpdate
    );

    graph.setData(appData);

    setupEventListeners();
    updateCohortBadge('ALL');
  } catch (err) {
    console.error('Failed to initialize opinion network visualizer:', err);
  }
}

function setupEventListeners() {
  // View Switcher
  const btnViewGraph = document.getElementById('btn-view-graph');
  const btnViewHeatmap = document.getElementById('btn-view-heatmap');
  const graphViewPanel = document.getElementById('graph-view');
  const heatmapViewPanel = document.getElementById('heatmap-view');

  btnViewGraph.addEventListener('click', () => {
    currentView = 'graph';
    btnViewGraph.classList.add('active');
    btnViewHeatmap.classList.remove('active');
    graphViewPanel.classList.add('active');
    heatmapViewPanel.classList.remove('active');
  });

  btnViewHeatmap.addEventListener('click', () => {
    currentView = 'heatmap';
    btnViewHeatmap.classList.add('active');
    btnViewGraph.classList.remove('active');
    heatmapViewPanel.classList.add('active');
    graphViewPanel.classList.remove('active');
    renderHeatmap();
  });

  // Category Selector
  const catSelect = document.getElementById('category-select');
  catSelect.addEventListener('change', (e) => {
    const cat = e.target.value;
    graph.setCategory(cat);
    updateCohortBadge(cat);
    if (currentView === 'heatmap') {
      renderHeatmap();
    }
  });

  // Metric Selector
  const metricSelect = document.getElementById('metric-select');
  const metricHint = document.getElementById('metric-hint');
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdValBadge = document.getElementById('threshold-val-badge');
  const sliderTicksContainer = document.querySelector('.slider-ticks');

  metricSelect.addEventListener('change', (e) => {
    const metric = e.target.value;
    const cfg = METRIC_CONFIGS[metric];

    // Update hint
    metricHint.textContent = cfg.hint;

    // Update slider properties
    thresholdSlider.min = cfg.min;
    thresholdSlider.max = cfg.max;
    thresholdSlider.step = cfg.step;
    thresholdSlider.value = cfg.default;
    thresholdValBadge.textContent = cfg.format(cfg.default);

    // Update ticks
    sliderTicksContainer.innerHTML = cfg.ticks
      .map(t => `<span class="tick" data-val="${t}">${metric === 'agreement' ? (t*100).toFixed(0)+'%' : t}</span>`)
      .join('');

    bindSliderTicks();

    graph.setMetric(metric);
    graph.setThreshold(cfg.default);

    if (currentView === 'heatmap') {
      renderHeatmap();
    }
  });

  // Threshold Slider
  thresholdSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    const metric = metricSelect.value;
    thresholdValBadge.textContent = METRIC_CONFIGS[metric].format(val);
    graph.setThreshold(val);
  });

  bindSliderTicks();

  // MST Guarantee Toggle
  document.getElementById('mst-guarantee-toggle').addEventListener('change', (e) => {
    graph.setMstGuarantee(e.target.checked);
  });

  // Node Size Mode
  document.getElementById('node-size-select').addEventListener('change', (e) => {
    graph.setNodeSizeMode(e.target.value);
  });

  // Color Mode
  document.getElementById('color-mode-select').addEventListener('change', (e) => {
    graph.setColorMode(e.target.value);
  });

  // Physics Freeze
  document.getElementById('physics-freeze-toggle').addEventListener('change', (e) => {
    graph.setFrozen(e.target.checked);
  });

  // Show Labels
  document.getElementById('show-labels-toggle').addEventListener('change', (e) => {
    graph.setShowLabels(e.target.checked);
  });

  // Search Input
  const searchInput = document.getElementById('question-search');
  const clearBtn = document.getElementById('search-clear-btn');
  searchInput.addEventListener('input', (e) => {
    const q = e.target.value;
    clearBtn.style.display = q ? 'block' : 'none';
    graph.setSearchQuery(q);
  });

  clearBtn.addEventListener('click', () => {
    searchInput.value = '';
    clearBtn.style.display = 'none';
    graph.setSearchQuery('');
  });

  // Reset Zoom
  document.getElementById('btn-reset-zoom').addEventListener('click', () => {
    graph.resetZoom();
  });

  // Export PNG
  document.getElementById('btn-export-png').addEventListener('click', exportPng);

  // Close Inspector Button
  document.getElementById('btn-close-inspector').addEventListener('click', () => {
    graph.selectNode(null);
  });
}

function bindSliderTicks() {
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdValBadge = document.getElementById('threshold-val-badge');
  const metricSelect = document.getElementById('metric-select');

  document.querySelectorAll('.slider-ticks .tick').forEach(tick => {
    tick.addEventListener('click', () => {
      const val = parseFloat(tick.dataset.val);
      thresholdSlider.value = val;
      const metric = metricSelect.value;
      thresholdValBadge.textContent = METRIC_CONFIGS[metric].format(val);
      graph.setThreshold(val);
    });
  });
}

function updateCohortBadge(cat) {
  const textEl = document.getElementById('respondent-count-text');
  const hintEl = document.getElementById('category-scope-hint');

  if (cat === 'ALL') {
    const count = appData.metadata.full_graph_respondents;
    textEl.textContent = `${count} Complete Respondents`;
    hintEl.textContent = `Includes ${count} respondents with 100% completed responses across all 60 questions (5 empty rows excluded).`;
  } else {
    const count = appData.metadata.category_respondents[cat];
    const catName = appData.metadata.categories[cat];
    textEl.textContent = `${count} Respondents (${catName})`;
    hintEl.textContent = `Considers all ${count} respondents who fully completed Category ${cat} (${catName}).`;
  }
}

function handleMetricsUpdate(metrics) {
  document.getElementById('metric-nodes').textContent = metrics.nodesCount;
  document.getElementById('metric-edges').textContent = metrics.edgesCount;
  document.getElementById('metric-density').textContent = metrics.density;
  document.getElementById('metric-avg-degree').textContent = metrics.avgDegree;
  document.getElementById('metric-clustering').textContent = metrics.avgClustering;
  document.getElementById('metric-modularity').textContent = metrics.modularity;
  document.getElementById('metric-path-length').textContent = metrics.avgPathLength;
  document.getElementById('metric-diameter').textContent = metrics.diameter;
  document.getElementById('active-edges-count').textContent = `${metrics.edgesCount} active links`;
}

function handleNodeSelect(node) {
  const emptyState = document.getElementById('inspector-empty-state');
  const details = document.getElementById('inspector-details');

  if (!node) {
    emptyState.style.display = 'flex';
    details.style.display = 'none';
    return;
  }

  emptyState.style.display = 'none';
  details.style.display = 'flex';

  const cat = node.cat;
  const isAll = graph.activeCategory === 'ALL';
  const stats = isAll ? node.stats_full : (node.stats_cat || node.stats_full);

  // Header & Tags
  const badge = document.getElementById('node-badge');
  badge.textContent = node.cat_name;
  badge.style.background = `${node.color}15`;
  badge.style.color = node.color;
  badge.style.border = `1px solid ${node.color}40`;

  document.getElementById('node-id').textContent = node.id;
  document.getElementById('node-text').textContent = node.text;

  // Overview pills
  document.getElementById('node-mean').textContent = stats.mean.toFixed(2);
  document.getElementById('node-variance').textContent = `σ = ${stats.std.toFixed(2)}`;
  document.getElementById('node-degree-val').textContent = node.degree || 0;

  // Likert Distribution Bars
  const dist = stats.distribution || {};
  const total = stats.n_respondents || 1;
  const distContainer = document.getElementById('dist-bars-container');

  const likertConfig = [
    { label: 'Strongly Agree', key: 'Strongly Agree', color: 'var(--likert-sa)' },
    { label: 'Agree', key: 'Agree', color: 'var(--likert-a)' },
    { label: 'Neutral', key: 'Neutral', color: 'var(--likert-n)' },
    { label: 'Disagree', key: 'Disagree', color: 'var(--likert-d)' },
    { label: 'Strongly Disagree', key: 'Strongly Disagree', color: 'var(--likert-sd)' }
  ];

  distContainer.innerHTML = likertConfig.map(item => {
    const count = dist[item.key] || 0;
    const pct = ((count / total) * 100).toFixed(1);
    return `
      <div class="dist-row">
        <div class="dist-meta">
          <span>${item.label}</span>
          <span><b>${count}</b> (${pct}%)</span>
        </div>
        <div class="dist-track">
          <div class="dist-fill" style="width: ${pct}%; background: ${item.color};"></div>
        </div>
      </div>
    `;
  }).join('');

  // Centrality Details
  document.getElementById('val-deg-centrality').textContent = (node.degreeCentrality || 0).toFixed(3);
  document.getElementById('val-betweenness').textContent = (node.betweenness || 0).toFixed(4);
  document.getElementById('val-closeness').textContent = (node.clustering || 0).toFixed(3) + ' (clust)';
  document.getElementById('val-community').textContent = `Cluster ${node.community + 1}`;

  // Find Top Similar and Top Divergent Questions
  populateRelations(node);
}

function populateRelations(node) {
  const metric = graph.activeMetric;
  const isAll = graph.activeCategory === 'ALL';
  const allEdges = isAll ? appData.full_graph.edges : appData.category_graphs[node.cat].edges;

  // Filter edges involving this node
  const nodeEdges = [];
  allEdges.forEach(e => {
    let otherId = null;
    if (e.source === node.id) otherId = e.target;
    else if (e.target === node.id) otherId = e.source;

    if (otherId) {
      const otherNode = appData.questions.find(q => q.id === otherId);
      if (otherNode) {
        nodeEdges.push({
          id: otherId,
          node: otherNode,
          weight: e[metric],
          pearson: e.pearson,
          agreement: e.agreement
        });
      }
    }
  });

  // Sort descending for similar
  nodeEdges.sort((a, b) => b.weight - a.weight);
  const topSimilar = nodeEdges.slice(0, 3);

  // Sort ascending for divergent
  const topDivergent = [...nodeEdges].sort((a, b) => a.weight - b.weight).slice(0, 3);

  const similarContainer = document.getElementById('top-similar-list');
  const divergentContainer = document.getElementById('top-divergent-list');

  similarContainer.innerHTML = topSimilar.map(item => createRelationItemHtml(item, 'positive')).join('');
  divergentContainer.innerHTML = topDivergent.map(item => createRelationItemHtml(item, 'negative')).join('');

  // Add click handler to jump to node
  document.querySelectorAll('.relation-item').forEach(el => {
    el.addEventListener('click', () => {
      const targetId = el.dataset.qid;
      graph.selectNode(targetId);
    });
  });
}

function createRelationItemHtml(item, sentiment) {
  const isPearson = graph.activeMetric === 'pearson';
  const displayVal = isPearson 
    ? (item.weight >= 0 ? `+${item.weight.toFixed(2)}` : item.weight.toFixed(2))
    : `${(item.weight * 100).toFixed(0)}%`;

  return `
    <div class="relation-item" data-qid="${item.id}" title="${item.node.text}">
      <div class="rel-left">
        <span class="rel-badge" style="background: ${item.node.color}20; color: ${item.node.color};">
          ${item.id}
        </span>
        <span class="rel-text">${item.node.text}</span>
      </div>
      <span class="rel-weight ${sentiment === 'positive' ? 'weight-positive' : 'weight-negative'}">
        ${displayVal}
      </span>
    </div>
  `;
}

function renderHeatmap() {
  const container = document.getElementById('heatmap-container');
  container.innerHTML = '';

  const cat = graph.activeCategory;
  const metric = graph.activeMetric;

  let questions = appData.questions;
  let matrix = [];

  if (cat === 'ALL') {
    questions = appData.questions;
    matrix = appData.full_graph[`${metric}_matrix`];
  } else {
    questions = appData.questions.filter(q => q.cat === cat);
    matrix = appData.category_graphs[cat][`${metric}_matrix`];
  }

  const n = questions.length;
  const cellSize = n > 20 ? 12 : 28;
  const margin = { top: 60, right: 30, bottom: 30, left: 60 };
  const width = cellSize * n + margin.left + margin.right;
  const height = cellSize * n + margin.top + margin.bottom;

  const svg = d3.select(container)
    .append('svg')
    .attr('width', width)
    .attr('height', height);

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  // Color scale
  let colorScale;
  if (metric === 'pearson') {
    colorScale = d3.scaleDiverging()
      .domain([-0.4, 0, 0.7])
      .interpolator(d3.interpolateRdBu);
  } else {
    colorScale = d3.scaleSequential()
      .domain([0.5, 1.0])
      .interpolator(d3.interpolateBlues);
  }

  // Draw cells
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const val = matrix[i][j];
      const q1 = questions[i];
      const q2 = questions[j];

      const rect = g.append('rect')
        .attr('x', j * cellSize)
        .attr('y', i * cellSize)
        .attr('width', cellSize - 1)
        .attr('height', cellSize - 1)
        .attr('fill', colorScale(val))
        .attr('rx', 2)
        .style('cursor', 'pointer');

      rect.append('title').text(`${q1.id} & ${q2.id}\n${q1.text}\nvs\n${q2.text}\n${metric}: ${val}`);

      rect.on('click', () => {
        // Switch back to graph and select q1
        document.getElementById('btn-view-graph').click();
        graph.selectNode(q1.id);
      });
    }
  }

  // Row & Column Labels
  for (let i = 0; i < n; i++) {
    // Top axis (cols)
    g.append('text')
      .attr('x', i * cellSize + cellSize / 2)
      .attr('y', -8)
      .attr('text-anchor', 'start')
      .attr('transform', `rotate(-60, ${i * cellSize + cellSize / 2}, -8)`)
      .style('font-size', n > 20 ? '8px' : '10px')
      .style('font-weight', '700')
      .style('fill', questions[i].color)
      .text(questions[i].id);

    // Left axis (rows)
    g.append('text')
      .attr('x', -8)
      .attr('y', i * cellSize + cellSize / 2 + 3)
      .attr('text-anchor', 'end')
      .style('font-size', n > 20 ? '8px' : '10px')
      .style('font-weight', '700')
      .style('fill', questions[i].color)
      .text(questions[i].id);
  }
}

function exportPng() {
  const svgEl = document.getElementById('network-svg');
  const serializer = new XMLSerializer();
  let svgString = serializer.serializeToString(svgEl);

  // Add background
  svgString = svgString.replace('<svg ', '<svg style="background: #ffffff;" ');

  const img = new Image();
  const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(svgBlob);

  img.onload = () => {
    const canvas = document.createElement('canvas');
    canvas.width = svgEl.clientWidth * 2;
    canvas.height = svgEl.clientHeight * 2;
    const ctx = canvas.getContext('2d');
    ctx.scale(2, 2);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    const pngUrl = canvas.toDataURL('image/png');
    const downloadLink = document.createElement('a');
    downloadLink.href = pngUrl;
    downloadLink.download = `opinion_network_${graph.activeCategory}_${graph.activeMetric}.png`;
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
    URL.revokeObjectURL(url);
  };

  img.src = url;
}

// Boot
window.addEventListener('DOMContentLoaded', initApp);
