from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_etl" / "indexes" / "topic_graph_with_similarity.json"
DEFAULT_OUTPUT = ROOT / "results" / "etl_graph_viewer_sigma.html"


HTML_TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>eTL Guide Graph Explorer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #171b20;
      --panel-2: #20262d;
      --text: #e8edf2;
      --muted: #98a3ad;
      --line: #303841;
      --accent: #5bc0eb;
      --document: #7bd88f;
      --topic: #f7b267;
      --concept: #66c2ff;
      --fact: #ff8fab;
      --evidence: #d0a2f7;
      --outcome: #ffd166;
    }

    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; background: var(--bg); color: var(--text); font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { display: grid; grid-template-columns: 360px 1fr 380px; overflow: hidden; }

    aside, section { min-width: 0; min-height: 0; }
    .left, .right { background: var(--panel); border-color: var(--line); overflow: auto; }
    .left { border-right: 1px solid var(--line); padding: 18px; }
    .right { border-left: 1px solid var(--line); padding: 18px; }
    .main { position: relative; min-width: 0; }
    #container { position: absolute; inset: 0; background: radial-gradient(circle at 30% 20%, #1c2630 0, #101214 42%); }

    h1 { font-size: 18px; margin: 0 0 4px; letter-spacing: 0; }
    h2 { font-size: 13px; margin: 22px 0 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0; }
    p { margin: 7px 0; color: var(--muted); font-size: 13px; line-height: 1.5; }
    label { display: block; margin: 12px 0 6px; color: var(--muted); font-size: 12px; }
    input[type="search"], input[type="number"], select {
      width: 100%;
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      font-size: 14px;
      outline: none;
    }
    input:focus, select:focus { border-color: var(--accent); }

    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .checks { display: grid; gap: 8px; margin-top: 8px; }
    .checks label { display: flex; align-items: center; gap: 8px; margin: 0; color: var(--text); font-size: 13px; }
    .checks input { accent-color: var(--accent); }

    button {
      width: 100%;
      margin-top: 12px;
      background: #2a8db8;
      color: white;
      border: 0;
      border-radius: 6px;
      padding: 10px 12px;
      font-weight: 650;
      cursor: pointer;
    }
    button.secondary { background: var(--panel-2); color: var(--text); border: 1px solid var(--line); }
    button:hover { filter: brightness(1.08); }

    .stats {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 14px;
    }
    .stat { background: var(--panel-2); border: 1px solid var(--line); border-radius: 7px; padding: 10px; }
    .stat strong { display: block; font-size: 18px; }
    .stat span { display: block; color: var(--muted); font-size: 11px; margin-top: 2px; }

    .legend { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
    .chip { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 12px; }
    .dot { width: 10px; height: 10px; border-radius: 99px; flex: 0 0 auto; }

    #results { margin-top: 12px; display: grid; gap: 7px; }
    .result {
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 7px;
      padding: 9px 10px;
      cursor: pointer;
    }
    .result:hover { border-color: var(--accent); }
    .result .label { font-size: 13px; line-height: 1.35; word-break: keep-all; overflow-wrap: anywhere; }
    .result .meta { color: var(--muted); font-size: 11px; margin-top: 4px; }

    .topbar {
      position: absolute;
      top: 14px;
      left: 14px;
      right: 14px;
      z-index: 3;
      display: flex;
      gap: 10px;
      align-items: center;
      pointer-events: none;
    }
    .pill {
      background: rgba(16, 18, 20, .78);
      border: 1px solid rgba(255,255,255,.12);
      color: var(--text);
      backdrop-filter: blur(10px);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 12px;
      pointer-events: auto;
    }

    .detail-title { font-size: 16px; line-height: 1.45; margin: 0 0 8px; overflow-wrap: anywhere; }
    .kv { border-top: 1px solid var(--line); padding: 10px 0; }
    .kv .k { color: var(--muted); font-size: 11px; margin-bottom: 4px; text-transform: uppercase; }
    .kv .v { font-size: 13px; line-height: 1.5; overflow-wrap: anywhere; white-space: pre-wrap; }
    .neighbors { display: grid; gap: 6px; }
    .neighbor { font-size: 12px; color: var(--muted); background: var(--panel-2); border: 1px solid var(--line); border-radius: 6px; padding: 7px 8px; cursor: pointer; }
    .neighbor:hover { color: var(--text); border-color: var(--accent); }

    @media (max-width: 1100px) {
      body { grid-template-columns: 320px 1fr; }
      .right { display: none; }
    }
  </style>
</head>
<body>
  <aside class="left">
    <h1>eTL Guide Graph</h1>
    <p>Search an eTL guide topic, then render its local neighborhood.</p>

    <div class="stats">
      <div class="stat"><strong id="nodeCount">-</strong><span>nodes</span></div>
      <div class="stat"><strong id="edgeCount">-</strong><span>edges</span></div>
    </div>

    <h2>Search</h2>
    <input id="search" type="search" value="과제 추가하기" placeholder="과제, 퀴즈, 동영상 출결, 메시지..." />
    <label for="typeFilter">Node type</label>
    <select id="typeFilter">
      <option value="">All types</option>
      <option>Document</option>
      <option>Topic</option>
      <option>Screen</option>
      <option>Task</option>
      <option>Setting</option>
      <option>Outcome</option>
    </select>

    <div class="row">
      <div>
        <label for="depth">Depth</label>
        <input id="depth" type="number" min="1" max="4" value="2" />
      </div>
      <div>
        <label for="maxNodes">Max nodes</label>
        <input id="maxNodes" type="number" min="20" max="900" value="260" />
      </div>
    </div>

    <h2>Relations</h2>
    <div class="checks" id="relationChecks"></div>

    <button id="renderBtn">Render Neighborhood</button>
    <button class="secondary" id="resetBtn">Reset Camera</button>

    <h2>Legend</h2>
    <div class="legend" id="legend"></div>

    <h2>Matches</h2>
    <div id="results"></div>
  </aside>

  <main class="main">
    <div id="container"></div>
    <div class="topbar">
      <div class="pill" id="subgraphStats">No subgraph rendered</div>
      <div class="pill">Scroll to zoom · drag to pan · click node for detail</div>
    </div>
  </main>

  <section class="right">
    <h2>Selected Node</h2>
    <div id="details"><p>Click a node or a search result.</p></div>
    <h2>Visible Neighbors</h2>
    <div class="neighbors" id="neighbors"></div>
  </section>

  <script id="graph-data" type="application/json">__GRAPH_JSON__</script>
  <script type="module">
    import Graph from "https://esm.sh/graphology@0.26.0";
    import Sigma from "https://esm.sh/sigma@3";
    import forceAtlas2 from "https://esm.sh/graphology-layout-forceatlas2@0.10.1";

    const raw = JSON.parse(document.getElementById("graph-data").textContent);
    const nodes = raw.nodes;
    const edges = raw.edges;
    const nodeById = new Map(nodes.map(node => [node.id, node]));

    const colors = {
      Document: "#7bd88f",
      Topic: "#f7b267",
      Screen: "#66c2ff",
      Task: "#ff8fab",
      Setting: "#d0a2f7",
      Outcome: "#ffd166",
    };
    const sizes = {
      Document: 6,
      Topic: 5,
      Screen: 4,
      Task: 4,
      Setting: 4,
      Outcome: 5,
    };
    const relationColors = {
      SIMILAR_TO: "rgba(140, 170, 190, 0.35)",
      HAS_ISSUE: "rgba(247, 178, 103, 0.45)",
      HAS_OUTCOME: "rgba(255, 209, 102, 0.5)",
      INVOLVES_CONCEPT: "rgba(102, 194, 255, 0.35)",
      HAS_FACT_PATTERN: "rgba(255, 143, 171, 0.35)",
      HAS_EVIDENCE_TYPE: "rgba(208, 162, 247, 0.35)",
    };

    const relationTypes = [...new Set(edges.map(edge => edge.relation))].sort();
    const adjacency = new Map();
    for (const edge of edges) {
      if (!adjacency.has(edge.source)) adjacency.set(edge.source, []);
      if (!adjacency.has(edge.target)) adjacency.set(edge.target, []);
      adjacency.get(edge.source).push({ ...edge, other: edge.target, direction: "out" });
      adjacency.get(edge.target).push({ ...edge, other: edge.source, direction: "in" });
    }

    document.getElementById("nodeCount").textContent = nodes.length.toLocaleString();
    document.getElementById("edgeCount").textContent = edges.length.toLocaleString();

    const relationChecks = document.getElementById("relationChecks");
    for (const relation of relationTypes) {
      const label = document.createElement("label");
      label.innerHTML = `<input type="checkbox" value="${relation}" checked /> ${relation}`;
      relationChecks.appendChild(label);
    }

    const legend = document.getElementById("legend");
    for (const [type, color] of Object.entries(colors)) {
      const chip = document.createElement("div");
      chip.className = "chip";
      chip.innerHTML = `<span class="dot" style="background:${color}"></span>${type}`;
      legend.appendChild(chip);
    }

    let renderer = null;
    let currentGraph = null;
    let selectedNode = null;
    let lastMatches = [];

    function allowedRelations() {
      return new Set([...relationChecks.querySelectorAll("input:checked")].map(input => input.value));
    }

    function searchNodes() {
      const q = document.getElementById("search").value.trim().toLowerCase();
      const type = document.getElementById("typeFilter").value;
      const results = nodes
        .filter(node => !type || node.type === type)
        .filter(node => !q || node.label.toLowerCase().includes(q) || node.id.toLowerCase().includes(q))
        .slice(0, 30);
      lastMatches = results;
      renderResults(results);
      return results;
    }

    function renderResults(results) {
      const el = document.getElementById("results");
      el.innerHTML = "";
      for (const node of results) {
        const item = document.createElement("div");
        item.className = "result";
        item.innerHTML = `<div class="label">${escapeHtml(node.label)}</div><div class="meta">${node.type} · ${escapeHtml(node.id)}</div>`;
        item.addEventListener("click", () => renderNeighborhood(node.id));
        el.appendChild(item);
      }
      if (!results.length) el.innerHTML = "<p>No matches.</p>";
    }

    function collectNeighborhood(seedId) {
      const depth = Number(document.getElementById("depth").value || 2);
      const maxNodes = Number(document.getElementById("maxNodes").value || 260);
      const relations = allowedRelations();
      const visited = new Set([seedId]);
      const queue = [{ id: seedId, depth: 0 }];
      const edgeSet = new Set();

      while (queue.length && visited.size < maxNodes) {
        const current = queue.shift();
        if (current.depth >= depth) continue;
        const incident = adjacency.get(current.id) || [];
        incident.sort((a, b) => relationRank(a.relation) - relationRank(b.relation));
        for (const edge of incident) {
          if (!relations.has(edge.relation)) continue;
          edgeSet.add(edgeKey(edge));
          if (!visited.has(edge.other)) {
            visited.add(edge.other);
            queue.push({ id: edge.other, depth: current.depth + 1 });
            if (visited.size >= maxNodes) break;
          }
        }
      }

      const visibleEdges = edges.filter(edge =>
        visited.has(edge.source) &&
        visited.has(edge.target) &&
        relations.has(edge.relation) &&
        edgeSet.has(edgeKey(edge))
      );
      return { nodeIds: [...visited], visibleEdges };
    }

    function relationRank(relation) {
      return relation === "SIMILAR_TO" ? 5 : 1;
    }

    function edgeKey(edge) {
      return `${edge.source}--${edge.relation}--${edge.target}--${edge.weight || ""}`;
    }

    function renderNeighborhood(seedId) {
      const seed = nodeById.get(seedId);
      if (!seed) return;
      selectedNode = seedId;

      const { nodeIds, visibleEdges } = collectNeighborhood(seedId);
      const graph = new Graph({ multi: true, type: "directed" });
      const byTypeIndex = new Map();

      for (const id of nodeIds) {
        const node = nodeById.get(id);
        const typeIndex = byTypeIndex.get(node.type) || 0;
        byTypeIndex.set(node.type, typeIndex + 1);
        const angle = typeIndex * 2.399963 + typeRadius(node.type);
        const radius = 10 + typeIndex * 0.08 + typeRadius(node.type);
        graph.addNode(id, {
          label: node.label,
          x: Math.cos(angle) * radius,
          y: Math.sin(angle) * radius,
          size: sizes[node.type] || 4,
          color: colors[node.type] || "#ddd",
          nodeType: node.type,
          original: node,
          forceLabel: id === seedId || node.type === "Outcome",
          zIndex: id === seedId ? 10 : 1,
        });
      }

      let edgeIndex = 0;
      for (const edge of visibleEdges) {
        if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) continue;
        graph.addDirectedEdgeWithKey(`e${edgeIndex++}`, edge.source, edge.target, {
          label: edge.relation,
          relation: edge.relation,
          size: edge.relation === "SIMILAR_TO" ? 1 : 1.6,
          color: relationColors[edge.relation] || "rgba(255,255,255,.22)",
          weight: edge.weight,
        });
      }

      if (graph.order > 2) {
        forceAtlas2.assign(graph, {
          iterations: graph.order < 120 ? 120 : 70,
          settings: {
            gravity: 1,
            scalingRatio: 12,
            slowDown: 2,
            strongGravityMode: false,
            barnesHutOptimize: graph.order > 120,
            barnesHutTheta: 0.6,
          },
        });
      }

      if (renderer) renderer.kill();
      currentGraph = graph;
      renderer = new Sigma(graph, document.getElementById("container"), {
        renderEdgeLabels: false,
        zIndex: true,
        labelDensity: 0.08,
        labelGridCellSize: 90,
        labelColor: { color: "#e8edf2" },
        defaultEdgeType: "line",
        nodeReducer: (node, data) => {
          if (!selectedNode) return data;
          if (node === selectedNode) return { ...data, size: data.size * 1.8, color: "#ffffff", forceLabel: true };
          return data;
        },
      });

      renderer.on("clickNode", event => {
        selectedNode = event.node;
        renderer.refresh();
        renderDetails(event.node);
      });

      renderDetails(seedId);
      document.getElementById("subgraphStats").textContent =
        `${graph.order.toLocaleString()} nodes · ${graph.size.toLocaleString()} edges · seed: ${seed.label}`;
    }

    function typeRadius(type) {
      return {
        Document: 3,
        Topic: 7,
        Screen: 11,
        Task: 15,
        Setting: 19,
        Outcome: 23,
      }[type] || 10;
    }

    function renderDetails(id) {
      const node = nodeById.get(id);
      if (!node) return;

      const details = document.getElementById("details");
      const fields = Object.entries(node)
        .filter(([key]) => !["id", "label"].includes(key))
        .map(([key, value]) => `<div class="kv"><div class="k">${escapeHtml(key)}</div><div class="v">${escapeHtml(String(value))}</div></div>`)
        .join("");
      details.innerHTML = `<h3 class="detail-title">${escapeHtml(node.label)}</h3><div class="kv"><div class="k">id</div><div class="v">${escapeHtml(node.id)}</div></div>${fields}`;

      const neighborEl = document.getElementById("neighbors");
      neighborEl.innerHTML = "";
      const seen = new Set();
      for (const edge of adjacency.get(id) || []) {
        if (!currentGraph || !currentGraph.hasNode(edge.other) || seen.has(edge.other)) continue;
        seen.add(edge.other);
        const other = nodeById.get(edge.other);
        const div = document.createElement("div");
        div.className = "neighbor";
        div.textContent = `${edge.relation} · ${other.type} · ${other.label}`;
        div.addEventListener("click", () => {
          selectedNode = other.id;
          renderer.refresh();
          renderDetails(other.id);
        });
        neighborEl.appendChild(div);
      }
      if (!seen.size) neighborEl.innerHTML = "<p>No visible neighbors.</p>";
    }

    function escapeHtml(value) {
      return value.replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[ch]));
    }

    document.getElementById("search").addEventListener("input", searchNodes);
    document.getElementById("typeFilter").addEventListener("change", searchNodes);
    document.getElementById("renderBtn").addEventListener("click", () => {
      const seed = lastMatches[0];
      if (seed) renderNeighborhood(seed.id);
    });
    document.getElementById("resetBtn").addEventListener("click", () => renderer?.getCamera().animatedReset());
    relationChecks.addEventListener("change", () => {
      if (selectedNode) renderNeighborhood(selectedNode);
    });

    const initial = searchNodes()[0] || nodes[0];
    renderNeighborhood(initial.id);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an interactive sigma.js graph viewer.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = json.loads(args.input.read_text(encoding="utf-8"))
    html = HTML_TEMPLATE.replace(
        "__GRAPH_JSON__",
        json.dumps(graph, ensure_ascii=False).replace("</script>", "<\\/script>"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(args.output)
    print(f"nodes={len(graph['nodes'])} edges={len(graph['edges'])}")


if __name__ == "__main__":
    main()
