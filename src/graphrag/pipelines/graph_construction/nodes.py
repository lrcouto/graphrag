"""Graph construction pipeline nodes."""
import logging
from pathlib import Path

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def build_knowledge_graph(df: pd.DataFrame, node_colors: dict) -> nx.Graph:
    G = nx.Graph()

    def _tooltip(label: str, node_type: str, count: int) -> str:
        return f"<b>{label}</b><br>Type: {node_type}<br>Patients: {count:,}"

    for cond in df["Medical Condition"].unique():
        n = len(df[df["Medical Condition"] == cond])
        G.add_node(cond, node_type="condition", color=node_colors["condition"], count=n,
                   title=_tooltip(cond, "Medical Condition", n))

    for med in df["Medication"].unique():
        n = len(df[df["Medication"] == med])
        G.add_node(med, node_type="medication", color=node_colors["medication"], count=n,
                   title=_tooltip(med, "Medication", n))

    for ins in df["Insurance Provider"].unique():
        n = len(df[df["Insurance Provider"] == ins])
        G.add_node(ins, node_type="insurer", color=node_colors["insurer"], count=n,
                   title=_tooltip(ins, "Insurance Provider", n))

    for adm in df["Admission Type"].unique():
        n = len(df[df["Admission Type"] == adm])
        G.add_node(adm, node_type="admission_type", color=node_colors["admission_type"], count=n,
                   title=_tooltip(adm, "Admission Type", n))

    for res in df["Test Results"].unique():
        n = len(df[df["Test Results"] == res])
        G.add_node(res, node_type="test_result", color=node_colors["test_result"], count=n,
                   title=_tooltip(res, "Test Result", n))

    for bt in df["Blood Type"].unique():
        n = len(df[df["Blood Type"] == bt])
        G.add_node(bt, node_type="blood_type", color=node_colors["blood_type"], count=n,
                   title=_tooltip(bt, "Blood Type", n))

    for (cond, med), n in df.groupby(["Medical Condition", "Medication"]).size().items():
        G.add_edge(cond, med, weight=int(n), relationship="TREATED_WITH",
                   title=f"TREATED_WITH<br>{n:,} patients")

    for (cond, ins), n in df.groupby(["Medical Condition", "Insurance Provider"]).size().items():
        G.add_edge(cond, ins, weight=int(n), relationship="COVERED_BY",
                   title=f"COVERED_BY<br>{n:,} patients")

    for (cond, adm), n in df.groupby(["Medical Condition", "Admission Type"]).size().items():
        G.add_edge(cond, adm, weight=int(n), relationship="ADMITTED_AS",
                   title=f"ADMITTED_AS<br>{n:,} cases")

    for (cond, res), n in df.groupby(["Medical Condition", "Test Results"]).size().items():
        G.add_edge(cond, res, weight=int(n), relationship="SHOWS_RESULT",
                   title=f"SHOWS_RESULT<br>{n:,} cases")

    # Top-2 blood types per condition to add richness without clutter
    for cond in df["Medical Condition"].unique():
        top_bt = df[df["Medical Condition"] == cond]["Blood Type"].value_counts().head(4)
        for bt, n in top_bt.items():
            G.add_edge(cond, bt, weight=int(n), relationship="ASSOCIATED_WITH",
                       title=f"ASSOCIATED_WITH<br>{n:,} patients")

    logger.info("Knowledge graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def render_graph_html(knowledge_graph: nx.Graph, entity_summaries: dict, graph_html_path: str) -> pd.DataFrame:
    import json

    Path(graph_html_path).parent.mkdir(parents=True, exist_ok=True)

    centrality = nx.degree_centrality(knowledge_graph)

    nodes_data = []
    for node_id, attrs in knowledge_graph.nodes(data=True):
        radius = 18 + centrality[node_id] * 44
        nodes_data.append({
            "id": node_id,
            "label": node_id,
            "color": attrs.get("color", "#888888"),
            "radius": round(radius, 1),
            "node_type": attrs.get("node_type", "unknown"),
            "count": attrs.get("count", 0),
            "tooltip": attrs.get("title", node_id).replace("<br>", "\n").replace("<b>", "").replace("</b>", ""),
        })

    weights = [d.get("weight", 1) for _, _, d in knowledge_graph.edges(data=True)]
    max_weight = max(weights) if weights else 1

    edges_data = []
    for src, dst, attrs in knowledge_graph.edges(data=True):
        w = attrs.get("weight", 1)
        edges_data.append({
            "source": src,
            "target": dst,
            "weight": w,
            "width": round(0.8 + (w / max_weight) * 5, 2),
            "relationship": attrs.get("relationship", ""),
            "tooltip": attrs.get("title", "").replace("<br>", "\n"),
        })

    graph_json = json.dumps({"nodes": nodes_data, "links": edges_data})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
  svg {{ width: 100vw; height: 100vh; display: block; }}
  .node circle {{ cursor: pointer; transition: filter 0.2s; }}
  .node circle:hover {{ filter: brightness(1.4); }}
  .node text {{
    fill: #ffffff;
    font-size: 11px;
    font-weight: 600;
    pointer-events: none;
    text-anchor: middle;
    dominant-baseline: central;
    text-shadow: 0 0 4px #000;
  }}
  .link {{ fill: none; stroke-opacity: 0.45; transition: stroke-opacity 0.2s; }}
  .link:hover {{ stroke-opacity: 0.9; }}
  #tooltip {{
    position: fixed;
    background: rgba(13,17,23,0.92);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px;
    color: #e6edf3;
    font-size: 12px;
    line-height: 1.6;
    max-width: 220px;
    padding: 8px 12px;
    pointer-events: none;
    white-space: pre-line;
    display: none;
    z-index: 10;
    box-shadow: 0 4px 20px rgba(0,0,0,0.6);
  }}
</style>
</head>
<body>
<svg id="graph"></svg>
<div id="tooltip"></div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const graphData = {graph_json};

const W = window.innerWidth, H = window.innerHeight;
const svg = d3.select("#graph");
const defs = svg.append("defs");

// glow filter
const filter = defs.append("filter").attr("id", "glow").attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
filter.append("feGaussianBlur").attr("stdDeviation", "3.5").attr("result", "coloredBlur");
const feMerge = filter.append("feMerge");
feMerge.append("feMergeNode").attr("in", "coloredBlur");
feMerge.append("feMergeNode").attr("in", "SourceGraphic");

// arrow marker (unused but keeps edge direction concept)
const g = svg.append("g");

// zoom
svg.call(d3.zoom().scaleExtent([0.2, 4]).on("zoom", e => g.attr("transform", e.transform)));

const tooltip = document.getElementById("tooltip");

// build maps
const nodeById = new Map(graphData.nodes.map(n => [n.id, n]));

const simulation = d3.forceSimulation(graphData.nodes)
  .force("link", d3.forceLink(graphData.links)
    .id(d => d.id)
    .distance(d => 120 + (1 - d.width / 6) * 80)
    .strength(0.6))
  .force("charge", d3.forceManyBody().strength(-600).distanceMax(500))
  .force("center", d3.forceCenter(W / 2, H / 2))
  .force("collide", d3.forceCollide(d => d.radius + 12).strength(0.8))
  .alphaDecay(0.025);

// edges
const link = g.append("g").selectAll("line")
  .data(graphData.links)
  .join("line")
  .attr("class", "link")
  .attr("stroke", "#6e7681")
  .attr("stroke-width", d => d.width)
  .on("mouseenter", (e, d) => showTip(e, d.tooltip || d.relationship))
  .on("mouseleave", hideTip)
  .on("mousemove", moveTip);

// nodes
const node = g.append("g").selectAll("g")
  .data(graphData.nodes)
  .join("g")
  .attr("class", "node")
  .call(d3.drag()
    .on("start", (e, d) => {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
    .on("drag", (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
    .on("end", (e, d) => {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}));

node.append("circle")
  .attr("r", d => d.radius)
  .attr("fill", d => d.color)
  .attr("stroke", "#ffffff")
  .attr("stroke-width", 1.5)
  .attr("filter", "url(#glow)")
  .on("mouseenter", (e, d) => {{ highlightNode(d); showTip(e, d.tooltip); }})
  .on("mouseleave", (e, d) => {{ unhighlightNode(); hideTip(); }})
  .on("mousemove", moveTip);

node.append("text")
  .each(function(d) {{
    const el = d3.select(this);
    const words = d.label.split(" ");
    if (words.length === 1) {{
      el.append("tspan").attr("x", 0).attr("dy", "0").text(d.label);
    }} else {{
      const mid = Math.ceil(words.length / 2);
      el.append("tspan").attr("x", 0).attr("dy", "-0.6em").text(words.slice(0, mid).join(" "));
      el.append("tspan").attr("x", 0).attr("dy", "1.2em").text(words.slice(mid).join(" "));
    }}
  }});

simulation.on("tick", () => {{
  link
    .attr("x1", d => d.source.x)
    .attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x)
    .attr("y2", d => d.target.y);
  node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
}});

function highlightNode(d) {{
  const connected = new Set([d.id]);
  graphData.links.forEach(l => {{
    if (l.source.id === d.id) connected.add(l.target.id);
    if (l.target.id === d.id) connected.add(l.source.id);
  }});
  node.select("circle").attr("opacity", n => connected.has(n.id) ? 1 : 0.2);
  link.attr("stroke-opacity", l => (l.source.id === d.id || l.target.id === d.id) ? 0.9 : 0.08)
      .attr("stroke", l => (l.source.id === d.id || l.target.id === d.id) ? "#ffffff" : "#6e7681");
  node.select("text").attr("opacity", n => connected.has(n.id) ? 1 : 0.2);
}}

function unhighlightNode() {{
  node.select("circle").attr("opacity", 1);
  link.attr("stroke-opacity", 0.45).attr("stroke", "#6e7681");
  node.select("text").attr("opacity", 1);
}}

function showTip(e, text) {{
  tooltip.textContent = text;
  tooltip.style.display = "block";
  moveTip(e);
}}
function hideTip() {{ tooltip.style.display = "none"; }}
function moveTip(e) {{
  const x = e.clientX + 14, y = e.clientY - 10;
  tooltip.style.left = (x + 220 > window.innerWidth ? x - 240 : x) + "px";
  tooltip.style.top = y + "px";
}}
</script>
</body>
</html>"""

    Path(graph_html_path).write_text(html, encoding="utf-8")
    logger.info("Saved D3.js knowledge graph HTML to %s", graph_html_path)

    metadata = pd.DataFrame([
        {
            "node": n,
            "node_type": d.get("node_type", "unknown"),
            "degree": knowledge_graph.degree(n),
            "centrality": round(centrality[n], 4),
            "patient_count": d.get("count", 0),
        }
        for n, d in knowledge_graph.nodes(data=True)
    ]).sort_values("degree", ascending=False)

    return metadata
