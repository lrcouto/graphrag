"""Healthcare Knowledge Graph — Streamlit demo app."""
import json
import pickle
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
# OSC 8 hyperlinks: ESC ] 8 ; params ; uri ST  visible-text  ESC ] 8 ;; ST
# ST is either BEL (\x07) or ESC \ (\x1b\). Keep visible text, drop the rest.
_OSC8_LINK = re.compile(
    r"\x1b\]8;[^;]*;[^\x07\x1b]*(?:\x07|\x1b\\)(.*?)\x1b\]8;;(?:\x07|\x1b\\)",
    re.DOTALL,
)
# Any remaining OSC sequences not caught above.
_OSC_STRIP = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

from ansi2html import Ansi2HTMLConverter
_ANSI_CONV = Ansi2HTMLConverter(inline=True, dark_bg=True)

import streamlit as st

_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

BASE_DIR = Path(__file__).parent.parent
GRAPH_PATH = BASE_DIR / "data/04_feature/knowledge_graph.json"
GRAPH_HTML_PATH = BASE_DIR / "data/08_reporting/knowledge_graph.html"
CHROMA_PATH = str(BASE_DIR / "data/06_models/chroma_db")
ENTITY_SUMMARIES_PATH = BASE_DIR / "data/03_primary/entity_summaries.pkl"
SQLITE_PATH = BASE_DIR / "data/07_model_output/healthcare_stats.db"
RAW_DATA_PATH = BASE_DIR / "data/01_raw/healthcare_dataset.csv"
VIZ_PORT = 4141

st.set_page_config(
    page_title="Healthcare GraphRAG · Powered by Kedro",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #e6edf3; }
h1, h2, h3 { color: #58a6ff; }
.stChatMessage { background-color: #161b22; border-radius: 8px; }

/* ── Hero ── */
.hero-wrap { padding: 4rem 0 3rem; }
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: #e6edf3;
    line-height: 1.15;
    margin: 0 0 0.75rem;
}
.hero-title span { color: #58a6ff; }
.hero-sub {
    font-size: 1.2rem;
    color: #8b949e;
    margin-bottom: 1.75rem;
    max-width: 680px;
    line-height: 1.65;
}
.hero-badge {
    display: inline-block;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    color: #8b949e;
    margin-right: 8px;
    margin-bottom: 8px;
}
.scroll-hint {
    text-align: center;
    color: #30363d;
    font-size: 0.9rem;
    margin-top: 3rem;
    letter-spacing: 0.08em;
}

/* ── Cards ── */
.pillar-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.5rem 1.5rem 1.75rem;
    height: 100%;
}
.pillar-icon { font-size: 1.75rem; margin-bottom: 0.75rem; }
.pillar-title {
    font-size: 1rem;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 0.6rem;
}
.pillar-body { font-size: 0.9rem; color: #8b949e; line-height: 1.65; }

/* ── Step headers ── */
.step-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 4rem 0 0.75rem;
}
.step-num {
    background: #1f6feb;
    color: #fff;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1rem;
    flex-shrink: 0;
}
.step-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #58a6ff;
    margin-bottom: 2px;
}
.step-title { font-size: 1.6rem; font-weight: 700; color: #e6edf3; line-height: 1.2; }
.step-body {
    color: #8b949e;
    font-size: 1rem;
    line-height: 1.7;
    max-width: 740px;
    margin-bottom: 1.5rem;
}

/* ── Comparison columns ── */
.compare-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 0.4rem;
}
.compare-tag {
    display: inline-block;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}
.tag-rag { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
.tag-graphrag { background: #0d419d; color: #79c0ff; border: 1px solid #1f6feb; }

/* ── Node legend dot ── */
.legend-dot {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)


# ── Cache functions ────────────────────────────────────────────────────────────

@st.cache_resource
def start_viz_server() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("localhost", VIZ_PORT)) == 0:
            return True
    subprocess.Popen(
        [sys.executable, "-m", "kedro", "viz", "run",
         "--host", "0.0.0.0", "--port", str(VIZ_PORT), "--no-browser"],
        cwd=str(BASE_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(15):
        time.sleep(1)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", VIZ_PORT)) == 0:
                return True
    return False


@st.cache_resource
def load_graph():
    import networkx as nx
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        return nx.node_link_graph(json.load(f))


@st.cache_resource
def load_entity_summaries():
    with open(ENTITY_SUMMARIES_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_entity_stats():
    import sqlite3
    import pandas as pd
    conn = sqlite3.connect(str(SQLITE_PATH))
    df = pd.read_sql("SELECT * FROM entity_statistics ORDER BY patient_count DESC", conn)
    conn.close()
    return df


@st.cache_data
def load_raw_sample():
    import pandas as pd
    return pd.read_csv(RAW_DATA_PATH, nrows=6)


@st.cache_resource
def load_chroma_collection():
    import chromadb
    return chromadb.PersistentClient(path=CHROMA_PATH).get_collection("healthcare_knowledge")


@st.cache_resource
def load_agent_tools():
    from graphrag.pipelines.query_answering.nodes import build_graph_context_tool, build_search_tool
    graph = load_graph()
    collection = load_chroma_collection()
    search_tool = build_search_tool(knowledge_graph=graph, chroma_collection=collection)
    graph_context_tool = build_graph_context_tool(graph)
    return search_tool, graph_context_tool


@st.cache_resource
def load_openai_client():
    from openai import OpenAI
    from graphrag.utils import get_openai_api_key
    return OpenAI(api_key=get_openai_api_key())


@st.cache_resource
def load_agent_prompt():
    from kedro_datasets_experimental.langchain import PromptDataset
    ds = PromptDataset(
        filepath=str(BASE_DIR / "data/prompts/healthcare_agent.json"),
        template="ChatPromptTemplate",
        dataset={"type": "json.JSONDataset"},
    )
    return ds.load()


# ── Pipeline runner ────────────────────────────────────────────────────────────

def _run_pipelines(pipelines: list[str], label: str):
    import os
    cmd = [sys.executable, "-m", "kedro", "run", "--pipelines", ",".join(pipelines)]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "FORCE_COLOR": "1"}
    with st.status(label, expanded=True) as status:
        log = st.empty()
        lines: list[str] = []
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(BASE_DIR), env=env,
        )
        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)
            clean = _OSC_STRIP.sub("", _OSC8_LINK.sub(r"\1", line))
            lines.append(clean)
            html_body = _ANSI_CONV.convert("".join(lines), full=False)
            log.html(
                '<div id="kedro-log" style="'
                "height:360px;overflow-y:auto;"
                "background:#0d1117;"
                "font-family:ui-monospace,SFMono-Regular,monospace;"
                "font-size:0.78rem;line-height:1.5;"
                "white-space:pre-wrap;"
                'padding:0.75rem;border-radius:6px;border:1px solid #30363d;">'
                f"{html_body}"
                "<script>"
                "var el=document.getElementById('kedro-log');"
                "if(el)el.scrollTop=el.scrollHeight;"
                "</script>"
                "</div>"
            )
        proc.wait()
        if proc.returncode == 0:
            status.update(label=f"{label} — done ✓", state="complete", expanded=True)
            st.cache_resource.clear()
            st.cache_data.clear()
        else:
            status.update(label=f"{label} — failed ✗", state="error", expanded=True)


# ── Plain RAG (no graph enrichment) ───────────────────────────────────────────

def _plain_rag(question: str) -> dict:
    client = load_openai_client()
    collection = load_chroma_collection()

    embed_response = client.embeddings.create(model="text-embedding-3-small", input=question)
    query_embedding = embed_response.data[0].embedding

    results = collection.query(query_embeddings=[query_embedding], n_results=4)
    docs = results["documents"][0]

    context = "\n\n---\n\n".join(docs)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "You are a healthcare data analyst. Answer the user's question "
                "based only on the provided context. Be concise and specific."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return {"answer": completion.choices[0].message.content, "context_docs": docs}


# ── Load artifacts ─────────────────────────────────────────────────────────────

graph_loaded = False
rag_ready = False

try:
    graph = load_graph()
    entity_summaries = load_entity_summaries()
    graph_loaded = True
except Exception:
    pass

_rag_load_error: Exception | None = None
try:
    openai_client = load_openai_client()
    agent_prompt = load_agent_prompt()
    search_tool, graph_context_tool = load_agent_tools()
    rag_ready = True
except Exception as _e:
    _rag_load_error = _e

viz_available = start_viz_server()


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_story, tab_pipeline, tab_chat = st.tabs([
    "🏥  The Story",
    "⚙️  Pipeline",
    "💬  Ask the Graph",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — The Story
# ═══════════════════════════════════════════════════════════════════════════════

with tab_story:

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="hero-wrap">
  <div class="hero-title">Healthcare <span>GraphRAG</span></div>
  <div class="hero-sub">
    55,500 patient records transformed into a queryable knowledge graph —
    see how <strong style="color:#e6edf3;">Kedro</strong> orchestrates the full pipeline,
    from raw CSV to graph-augmented Q&amp;A, across three storage backends simultaneously.
  </div>
  <div>
    <span class="hero-badge">55,500 synthetic patient records</span>
    <span class="hero-badge">30-node knowledge graph</span>
    <span class="hero-badge">3 storage backends</span>
    <span class="hero-badge">OpenAI GPT-4o</span>
    <span class="hero-badge">Powered by Kedro</span>
    <span class="hero-badge" style="border-color:#58a6ff;color:#58a6ff;">⚠ Synthetic data — no real patients</span>
  </div>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.markdown("""
<div class="pillar-card">
  <div class="pillar-icon">🔗</div>
  <div class="pillar-title">The Problem</div>
  <div class="pillar-body">
    Vector search finds similar text — but medical data is <em>relational</em>.
    Which conditions share treatments? Which insurers see the worst outcomes?
    Flat embeddings lose those connections.
  </div>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="pillar-card">
  <div class="pillar-icon">🕸️</div>
  <div class="pillar-title">GraphRAG</div>
  <div class="pillar-body">
    A knowledge graph captures entity relationships explicitly.
    At query time the system combines vector search with graph traversal —
    giving the LLM both relevant documents <em>and</em> the relationships that explain them.
  </div>
</div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
<div class="pillar-card">
  <div class="pillar-icon">⚡</div>
  <div class="pillar-title">Why Kedro</div>
  <div class="pillar-body">
    Kedro orchestrates the entire pipeline — raw CSV → graph → embeddings → Q&amp;A —
    across three storage backends simultaneously. Swap any backend with one line
    in the Data Catalog. No pipeline code changes needed.
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="scroll-hint">↓ &nbsp; scroll to walk through the pipeline</div>',
                unsafe_allow_html=True)

    # ── Step 1: The Raw Data ───────────────────────────────────────────────────
    st.markdown("""
<div class="step-header">
  <div class="step-num">1</div>
  <div>
    <div class="step-label">Step 1</div>
    <div class="step-title">The Raw Data</div>
  </div>
</div>
<div class="step-body">
  Everything begins with a flat CSV — 55,500 anonymised patient records, each describing
  a medical condition, prescribed medication, insurance provider, admission type,
  diagnostic test result, and blood type. On its own it is just rows and columns.
  Kedro's <code>data_ingestion</code> pipeline cleans it and extracts the entities
  that become knowledge graph nodes.
</div>
""", unsafe_allow_html=True)

    if RAW_DATA_PATH.exists():
        raw_sample = load_raw_sample()
        DISPLAY_COLS = [
            "Name", "Age", "Medical Condition", "Medication",
            "Insurance Provider", "Admission Type", "Test Results", "Blood Type",
        ]
        cols_to_show = [c for c in DISPLAY_COLS if c in raw_sample.columns]
        st.dataframe(raw_sample[cols_to_show] if cols_to_show else raw_sample,
                     width="stretch", height=245)
        st.caption("First 6 rows · `data/01_raw/healthcare_dataset.csv` · 55,500 total records · Synthetic data from Kaggle — no real patient information")
    else:
        st.info("Raw data not found at `data/01_raw/healthcare_dataset.csv`.")

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "Runs `data_ingestion` + `graph_construction` — cleans the records, "
        "extracts entity summaries, writes statistics to SQLite, and builds the knowledge graph. "
        "Takes ~2 seconds. No API key needed."
    )
    if st.button("▶ Run Graph Pipeline", type="primary",
                 help="Runs data_ingestion + graph_construction (~2s, no API key needed)"):
        _run_pipelines(["data_ingestion", "graph_construction"], "Running graph pipeline…")

    # ── Step 2: The Knowledge Graph ────────────────────────────────────────────
    st.markdown("""
<div class="step-header">
  <div class="step-num">2</div>
  <div>
    <div class="step-label">Step 2</div>
    <div class="step-title">The Knowledge Graph</div>
  </div>
</div>
<div class="step-body">
  The <code>graph_construction</code> pipeline distils 55,500 records into 30 entity nodes
  across 6 types and 120 typed edges —
  <em>TREATED_WITH</em>, <em>COVERED_BY</em>, <em>ADMITTED_AS</em>,
  <em>SHOWS_RESULT</em>, <em>ASSOCIATED_WITH</em>.
  The graph is persisted as JSON via <code>networkx.JSONDataset</code>.
  The storage backend is swappable via the Kedro Data Catalog with no pipeline code changes.
</div>
""", unsafe_allow_html=True)

    if graph_loaded:
        col_viz, col_meta = st.columns([4, 1])
        with col_meta:
            st.metric("Nodes", graph.number_of_nodes())
            st.metric("Edges", graph.number_of_edges())
            st.markdown("<br>**Node types**", unsafe_allow_html=True)
            for color, label in [
                ("#E74C3C", "Medical Condition"),
                ("#3498DB", "Medication"),
                ("#2ECC71", "Insurance Provider"),
                ("#F39C12", "Admission Type"),
                ("#9B59B6", "Test Result"),
                ("#1ABC9C", "Blood Type"),
            ]:
                st.markdown(
                    f'<span class="legend-dot" style="background:{color};"></span>{label}',
                    unsafe_allow_html=True,
                )
        with col_viz:
            if GRAPH_HTML_PATH.exists():
                st.iframe(GRAPH_HTML_PATH, height=820)
            else:
                st.warning("Graph HTML not found — run **▶ Run Graph Pipeline** above to generate it.")

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown(
            "Runs the `graph_update` pipeline — merges the 5,000 most recent patient records "
            "into the existing graph without rebuilding from scratch. New entities are added; "
            "existing nodes and edges get updated counts. Demonstrates Kedro's incremental update pattern."
        )
        if st.button("🔄 Update Graph",
                     help="Merges the 5,000 most recent records into the existing graph"):
            _run_pipelines(["graph_update"], "Merging new patient batch into ontology…")
    else:
        st.info("Run **▶ Run Graph Pipeline** above to build and display the knowledge graph.")

    # ── Step 3: Three Storage Backends ─────────────────────────────────────────
    st.markdown("""
<div class="step-header">
  <div class="step-num">3</div>
  <div>
    <div class="step-label">Step 3</div>
    <div class="step-title">Three Storage Backends</div>
  </div>
</div>
<div class="step-body">
  One <code>kedro run</code> populates three stores simultaneously.
  Kedro's Data Catalog abstracts each backend — the pipeline code never touches
  file paths or connection strings. Swap any store by editing one line in
  <code>conf/base/catalog.yml</code>.
</div>
""", unsafe_allow_html=True)

    col_g, col_s, col_v = st.columns(3, gap="medium")
    with col_g:
        st.markdown("""
<div class="pillar-card">
  <div class="pillar-icon">🕸️</div>
  <div class="pillar-title">Graph Store</div>
  <div class="pillar-body">
    <code>networkx.JSONDataset</code><br><br>
    Persists the full graph with all node and edge attributes as portable JSON.
    Swap the backend via the Kedro Data Catalog —
    no pipeline code changes needed.
  </div>
</div>""", unsafe_allow_html=True)
    with col_s:
        st.markdown("""
<div class="pillar-card">
  <div class="pillar-icon">📋</div>
  <div class="pillar-title">Relational Store</div>
  <div class="pillar-body">
    <code>pandas.SQLTableDataset</code><br><br>
    Writes entity statistics to SQLite — patient counts, billing averages,
    and age distributions per entity. Point the connection string at PostgreSQL
    or BigQuery and the pipeline is unchanged.
  </div>
</div>""", unsafe_allow_html=True)
    with col_v:
        st.markdown("""
<div class="pillar-card">
  <div class="pillar-icon">🔍</div>
  <div class="pillar-title">Vector Store</div>
  <div class="pillar-body">
    <code>ChromaDBDataset</code><br><br>
    18 RAG documents — one per entity — embedded with
    <code>text-embedding-3-small</code> and indexed in ChromaDB
    for semantic search at query time.
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    if SQLITE_PATH.exists():
        try:
            entity_stats = load_entity_stats()
            col_filter, col_metric = st.columns([2, 1])
            with col_filter:
                types = ["All"] + sorted(entity_stats["entity_type"].dropna().unique().tolist())
                selected = st.selectbox("Filter by entity type", types)
            with col_metric:
                st.metric("Total entities", len(entity_stats))
            df_show = entity_stats if selected == "All" else entity_stats[entity_stats["entity_type"] == selected]
            st.dataframe(
                df_show.style.format({
                    "avg_billing": "${:,.2f}",
                    "avg_age": "{:.1f}",
                    "avg_stay": "{:.1f}",
                    "patient_count": "{:,}",
                }, na_rep="—"),
                width="stretch",
                height=380,
            )
            st.caption(f"SQLite · `{SQLITE_PATH.relative_to(BASE_DIR)}` · {len(entity_stats)} rows")
        except Exception as e:
            st.error(f"Could not read SQLite store: {e}")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "Runs the full pipeline including `vector_indexing` and `query_answering`. "
        "Embeds all 18 entity documents into ChromaDB using `text-embedding-3-small`. "
        "Requires an OpenAI key in `conf/local/credentials.yml`."
    )
    if st.button("🔍 Rebuild Vector Index",
                 help="Runs the full pipeline including embeddings (requires OpenAI key)"):
        _run_pipelines(
            ["data_ingestion", "graph_construction", "vector_indexing", "query_answering"],
            "Running full pipeline…",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

with tab_pipeline:
    st.markdown("### Kedro Pipeline DAG")
    st.markdown(
        "The full pipeline is visualised in **Kedro-Viz** — "
        "an interactive DAG explorer showing every node, dataset, and parameter."
    )

    if viz_available:
        col_btn_viz, col_info_viz = st.columns([1, 2], gap="large")
        with col_btn_viz:
            st.link_button("Open Kedro-Viz ↗", url=f"http://localhost:{VIZ_PORT}",
                           type="primary", width="stretch")
        with col_info_viz:
            st.caption(f"Live Kedro-Viz server · http://localhost:{VIZ_PORT}")
    else:
        st.warning("Kedro-Viz failed to start. Make sure `kedro-viz` is installed.")

    st.divider()
    st.markdown("#### Sub-pipelines")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**📥 data_ingestion**")
        st.markdown("- `clean_data`\n- `extract_entity_summaries`\n- `store_entity_stats` → SQLite")
    with col2:
        st.markdown("**🕸️ graph_construction**")
        st.markdown("- `build_knowledge_graph`\n- `render_graph_html`")
    with col3:
        st.markdown("**🔍 vector_indexing**")
        st.markdown("- `create_rag_documents`\n- `embed_documents` → ChromaDB")
    with col4:
        st.markdown("**🤖 query_answering**")
        st.markdown("- `build_agent_context` (LLMContextNode)\n- `run_agent`")

    st.divider()
    st.markdown("#### Ontology updates — `graph_update` pipeline")
    st.markdown(
        "Run `kedro run --pipelines graph_update` to merge the 5,000 most recent records "
        "into the existing graph. New entities are created; existing nodes and edges get "
        "updated counts. Excluded from `__default__` to avoid writing to the same output "
        "as `graph_construction`."
    )

    st.divider()
    st.markdown("#### Full dataset flow")
    st.code(
        "healthcare_dataset.csv\n"
        "  └─ cleaned_healthcare_data  ──┬─ knowledge_graph (networkx.JSONDataset)\n"
        "                               │               │\n"
        "                               └─ entity_summaries ─ rag_documents\n"
        "                                                        └─ chroma_collection\n"
        "                                                                    │\n"
        "openai_llm  ──────────────────────────────────────────────────────┐ │\n"
        "agent_prompt (LangChainPromptDataset)  ───────────────────────────┤ │\n"
        "knowledge_graph  ─────────────────────────────────────────────────┤ │\n"
        "                                                                   ▼ ▼\n"
        "                                            build_agent_context_node (LLMContextNode)\n"
        "                                                                    │\n"
        "                                                             agent_context\n"
        "                                                                    │\n"
        "                                                             run_agent_node\n"
        "                                                                    │\n"
        "                                                             agent_report",
        language="text",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Ask the Graph (GraphRAG vs Plain RAG)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.markdown("### GraphRAG vs Plain RAG")
    st.markdown(
        "Type a question below. Both systems search the same ChromaDB vector index — "
        "but **GraphRAG** enriches each result with 1-hop graph neighbours before "
        "calling GPT-4o. The expander under each answer shows exactly what context "
        "the model received."
    )

    col_rag_hdr, col_graph_hdr = st.columns(2, gap="large")
    with col_rag_hdr:
        st.markdown("""
<div class="compare-header">Plain RAG</div>
<span class="compare-tag tag-rag">vector search only</span>
<p style="color:#8b949e;font-size:0.9rem;line-height:1.6;">
Embeds the question, fetches the 4 most similar entity documents from ChromaDB,
and passes the raw text directly to GPT-4o. Fast and simple — but the model
sees documents in isolation, with no knowledge of how entities relate to each other.
</p>""", unsafe_allow_html=True)

    with col_graph_hdr:
        st.markdown("""
<div class="compare-header">GraphRAG</div>
<span class="compare-tag tag-graphrag">vector search + graph traversal</span>
<p style="color:#8b949e;font-size:0.9rem;line-height:1.6;">
Same ChromaDB search — but each retrieved document is then enriched with its
1-hop NetworkX neighbours. The model receives the semantic results
<em>and</em> the structural relationships between them, enabling more connected answers.
</p>""", unsafe_allow_html=True)

    st.divider()

    if not rag_ready:
        if _rag_load_error is not None:
            st.error(f"Failed to load Q&A components: {_rag_load_error}")
            st.exception(_rag_load_error)
        else:
            st.warning(
                "Run **🔍 Rebuild Vector Index** in The Story tab to enable Q&A. "
                "The graph visualization works without it."
            )
    else:
        SAMPLE_QUESTIONS = [
            "Which medical conditions have the highest billing costs?",
            "What medications are most commonly prescribed for cancer?",
            "Which insurance provider covers the most emergency admissions?",
            "How do test results vary across different conditions?",
        ]

        st.markdown(
            "<div style='color:#8b949e;font-size:0.85rem;margin-bottom:0.5rem;'>"
            "Try: " + " · ".join(f"<em>{q}</em>" for q in SAMPLE_QUESTIONS)
            + "</div>",
            unsafe_allow_html=True,
        )

        question = st.chat_input("Ask a question to compare both approaches…")

        if "comparison" not in st.session_state:
            st.session_state.comparison = None

        if question:
            from graphrag.pipelines.query_answering.nodes import _run_agent

            col_rag, col_graph = st.columns(2, gap="large")

            with col_rag:
                with st.spinner("Plain RAG searching…"):
                    rag_result = _plain_rag(question)

            with col_graph:
                with st.spinner("GraphRAG searching…"):
                    graphrag_result = _run_agent(
                        question=question,
                        prompt_template=agent_prompt,
                        openai_client=openai_client,
                        search_tool=search_tool,
                        graph_context_tool=graph_context_tool,
                    )

            st.session_state.comparison = {
                "question": question,
                "rag": rag_result,
                "graphrag": graphrag_result,
            }

        if st.session_state.comparison:
            comp = st.session_state.comparison
            st.markdown(
                f"<div style='color:#58a6ff;font-weight:600;margin-bottom:1rem;'>"
                f"Q: {comp['question']}</div>",
                unsafe_allow_html=True,
            )

            col_rag, col_graph = st.columns(2, gap="large")

            with col_rag:
                st.markdown(comp["rag"]["answer"])
                with st.expander("Context the model received"):
                    for i, doc in enumerate(comp["rag"]["context_docs"], 1):
                        st.markdown(f"**Document {i}**")
                        st.text(doc)

            with col_graph:
                st.markdown(comp["graphrag"]["answer"])
                if comp["graphrag"]["tool_calls"]:
                    with st.expander(
                        f"Context + graph enrichment · "
                        f"{len(comp['graphrag']['tool_calls'])} tool call(s)"
                    ):
                        for tc in comp["graphrag"]["tool_calls"]:
                            st.markdown(f"**`{tc['tool']}`** — `{tc['args']}`")
                            st.text(tc["result"])
