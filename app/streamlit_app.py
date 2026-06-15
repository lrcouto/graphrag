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

import streamlit as st

# Make the Kedro project's src/ importable so we can reuse pipeline logic
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

BASE_DIR = Path(__file__).parent.parent
GRAPH_PATH = BASE_DIR / "data/04_feature/knowledge_graph.json"
GRAPH_HTML_PATH = BASE_DIR / "data/08_reporting/knowledge_graph.html"
CHROMA_PATH = str(BASE_DIR / "data/06_models/chroma_db")
ENTITY_SUMMARIES_PATH = BASE_DIR / "data/03_primary/entity_summaries.pkl"
SQLITE_PATH = BASE_DIR / "data/07_model_output/healthcare_stats.db"
VIZ_PORT = 4141

st.set_page_config(
    page_title="Healthcare Knowledge Graph",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #e6edf3; }
h1, h2, h3 { color: #58a6ff; }
.stChatMessage { background-color: #161b22; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Background servers ────────────────────────────────────────────────────────

@st.cache_resource
def start_viz_server() -> bool:
    """Launch a live Kedro-Viz server as a background subprocess."""
    # If something is already listening on the port, assume it's our server.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("localhost", VIZ_PORT)) == 0:
            return True

    subprocess.Popen(
        [
            sys.executable, "-m", "kedro", "viz", "run",
            "--host", "0.0.0.0",
            "--port", str(VIZ_PORT),
            "--no-browser",
        ],
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait up to 15 s for the server to become ready.
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


@st.cache_resource
def load_agent_tools():
    import chromadb
    from graphrag.pipelines.query_answering.nodes import build_graph_context_tool, build_search_tool
    graph = load_graph()
    collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection("healthcare_knowledge")
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Healthcare Knowledge Graph")
    st.caption("Powered by **Kedro**")
    st.divider()

    st.markdown("""
**What is this?**

A GraphRAG pipeline built with Kedro that transforms 55,500 patient records into a
queryable knowledge graph — connecting medical conditions, treatments, insurers, and outcomes.

**Kedro orchestrates three storage backends simultaneously:**
1. 📥 Raw data ingestion & cleaning
2. 🕸️ Knowledge graph → JSON *(Neo4j: one catalog line)*
3. 📋 Entity statistics → SQLite *(relational store)*
4. 🔍 Vector index → ChromaDB *(semantic search)*
5. 💬 Graph-augmented Q&A with OpenAI
    """)

    st.divider()
    st.markdown("**Node Legend**")
    legend = [
        ("#E74C3C", "Medical Condition"),
        ("#3498DB", "Medication"),
        ("#2ECC71", "Insurance Provider"),
        ("#F39C12", "Admission Type"),
        ("#9B59B6", "Test Result"),
        ("#1ABC9C", "Blood Type"),
    ]
    for color, label in legend:
        st.markdown(
            f'<span style="background:{color};display:inline-block;width:12px;height:12px;'
            f'border-radius:50%;margin-right:8px;vertical-align:middle;"></span>{label}',
            unsafe_allow_html=True,
        )

    st.divider()

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
                lines.append(_ANSI_ESCAPE.sub("", line))
                log.code("".join(lines), language="text")
            proc.wait()
            if proc.returncode == 0:
                status.update(label=f"{label} — done ✓", state="complete", expanded=True)
                st.cache_resource.clear()
                st.cache_data.clear()
            else:
                status.update(label=f"{label} — failed ✗", state="error", expanded=True)

    if st.button("▶ Run Graph Pipeline", type="primary", width='stretch',
                 help="Builds the knowledge graph and relational store from all records (~2s, no API key needed)"):
        _run_pipelines(["data_ingestion", "graph_construction"], "Running graph pipeline…")

    if st.button("🔄 Update Graph", width='stretch',
                 help="Merges the 5,000 most recent patient records into the existing ontology — demonstrates incremental create/update"):
        _run_pipelines(["graph_update"], "Merging new patient batch into ontology…")

    if st.button("🔍 Rebuild Vector Index", width='stretch',
                 help="Runs the full pipeline including embeddings (requires OpenAI key in credentials.yml)"):
        _run_pipelines(["data_ingestion", "graph_construction", "vector_indexing", "query_answering"], "Running full pipeline…")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="display:flex;align-items:baseline;gap:16px;margin-bottom:0.5rem;">'
    '<span style="font-size:1.4rem;font-weight:700;color:#58a6ff;">Healthcare Knowledge Graph</span>'
    '<span style="font-size:0.85rem;color:#8b949e;">55,500 patient records · 6 conditions · 5 medications · 5 insurers · 8 blood types</span>'
    "</div>",
    unsafe_allow_html=True,
)

# ── Load artifacts ────────────────────────────────────────────────────────────
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

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_graph, tab_pipeline, tab_data, tab_chat = st.tabs([
    "🕸️  Knowledge Graph",
    "⚙️  Pipeline DAG",
    "📋  Structured Store",
    "💬  Ask the Graph",
])

# ── Tab 1: Knowledge Graph ────────────────────────────────────────────────────
with tab_graph:
    if graph_loaded:
        col_title, col_stats = st.columns([3, 1])
        with col_title:
            st.markdown("### Knowledge Graph")
            st.caption("Drag nodes · scroll to zoom · hover for details")
        with col_stats:
            st.metric("Nodes", graph.number_of_nodes())
            st.metric("Edges", graph.number_of_edges())

        if GRAPH_HTML_PATH.exists():
            st.iframe(GRAPH_HTML_PATH, height=820)
        else:
            st.warning("Graph HTML not found — run the pipeline to generate it.")
    else:
        st.info("Pipeline artifacts not found. Click **Rebuild Vector Index** in the sidebar to run the full pipeline.")

# ── Tab 2: Pipeline DAG ───────────────────────────────────────────────────────
with tab_pipeline:
    st.markdown("### Kedro Pipeline DAG")
    st.markdown(
        "The full pipeline is visualised in **Kedro-Viz** — "
        "an interactive DAG explorer showing every node, dataset, and parameter."
    )

    if viz_available:
        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            st.link_button(
                "Open Kedro-Viz ↗",
                url=f"http://localhost:{VIZ_PORT}",
                type="primary",
                width='stretch',
            )
        with col_info:
            st.caption(f"Live Kedro-Viz server · http://localhost:{VIZ_PORT}")

        st.divider()
        st.markdown("#### What you'll see")
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
            st.markdown("- `build_agent_context` → `LLMContextNode`\n- `run_agent`")

        st.divider()
        st.markdown("#### Ontology updates — `graph_update` pipeline")
        st.markdown(
            "Run `kedro run --pipelines graph_update` (or click **🔄 Update Graph**) "
            "to merge the 5,000 most recent records into the existing graph. "
            "New entities are created; existing nodes and edges get updated counts."
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
            "                                                    build_agent_context_node (LLMContextNode)\n"
            "                                                                    │\n"
            "                                                             agent_context\n"
            "                                                                    │\n"
            "                                                             run_agent_node\n"
            "                                                                    │\n"
            "                                                             agent_report",
            language="text",
        )
    else:
        st.info(
            "Kedro-Viz failed to start. "
            "Make sure `kedro-viz` is installed in the current Python environment."
        )

# ── Tab 3: Structured Store ───────────────────────────────────────────────────
with tab_data:
    st.markdown("### Relational Data Store")
    st.markdown(
        "The same Kedro pipeline that builds the knowledge graph and vector index also writes "
        "structured entity statistics to **SQLite** — demonstrating that a single pipeline can "
        "feed multiple storage backends simultaneously."
    )

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
                width='stretch',
                height=420,
            )
            st.caption(f"SQLite · `{SQLITE_PATH.relative_to(BASE_DIR)}` · {len(entity_stats)} rows")
        except Exception as e:
            st.error(f"Could not read SQLite store: {e}")
    else:
        st.info("Run **▶ Run Graph Pipeline** to populate the structured store.")

    st.divider()
    st.markdown("#### Storage architecture")
    st.code(
        "kedro run --pipelines data_ingestion,graph_construction\n"
        "\n"
        "  cleaned_healthcare_data ──┬── knowledge_graph  →  networkx.JSONDataset  (graph store)\n"
        "                            │                        ↕ swap to Neo4j via catalog\n"
        "  entity_summaries        ──┼── entity_stats     →  pandas.SQLTableDataset (relational)\n"
        "                            │\n"
        "  rag_documents           ──┴── chroma_collection →  ChromaDBDataset       (vector store)",
        language="text",
    )


# ── Tab 4: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown("### Ask the Knowledge Graph")

    if not rag_ready:
        if _rag_load_error is not None:
            st.error(f"Failed to load Q&A components: {_rag_load_error}")
            st.exception(_rag_load_error)
        else:
            st.warning(
                "Run the full pipeline (including vector indexing) to enable Q&A. "
                "The graph visualization works without it."
            )
    else:
        SAMPLE_QUESTIONS = [
            "Which medical conditions have the highest billing costs?",
            "What medications are most commonly prescribed for cancer?",
            "Which insurance provider covers the most emergency admissions?",
            "How do test results vary across different conditions?",
        ]

        if "messages" not in st.session_state:
            st.session_state.messages = [{
                "role": "assistant",
                "content": (
                    "Hi! I can answer questions about this healthcare dataset using the knowledge graph. "
                    "Try one of these:\n\n"
                    + "\n".join(f"- *{q}*" for q in SAMPLE_QUESTIONS)
                ),
            }]

        chat_container = st.container(height=520)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about conditions, medications, insurers, or outcomes…"):
            from graphrag.pipelines.query_answering.nodes import _run_agent

            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Agent is searching the knowledge graph…"):
                        result = _run_agent(
                            question=prompt,
                            prompt_template=agent_prompt,
                            openai_client=openai_client,
                            search_tool=search_tool,
                            graph_context_tool=graph_context_tool,
                        )

                    st.markdown(result["answer"])

                    if result["tool_calls"]:
                        with st.expander(f"🔧 {len(result['tool_calls'])} tool call(s) · {result['iterations']} iteration(s)"):
                            for tc in result["tool_calls"]:
                                st.markdown(f"**`{tc['tool']}`** — `{tc['args']}`")
                                st.text(tc["result"])

            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
