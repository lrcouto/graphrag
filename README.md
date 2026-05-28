# Healthcare GraphRAG with Kedro

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

A GraphRAG (Graph Retrieval-Augmented Generation) demo built with Kedro. Takes 55,500 synthetic patient records, builds a knowledge graph, indexes it into a vector database, and exposes an agentic Q&A interface — all orchestrated as a Kedro pipeline.

![Knowledge graph screenshot](docs/screenshot.png)

---

## What it demonstrates

- **Kedro as a GenAI orchestrator** — four modular pipelines take raw data all the way to an AI agent, with every intermediate dataset tracked in the catalog
- **GraphRAG pattern** — semantic search is automatically enriched with 1-hop graph neighbours before being passed to the LLM, giving the agent structural context that pure vector search misses
- **Agentic pipeline** — the `query_answering` pipeline runs an OpenAI function-calling agent that decides which tools to call and how many times
- **Kedro-Viz** — live pipeline DAG explorer shows the full data lineage from CSV to agent report

---

## Architecture

```
healthcare_dataset.csv (55,500 rows)
        │
        ▼  data_ingestion
        ├── cleaned_healthcare_data
        └── entity_summaries

        cleaned_healthcare_data + entity_summaries
        │
        ▼  graph_construction
        ├── knowledge_graph (NetworkX, 30 nodes / 120 edges)
        └── knowledge_graph.html (D3.js visualisation)

        entity_summaries + knowledge_graph
        │
        ▼  vector_indexing
        └── chroma_collection (18 docs, OpenAI embeddings)

        knowledge_graph + chroma_collection + agent_prompt
        │
        ▼  query_answering
        └── agent_report.json
```

### Four Kedro pipelines

| Pipeline | What it does |
|---|---|
| `data_ingestion` | Cleans the CSV, computes per-entity statistics (conditions, medications, insurers) |
| `graph_construction` | Builds a NetworkX graph of entity relationships, renders interactive D3.js HTML |
| `vector_indexing` | Generates 18 entity summary documents, embeds them with OpenAI, stores in ChromaDB |
| `query_answering` | Runs an OpenAI function-calling agent over a configurable list of sample questions |

### Knowledge graph

30 nodes across 6 entity types, 120 edges. Edge weight = patient count.

| Entity type | Count | Relationships |
|---|---|---|
| Medical Condition | 6 | `TREATED_WITH`, `COVERED_BY`, `ADMITTED_AS`, `SHOWS_RESULT`, `ASSOCIATED_WITH` |
| Medication | 5 | — |
| Insurance Provider | 5 | — |
| Admission Type | 3 | — |
| Test Result | 3 | — |
| Blood Type | 8 | — |

### Agent tools

The `query_answering` agent has two tools:

- **`search_knowledge_base`** — semantic search over ChromaDB, automatically enriched with 1-hop graph neighbours (the GraphRAG step)
- **`get_graph_context`** — targeted neighbour lookup for a named entity

---

## Setup

### Prerequisites

- Python 3.10+
- An OpenAI API key (required for `vector_indexing` and `query_answering`; the graph pipelines work without it)

### Install

```bash
pip install -r requirements.txt
pip install -e src/
```

### Configure credentials

Add your OpenAI API key to `conf/local/credentials.yml` (this file is gitignored):

```yaml
openai:
  api_key: "sk-..."
```

---

## Running the pipelines

```bash
# Full pipeline (requires OpenAI API key)
kedro run

# Graph only — fast (~2s), no API key needed
kedro run --pipelines data_ingestion,graph_construction

# Vector index only — re-embeds documents (costs API tokens, do this sparingly)
kedro run --pipelines vector_indexing

# Agent report only — runs the agent over the sample questions in parameters_query_answering.yml
kedro run --pipelines query_answering
```

To customise the sample questions the agent answers, edit `conf/base/parameters_query_answering.yml`.

---

## Streamlit app

The app provides three views:

1. **Knowledge Graph** — interactive D3.js force graph (drag, zoom, hover for details)
2. **Pipeline DAG** — live Kedro-Viz explorer showing the full data lineage
3. **Ask the Graph** — chat interface backed by the same agentic logic as the pipeline

```bash
streamlit run app/streamlit_app.py
```

The app reads credentials from `conf/local/credentials.yml` via Kedro's config system — no environment variables needed.

---

## Project structure

```
├── app/
│   └── streamlit_app.py          # Streamlit dashboard
├── conf/
│   ├── base/
│   │   ├── catalog.yml           # Dataset definitions
│   │   ├── parameters_*.yml      # Per-pipeline parameters
│   │   └── parameters_query_answering.yml  # Sample questions, ChromaDB config
│   └── local/
│       └── credentials.yml       # OpenAI key (gitignored)
├── data/
│   ├── 01_raw/                   # healthcare_dataset.csv
│   ├── 04_feature/               # knowledge_graph.pkl
│   ├── 06_models/chroma_db/      # ChromaDB persistent storage
│   ├── 08_reporting/             # knowledge_graph.html, agent_report.json
│   └── prompts/                  # healthcare_agent.json (LangChainPromptDataset)
└── src/graphrag/pipelines/
    ├── data_ingestion/
    ├── graph_construction/
    ├── vector_indexing/
    └── query_answering/
```

---

## Data

The dataset is a synthetic healthcare CSV with 55,500 rows and 15 columns (age, gender, blood type, condition, medication, insurer, admission type, test result, billing amount, etc.). Doctor and hospital columns are excluded from the graph — they have ~40,000 unique synthetic values with no meaningful structure.

Source: [Kaggle — Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
