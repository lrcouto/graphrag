"""Query answering pipeline."""
from kedro.pipeline import Pipeline, node

from .nodes import run_agent


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=run_agent,
            inputs=[
                "agent_prompt",
                "knowledge_graph",
                "params:sample_questions",
                "params:chroma_db_path",
                "params:chroma_collection_name",
            ],
            outputs="agent_report",
            name="run_agent_node",
        ),
    ])
