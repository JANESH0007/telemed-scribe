"""
The TeleMed-Scribe workflow graph.

    START -> init -+-(entry=audio)---> transcribe -> extract -> medication_rag -> history_rag
                   |                                   ^                              |
                   +-(entry=extract)-------------------+                              v
                   |                                                         assemble_report
                   +-(entry=review)--> human_review <-- apply_edits                   |
                                          |  ^   (action=edit)                        v
                         (action=approve) |  +---------------------------------- persist_report
                                          v                                  (-> human_review)
                                       finalize -> END

`entry` lets one graph serve every way the API can start a run, and is also how a review is
re-attached after a restart (see runner.WorkflowRunner.review).
"""
from __future__ import annotations

from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from api.services import Services
from db import Store
from workflow.nodes import make_nodes
from workflow.state import WorkflowState


def route_entry(state: WorkflowState) -> str:
    return {"audio": "transcribe", "extract": "extract", "review": "human_review"}[state["entry"]]


def route_decision(state: WorkflowState) -> str:
    return "finalize" if state["decision"]["action"] == "approve" else "apply_edits"


def build_graph(store: Store, services: Services, checkpointer: Optional[BaseCheckpointSaver] = None):
    """Compile the workflow. Store/Services are closed over by the nodes (never put in state).
    A checkpointer is required for the human-review pause; the default is in-memory."""
    g = StateGraph(WorkflowState)
    for name, fn in make_nodes(store, services).items():
        g.add_node(name, fn)

    g.add_edge(START, "init")
    g.add_conditional_edges("init", route_entry,
                            {"transcribe": "transcribe", "extract": "extract", "human_review": "human_review"})
    g.add_edge("transcribe", "extract")
    g.add_edge("extract", "medication_rag")
    g.add_edge("medication_rag", "history_rag")
    g.add_edge("history_rag", "assemble_report")
    g.add_edge("assemble_report", "persist_report")
    g.add_edge("persist_report", "human_review")
    g.add_conditional_edges("human_review", route_decision,
                            {"finalize": "finalize", "apply_edits": "apply_edits"})
    g.add_edge("apply_edits", "human_review")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
