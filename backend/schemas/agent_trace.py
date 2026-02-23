from __future__ import annotations

from pydantic import BaseModel


class ThoughtNode(BaseModel):
    id: str
    content: str
    reasoning: str = ""
    expected_impact: float = 0.0
    confidence: float = 0.0
    score: float = 0.0
    children: list[ThoughtNode] = []
    selected: bool = False
    depth: int = 0


class ThoughtTree(BaseModel):
    root_thought: str
    branches: list[ThoughtNode] = []
    selected_path: list[str] = []
    iteration: int = 0


class AgentTraceOut(BaseModel):
    id: str
    experiment_id: str
    iteration: int | None = None
    agent_name: str
    step_type: str | None = None
    content: str = ""
    parent_trace_id: str | None = None
    depth: int = 0
    score: float | None = None
    selected: bool = False
    metadata: dict = {}
    created_at: str | None = None


class AgentStep(BaseModel):
    agent: str
    step: str
    content: str


class WSMessage(BaseModel):
    type: str
    data: dict = {}
