from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BackendConfig:
    backend_type: str = "mock"
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    api_protocol: str = "chat_completions"
    timeout_seconds: int = 120
    max_retries: int = 2
    request_interval_seconds: float = 0.0
    temperature: float = 0.0
    random_seed: Optional[int] = 42


@dataclass
class PipelineConfig:
    variant_name: str = "ocr-agent"
    reflection_language: str = "English"
    enable_capability_reflection: bool = True
    enable_memory_reflection: bool = True
    enable_verification: bool = False
    enable_blind_review: bool = False
    enable_answer_filter: bool = False
    enable_visual_tools: bool = False
    visual_tool_dir: str = ""
    blocked_action_patterns: List[str] = field(
        default_factory=lambda: [
            "enhance image",
            "retake picture",
            "zoom with external tool",
            "crop image externally",
            "search web",
            "call ocr api",
            "use another model",
            "open external file",
            "correct the answer to",
            "change the answer to",
            "replace the answer with",
            "answer should be",
        ]
    )


@dataclass
class SingleRequest:
    question: str
    image_paths: List[str]
    document_context: str = ""
    task_type: str = ""
    dataset_name: str = ""
    max_iterations: int = 3


@dataclass
class StepTrace:
    round_index: int
    previous_answer: str
    revised_answer: str
    previous_conclusion: str
    revised_conclusion: str
    text_changed: bool
    conclusion_changed: bool
    answer_changed: bool
    raw_reflection: str
    diagnosis: List[str]
    action_plan: List[str]
    avoid_repeated_prompt: str
    feasible_actions: List[Dict[str, Any]]
    feasible_action_texts: List[str]
    blocked_action_texts: List[str]
    feasible_action_count: int
    blocked_action_count: int
    memory_before: List[str]
    memory_after: List[str]
    structured_memory: Dict[str, Any]
    reflection_decision: str = ""
    reflection_parse_success: bool = False
    revision_requested: bool = False
    revision_evidence: List[str] = field(default_factory=list)
    revision_parse_success: bool = False
    revision_accepted: bool = False
    verification_requested: bool = False
    verification_parse_success: bool = False
    verification_decision: str = ""
    blind_review_requested: bool = False
    blind_review_parse_success: bool = False
    blind_review_answer: str = ""
    visual_tool_called: bool = False
    visual_tool_image_paths: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    variant_name: str
    initial_answer: str
    final_answer: str
    initial_conclusion: str
    final_conclusion: str
    text_changed: bool
    conclusion_changed: bool
    memory: List[str]
    traces: List[StepTrace]
    initial_reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant_name,
            "initial_answer": self.initial_answer,
            "final_answer": self.final_answer,
            "initial_conclusion": self.initial_conclusion,
            "initial_reasoning": self.initial_reasoning,
            "final_conclusion": self.final_conclusion,
            "text_changed": self.text_changed,
            "conclusion_changed": self.conclusion_changed,
            "answer_changed": self.text_changed,
            "total_iterations": len(self.traces),
            "memory": self.memory,
            "traces": [asdict(item) for item in self.traces],
        }


@dataclass
class DocVQASample:
    sample_id: str
    question: str
    image_path: str
    ground_truth_answers: List[str]
    document_context: str = ""
    task_type: str = "docvqa"
    data_split: str = ""
    dataset_name: str = ""


@dataclass
class DocVQAResult:
    sample_id: str
    task_type: str
    data_split: str
    question: str
    image_path: str
    ground_truth_answers: List[str]
    pipeline_output: Dict[str, Any]
    dataset_name: str = ""
