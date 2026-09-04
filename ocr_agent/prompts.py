from __future__ import annotations

import json
from typing import Iterable, List

from .capability_constraints import ActionItem


SYSTEM_PROMPT = (
    "You are an OCR reasoning assistant. Use only visible evidence from the image, "
    "the question, and the provided history. Do not claim to use external tools. "
    "Follow the output format explicitly requested in the user prompt. Keep all answers and evidence concise."
)


def build_initial_prompt(
    question: str,
    document_context: str,
    enable_chain_of_thought: bool,
    task_type: str = "",
    dataset_name: str = "",
) -> str:
    task_requires_structured_output = _task_requires_structured_output(question, document_context)
    if enable_chain_of_thought:
        # Keep the answer separately parseable so CoT does not pollute evaluation.
        reasoning_requirement = 'Reason step by step, then return strict JSON with keys "reasoning" and "answer". Put only the task answer in "answer".'
    else:
        reasoning_requirement = "Answer directly and briefly."
    output_constraints = _build_output_constraints(document_context)
    translation_constraints = _build_translation_constraints(question, document_context)
    return (
        f"{translation_constraints}"
        f"{_build_task_description(question, task_type, dataset_name)}"
        f"Question: {question}\n"
        f"Document context: {document_context or 'N/A'}\n"
        f"{reasoning_requirement}\n"
        "Return the answer grounded in visible textual or layout evidence.\n"
        "Instructions:\n"
        "- When a reasoning and answer JSON wrapper is requested, keep reasoning concise and put the evaluated answer in answer.\n"
        "- For non-reasoning variants, output only the final answer.\n"
        f"- Output requirement: {output_constraints}\n"
        "- Do not add explanation unless the task explicitly requires reasoning text.\n"
    )


def build_reflection_prompt(
    question: str,
    document_context: str,
    previous_answer: str,
    memory: Iterable[str],
    reflection_language: str,
    use_memory: bool,
    task_type: str = "",
    dataset_name: str = "",
) -> str:
    memory_text = "\n".join(f"- {item}" for item in memory) if use_memory else "N/A"
    return (
        "Check whether an OCR answer needs correction.\n"
        f"Language: {reflection_language}\n"
        f"{_build_task_description(question, task_type, dataset_name)}"
        f"Question: {question}\n"
        f"Document context: {document_context or 'N/A'}\n"
        f"Previous answer: {previous_answer}\n"
        f"Reflection memory: {memory_text}\n"
        "Instructions:\n"
        "- Use KEEP when the answer is supported or no specific visual error is found.\n"
        "- Use REVISE only when the image shows a concrete error or omission.\n"
        "- Evidence must quote real visible text, a visible value, or a visible location; never copy placeholder words from the schema.\n"
        "- The action must describe how to re-check the image or question alignment; do not directly say 'change the answer to X' unless the evidence itself proves X.\n"
        "- If evidence is empty, uncertain, or only a guess, return KEEP.\n"
        'Return only JSON: {"decision":"KEEP or REVISE","evidence":"","diagnosis":"short diagnosis","action":"one feasible action"}.\n'
    )


def build_revision_prompt(
    question: str,
    document_context: str,
    previous_answer: str,
    reflection_evidence: str,
    actions: List[ActionItem],
    memory: Iterable[str],
    use_memory: bool,
    task_type: str = "",
    dataset_name: str = "",
) -> str:
    feasible_action = [item.text for item in actions if item.feasible]
    blocked_action = [f"{item.text} [{item.reason}]" for item in actions if not item.feasible]
    memory_text = "\n".join(f"- {item}" for item in memory) if use_memory else "N/A"
    output_constraints = _build_output_constraints(document_context)
    translation_constraints = _build_translation_constraints(question, document_context)
    return (
        f"{translation_constraints}"
        "Refine the previous answer for an OCR-style visual question.\n"
        f"{_build_task_description(question, task_type, dataset_name)}"
        f"Question: {question}\n"
        f"Document context: {document_context or 'N/A'}\n"
        f"Previous answer: {previous_answer}\n"
        f"Reflection evidence: {reflection_evidence}\n"
        f"Feasible actions: {json.dumps(feasible_action, ensure_ascii=False)}\n"
        f"Blocked actions: {json.dumps(blocked_action, ensure_ascii=False)}\n"
        f"Reflection memory: {memory_text}\n"
        "Instructions:\n"
        "- Give a replacement only when the image supports it more directly than the previous answer.\n"
        "- Evidence must identify real visible text, value, or location supporting the replacement; never output placeholder evidence.\n"
        "- Follow feasible actions and ignore blocked actions.\n"
        "- If the evidence is uncertain, say so briefly.\n"
        '- Return only JSON: {"answer":"task-appropriate answer","evidence":"visible evidence for this answer"}.\n'
        f"- Output requirement: {output_constraints}\n"
    )


def build_verification_prompt(
    question: str,
    document_context: str,
    previous_answer: str,
    candidate_answer: str,
    reflection_evidence: str,
    revision_evidence: str,
    task_type: str = "",
    dataset_name: str = "",
) -> str:
    translation_constraints = _build_translation_constraints(question, document_context)
    return (
        f"{translation_constraints}"
        "Verify a proposed OCR answer correction using the image.\n"
        f"{_build_task_description(question, task_type, dataset_name)}"
        f"Question: {question}\n"
        f"Document context: {document_context or 'N/A'}\n"
        f"Previous answer: {previous_answer}\n"
        f"Candidate answer: {candidate_answer}\n"
        f"Reflection evidence: {reflection_evidence}\n"
        f"Candidate evidence: {revision_evidence}\n"
        "Instructions:\n"
        "- ACCEPT only when the image evidence supports the candidate and specifically contradicts the previous answer.\n"
        "- For counting, arithmetic, or multiple-choice questions, verify the calculation or option mapping before ACCEPT.\n"
        "- REJECT when evidence is merely related text, incomplete, ambiguous, or does not distinguish the two answers.\n"
        'Return only JSON: {"decision":"ACCEPT or REJECT","reason":"short evidence-based reason"}.\n'
    )


def build_blind_review_prompt(
    question: str,
    document_context: str,
    task_type: str = "",
    dataset_name: str = "",
) -> str:
    return (
        "Independently solve this OCR visual question from the image.\n"
        f"{_build_task_description(question, task_type, dataset_name)}"
        f"Question: {question}\n"
        f"Document context: {document_context or 'N/A'}\n"
        "Do not discuss any previous answer or proposed correction.\n"
        "For counting, arithmetic, or multiple-choice questions, check the count, calculation, or option mapping carefully.\n"
        'Return only JSON: {"answer":"task answer","evidence":"short visible evidence"}.\n'
    )


def _build_output_constraints(document_context: str) -> str:
    context = (document_context or "").lower()
    if "parsing" in context:
        return "Return the complete requested structure or Markdown. Do not abbreviate, summarize, or omit content."
    if any(keyword in context for keyword in ("full-page ocr", "text recognition", "formula recognition")):
        return "Return the complete requested transcription. Do not summarize or return only a fragment."
    if any(keyword in context for keyword in ("position", "grounding", "spotting")):
        return "Return only the requested location or coordinates in the requested format."
    return "Return only the shortest answer that directly answers the question."


def _build_task_description(question: str, task_type: str, dataset_name: str) -> str:
    """Tell the VLM what the benchmark asks, not only what text is visible."""
    type = (task_type or "").strip()
    datasets = (dataset_name or "").strip()
    text = f"{question}\n{type}\n{datasets}".lower()
    description = [
        f"Benchmark task type: {type or 'unknown'}\n",
        f"Benchmark source: {datasets or 'unknown'}\n",
    ]
    if any(keyword in text for keyword in ("translate", "translation", "translate_to")):
        description.extend(
            [
                "Task mode: answer the question, do not merely transcribe the image.\n",
                "Translation rule: output only the requested target-language answer. If the target is English, do not output Chinese source text.\n",
            ]
        )
    elif any(keyword in text for keyword in ("full-page ocr", "text recognition", "read all the text", "formula recognition")):
        description.append("Task mode: transcribe all requested visible content faithfully; do not summarize.\n")
    elif any(keyword in text for keyword in ("parsing", "html-formatted", "markdown format")):
        description.append("Task mode: produce the complete requested structure; do not replace it with a summary.\n")
    else:
        description.append("Task mode: answer the specific question, selecting or extracting the requested value rather than dumping unrelated OCR text.\n")
    return "".join(description)


def _build_translation_constraints(question: str, document_context: str) -> str:
    text = f"{question}\n{document_context}".lower()
    if not any(keyword in text for keyword in ("translate", "translation", "translate_to")):
        return ""
    return (
        "IMPORTANT TRANSLATION TASK: translate the requested content into the target language specified by the question.\n"
        "Return ONLY the translated content. Do not return the source-language text, OCR transcription, explanation, labels, or unrelated fields.\n"
        "If the question says 'to English', output English only; do not output Chinese characters.\n"
    )


def _task_requires_structured_output(question: str, document_context: str) -> bool:
    text = f"{question}\n{document_context}".lower()
    return any(keyword in text for keyword in (
        "json format", "jsonformat", "markdown format", "html-formatted",
        "nested python dict", "bounding box", "bbox", "coordinates",
        "parsing", "full-page ocr", "formula recognition",
    ))
