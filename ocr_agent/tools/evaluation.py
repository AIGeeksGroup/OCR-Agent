from __future__ import annotations

import json
import re
import ast
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List

from ..data_models import DocVQAResult


@dataclass
class MetricSummary:
    sample_count: int
    exact_match: float
    contains_answer: float
    anls: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_docvqa_results(result_list: Iterable[DocVQAResult]) -> MetricSummary:
    result_sequence = list(result_list)
    if not result_sequence:
        return MetricSummary(sample_count=0, exact_match=0.0, contains_answer=0.0, anls=0.0)

    exacthit_count = 0
    containhit_count = 0
    anls_total_score = 0.0

    for result in result_sequence:
        predicted_answer = str(result.pipeline_output.get("final_conclusion") or result.pipeline_output.get("final_answer") or "")
        ground_truth_answers = _normalize_answer_list(result.ground_truth_answers)

        if _is_exact_match(predicted_answer, ground_truth_answers):
            exacthit_count += 1
        if _is_contains_match(predicted_answer, ground_truth_answers):
            containhit_count += 1
        anls_total_score += _compute_single_sample_anls(predicted_answer, ground_truth_answers)

    sample_count = len(result_sequence)
    return MetricSummary(
        sample_count=sample_count,
        exact_match=round(exacthit_count / sample_count, 4),
        contains_answer=round(containhit_count / sample_count, 4),
        anls=round(anls_total_score / sample_count, 4),
    )


def evaluate_ocrbench_v2_results(result_list: Iterable[DocVQAResult]) -> dict:
    result_sequence = list(result_list)
    overall = _evaluate_ocrbench_v2_unified_results(result_sequence).to_dict()
    initial_final_comparison = _compare_initial_final_answers(result_sequence)
    per_type_results = {}
    for task_type in sorted({result.task_type for result in result_sequence}):
        type_result_list = [result for result in result_sequence if result.task_type == task_type]
        per_type_results[task_type] = {
            "metric_family": _task_metric_family(task_type),
            "metrics_note": _task_metric_description(task_type),
            "vqa_proxy": _evaluate_ocrbench_v2_unified_results(type_result_list).to_dict(),
            "initial_vs_final": _compare_initial_final_answers(type_result_list),
        }
    per_dataset_results = {}
    for dataset_name in sorted({result.dataset_name for result in result_sequence if result.dataset_name}):
        per_dataset_results[dataset_name] = _evaluate_ocrbench_v2_unified_results(
            result for result in result_sequence if result.dataset_name == dataset_name
        ).to_dict()
    return {
        "overall": overall,
        "by_type": per_type_results,
        "by_dataset_name": per_dataset_results,
        "initial_vs_final": initial_final_comparison,
        "reflection_audit": _summarize_reflection_audit(result_sequence),
        "evaluation_warning": "vqa_proxy is only a diagnostic proxy; official OCRBench-v2 task-specific metrics are still required for paper comparison.",
    }


def _compare_initial_final_answers(result_list: Iterable[DocVQAResult]) -> dict:
    result_sequence = list(result_list)
    initial_correct = 0
    final_correct = 0
    improved_count = 0
    worsened_count = 0
    keep_correct = 0
    keep_wrong = 0
    for result in result_sequence:
        ground_truth_answers = _normalize_answer_list(result.ground_truth_answers)
        initial_answer = str(result.pipeline_output.get("initial_conclusion") or result.pipeline_output.get("initial_answer") or "")
        final_answer = str(result.pipeline_output.get("final_conclusion") or result.pipeline_output.get("final_answer") or "")
        initial_hits = _is_exact_match(initial_answer, ground_truth_answers)
        final_hits = _is_exact_match(final_answer, ground_truth_answers)
        initial_correct += int(initial_hits)
        final_correct += int(final_hits)
        if initial_hits and not final_hits:
            worsened_count += 1
        elif not initial_hits and final_hits:
            improved_count += 1
        elif initial_hits:
            keep_correct += 1
        else:
            keep_wrong += 1
    sample_count = len(result_sequence)
    return {
        "sample_count": sample_count,
        "initial_exact_count": initial_correct,
        "final_exact_count": final_correct,
        "improved_after_reflection": improved_count,
        "worsened_after_reflection": worsened_count,
        "keep_correct_count": keep_correct,
        "keep_wrong_count": keep_wrong,
        "initial_accuracy": round(initial_correct / sample_count, 4) if sample_count else 0.0,
        "final_accuracy": round(final_correct / sample_count, 4) if sample_count else 0.0,
    }


def _summarize_reflection_audit(result_list: Iterable[DocVQAResult]) -> dict:
    traces = [trace for result in result_list for trace in result.pipeline_output.get("traces", [])]
    def count(field_name: str, target_value: object = True) -> int:
        return sum(trace.get(field_name) == target_value for trace in traces)
    def verification_count(decision: str) -> int:
        return sum(
            trace.get("verification_requested") and trace.get("verification_decision") == decision
            for text in traces
        )

    return {
        "reflection_round_count": len(traces),
        "reflection_json_parse_success_count": count("reflection_parse_success"),
        "decision_keep_count": count("reflection_decision", "KEEP"),
        "decision_revise_count": count("reflection_decision", "REVISE"),
        "revision_request_count": count("revision_requested"),
        "revision_json_parse_success_count": count("revision_parse_success"),
        "verification_request_count": count("verification_requested"),
        "verification_json_parse_success_count": count("verification_parse_success"),
        "verification_accept_count": verification_count("ACCEPT"),
        "verification_reject_count": verification_count("REJECT"),
        "verification_not_triggered_count": len(traces) - count("verification_requested"),
        "blind_review_request_count": count("blind_review_requested"),
        "blind_review_json_parse_success_count": count("blind_review_parse_success"),
        "revision_accept_count": count("revision_accepted"),
    }


def _task_metric_family(task_type: str) -> str:
    type = task_type.lower()
    if any(keyword in type for keyword in ("parsing", "html", "table")):
        return "parsing"
    if any(keyword in type for keyword in ("spotting", "grounding", "position")):
        return "localization"
    if any(keyword in type for keyword in ("extraction", "mapping")):
        return "extraction"
    if "counting" in type:
        return "counting"
    if any(keyword in type for keyword in ("recognition", "ocr", "formula")):
        return "recognition"
    return "vqa"


def _task_metric_description(task_type: str) -> str:
    metric_family = _task_metric_family(task_type)
    return {
        "parsing": "Official comparison should use TEDS or the dataset-provided parsing evaluator.",
        "localization": "Official comparison should use IoU or the dataset-provided localization evaluator.",
        "extraction": "Official comparison should use F1 or the dataset-provided extraction evaluator.",
        "counting": "Official comparison should use L1 or the dataset-provided counting evaluator.",
        "recognition": "Use the official recognition/reading evaluator; ANLS is only a proxy here.",
        "vqa": "Exact/Contain/ANLS are appropriate only for VQA-style samples.",
    }[metric_family]


def export_evaluation(metric: MetricSummary | dict, output_path: str) -> None:
    path_obj = Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    content = metric.to_dict() if hasattr(metric, "to_dict") else metric
    path_obj.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_answer_list(answers: Iterable[str]) -> List[str]:
    expand_answer_list: List[str] = []
    for item in answers or []:
        # legacy_compatible parquet/JSONL put numpy/list write_as "['a' 'b']" case.
        if isinstance(item, str):
            text = item.strip()
            try:
                parsed_value = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed_value = None
            if isinstance(parsed_value, (list, tuple)):
                expand_answer_list.extend(str(value) for value in parsed_value)
                continue
            numpylist = re.fullmatch(r"\[\s*'([^']*)'(?:\s+'([^']*)')*\s*\]", text)
            if numpylist:
                expand_answer_list.extend(re.findall(r"'([^']*)'", text))
                continue
        expand_answer_list.append(str(item))
    return [answer for answer in (_normalize_text(item) for item in expand_answer_list) if answer]


def _evaluate_ocrbench_v2_unified_results(result_list: Iterable[DocVQAResult]) -> MetricSummary:
    result_sequence = list(result_list)
    if not result_sequence:
        return MetricSummary(sample_count=0, exact_match=0.0, contains_answer=0.0, anls=0.0)

    exacthit_count = 0
    containhit_count = 0
    anls_total_score = 0.0

    for result in result_sequence:
        # evaluated_conclusion,avoid_explanation_or JSON include_shell_in_answer.
        predicted_answer = str(result.pipeline_output.get("final_conclusion") or result.pipeline_output.get("final_answer") or "")
        ground_truth_answers = _normalize_answer_list(result.ground_truth_answers)

        if _is_exact_match(predicted_answer, ground_truth_answers):
            exacthit_count += 1
        if _is_contains_match(predicted_answer, ground_truth_answers):
            containhit_count += 1
        anls_total_score += _compute_single_sample_anls(predicted_answer, ground_truth_answers)

    sample_count = len(result_sequence)
    return MetricSummary(
        sample_count=sample_count,
        exact_match=round(exacthit_count / sample_count, 4),
        contains_answer=round(containhit_count / sample_count, 4),
        anls=round(anls_total_score / sample_count, 4),
    )


def _is_exact_match(predicted_answer: str, ground_truth_answers: List[str]) -> bool:
    normalized_prediction = _normalize_text(predicted_answer)
    return bool(normalized_prediction) and any(normalized_prediction == ground_truth for ground_truth in ground_truth_answers)


def _is_contains_match(predicted_answer: str, ground_truth_answers: List[str]) -> bool:
    normalized_prediction = _normalize_text(predicted_answer)
    if not normalized_prediction:
        return False
    return any(ground_truth in normalized_prediction or normalized_prediction in ground_truth for ground_truth in ground_truth_answers)


def _compute_single_sample_anls(predicted_answer: str, ground_truth_answers: List[str]) -> float:
    normalized_prediction = _normalize_text(predicted_answer)
    if not normalized_prediction or not ground_truth_answers:
        return 0.0

    best_score = 0.0
    for ground_truth in ground_truth_answers:
        edit_distance = _edit_distance(normalized_prediction, ground_truth)
        max_length = max(len(normalized_prediction), len(ground_truth))
        if max_length == 0:
            similarity = 1.0
        else:
            similarity = 1.0 - edit_distance / max_length
        score = similarity if similarity >= 0.5 else 0.0
        best_score = max(best_score, score)
    return best_score


def _normalize_text(text: str) -> str:
    text = re.sub(r"^```(?:text|json)?\s*|\s*```$", "", str(text).strip(), flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^(?:final answer|refined answer|answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text.strip().lower())


def _edit_distance(text1: str, text2: str) -> int:
    if text1 == text2:
        return 0
    if not text1:
        return len(text2)
    if not text2:
        return len(text1)

    previous_row = list(range(len(text2) + 1))
    for index1, character1 in enumerate(text1, start=1):
        current_row = [index1]
        for index2, character2 in enumerate(text2, start=1):
            insert_cost = current_row[index2 - 1] + 1
            delete_cost = previous_row[index2] + 1
            replace_cost = previous_row[index2 - 1] + (0 if character1 == character2 else 1)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]
