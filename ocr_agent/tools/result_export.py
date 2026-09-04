from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..data_models import DocVQAResult


def export_jsonl(result_list: Iterable[DocVQAResult], output_path: str) -> None:
    path_obj = Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    structured_records = []
    with path_obj.open("w", encoding="utf-8") as file:
        for result in result_list:
            final_answer = result.pipeline_output.get("final_answer", "")
            initial_answer = result.pipeline_output.get("initial_answer", "")
            final_conclusion = result.pipeline_output.get("final_conclusion", "")
            initial_conclusion = result.pipeline_output.get("initial_conclusion", "")
            traces = result.pipeline_output.get("traces", [])
            record = {
                "sample_id": result.sample_id,
                "task_type": result.task_type,
                "data_split": result.data_split,
                "dataset_name": result.dataset_name,
                "question": result.question,
                "image_path": result.image_path,
                "ground_truth_answers": result.ground_truth_answers,
                "initial_answer": initial_answer,
                "final_answer": final_answer,
                "initial_conclusion": initial_conclusion,
                "final_conclusion": final_conclusion,
                "text_changed": bool(result.pipeline_output.get("text_changed", initial_answer != final_answer)),
                "conclusion_changed": bool(result.pipeline_output.get("conclusion_changed", initial_conclusion != final_conclusion)),
                "answer_changed": bool(result.pipeline_output.get("answer_changed", initial_answer != final_answer)),
                "iteration_count": len(traces),
                "pipeline_output": result.pipeline_output,
            }
            structured_records.append(record)
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
    pretty_output_path = _determineprettyoutput_path(path_obj)
    pretty_output_path.write_text(
        json.dumps(structured_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _determineprettyoutput_path(jsonlpath: Path) -> Path:
    parent_dir_name = jsonlpath.parent.name
    if parent_dir_name in {"raw_run_results", "semi_auto_labels"}:
        prettydirectory = jsonlpath.parent.parent / "prettyreadable_report"
        prettydirectory.mkdir(parents=True, exist_ok=True)
        return prettydirectory / f"{jsonlpath.stem}_pretty.json"
    return jsonlpath.with_name(f"{jsonlpath.stem}_pretty.json")
