from __future__ import annotations

import ast
import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List

from ..data_models import DocVQASample


class DocVQADataset:
    """DocVQA small_experiment_reader.

    supports_two_inputs:
    1. text json annotation_file
    2. parquet shard,among image embedded_field bytes and path
    """

    def __init__(self, annotation_path: str, image_root: str = "", temp_image_dir: str = "") -> None:
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root) if image_root else self.annotation_path.parent
        self.temp_image_dir = Path(temp_image_dir) if temp_image_dir else self._default_temp_image_dir()
        self.temp_image_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_temp_image_dir() -> Path:
        environment_path = os.environ.get("DOCVQA_TMP_IMAGE_DIR", "").strip()
        if environment_path:
            return Path(environment_path)
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]
        return project_root / "tmp" / "docvqa_images"

    def read_samples(self, max_samples: int | None = None) -> List[DocVQASample]:
        raw_data = self._read_raw_data()
        sample_list = self._extract_samples(raw_data)
        result: List[DocVQASample] = []
        for index, sample in enumerate(sample_list):
            structured_sample = self._convert_sample(sample, index)
            if structured_sample is None:
                continue
            result.append(structured_sample)
            if max_samples is not None and len(result) >= max_samples:
                break
        return result

    def _read_raw_data(self) -> Any:
        if self.annotation_path.is_dir():
            return self._readparquetdirectory()
        if self.annotation_path.suffix.lower() == ".parquet":
            import pandas as pd

            dataframe = pd.read_parquet(self.annotation_path)
            return dataframe.to_dict(orient="records")
        return json.loads(self.annotation_path.read_text(encoding="utf-8"))

    def _readparquetdirectory(self) -> List[Dict[str, Any]]:
        import pandas as pd

        parquetfile_list = sorted(self.annotation_path.glob("*.parquet"))
        if not parquetfile_list:
            raise ValueError(f"directory {self.annotation_path} not_found_under parquet file.")
        raw_records: List[Dict[str, Any]] = []
        for parquetfile in parquetfile_list:
            dataframe = pd.read_parquet(parquetfile)
            raw_records.extend(dataframe.to_dict(orient="records"))
        return raw_records

    @staticmethod
    def _extract_samples(raw_data: Any) -> List[Dict[str, Any]]:
        if isinstance(raw_data, list):
            return [item for item in raw_data if isinstance(item, dict)]
        if isinstance(raw_data, dict):
            for key in ["data", "dataset", "questions", "samples"]:
                value = raw_data.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        raise ValueError("cannot_identify_annotation DocVQA sample_list.")

    def _convert_sample(self, raw_sample: Dict[str, Any], index: int) -> DocVQASample | None:
        question = raw_sample.get("question") or raw_sample.get("query")
        if not question:
            return None
        sample_id = str(raw_sample.get("questionId") or raw_sample.get("question_id") or raw_sample.get("id") or index)
        ground_truth_answers = self._extract_answer_list(raw_sample)
        image_path = self._parse_image_path(raw_sample, sample_id)
        document_context = str(raw_sample.get("document_context") or raw_sample.get("context") or "")
        data_split = str(raw_sample.get("data_split") or raw_sample.get("split") or "")
        return DocVQASample(
            sample_id=sample_id,
            question=str(question),
            image_path=str(image_path),
            ground_truth_answers=ground_truth_answers,
            document_context=document_context,
            task_type="docvqa",
            data_split=data_split,
            dataset_name="DocVQA",
        )

    @staticmethod
    def _extract_answer_list(raw_sample: Dict[str, Any]) -> List[str]:
        for field_name in ["answers", "answer"]:
            if field_name not in raw_sample:
                continue
            answers = DocVQADataset._normalize_answer_field(raw_sample.get(field_name))
            if answers:
                return answers
        return []

    @staticmethod
    def _normalize_answer_field(answer_field: Any) -> List[str]:
        if answer_field is None:
            return []
        if isinstance(answer_field, list):
            return [str(item).strip() for item in answer_field if str(item).strip()]
        if isinstance(answer_field, Iterable) and not isinstance(answer_field, (str, bytes, bytearray, dict)):
            return [str(item).strip() for item in list(answer_field) if str(item).strip()]
        if isinstance(answer_field, str):
            text = answer_field.strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed_result = ast.literal_eval(text)
                except (SyntaxError, ValueError):
                    parsed_result = None
                if isinstance(parsed_result, list):
                    return [str(item).strip() for item in parsed_result if str(item).strip()]
                quoted_fragments = [fragment.strip() for fragment in ast.literal_eval(repr(text)).strip("[]").split("'") if fragment.strip() and fragment.strip() != ","]
                if quoted_fragments:
                    return [fragment for fragment in quoted_fragments if fragment != ","]
            return [text]
        return [str(answer_field).strip()] if str(answer_field).strip() else []

    def _parse_image_path(self, raw_sample: Dict[str, Any], sample_id: str) -> Path:
        image_field = raw_sample.get("image") or raw_sample.get("image_path") or raw_sample.get("image_id")
        if image_field is None:
            raise ValueError("sample_missing_image.")
        if isinstance(image_field, dict) and image_field.get("bytes") is not None:
            return self._export_embedded_image(
                image_bytes=image_field["bytes"],
                raw_path=str(image_field.get("path") or ""),
                sample_id=sample_id,
            )
        image_path = Path(str(image_field))
        if not image_path.is_absolute():
            image_path = (self.image_root / image_path).resolve()
        return image_path

    def _export_embedded_image(self, image_bytes: bytes, raw_path: str, sample_id: str) -> Path:
        suffix = Path(raw_path).suffix.lower() or ".png"
        summary = hashlib.md5(image_bytes).hexdigest()[:12]
        output_path = self.temp_image_dir / f"{sample_id}_{summary}{suffix}"
        if not output_path.exists():
            output_path.write_bytes(image_bytes)
        return output_path
