from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Dict, List

from ..data_models import DocVQASample


@dataclass
class OCRBenchV2SampleMetadata:
    sample_id: str
    task_type: str
    dataset_name: str
    shard_path: str
    shard_name: str
    shard_row: int


class OCRBenchV2Dataset:
    """read Hugging Face OCRBench-v2 of test parquet shard."""

    def __init__(self, dataset_path: str, temp_image_dir: str = "") -> None:
        self.dataset_path = Path(dataset_path)
        self.temp_image_dir = (
            Path(temp_image_dir)
            if temp_image_dir
            else self.dataset_path.parent.parent.parent / "tmp" / "ocrbench_v2_images"
        )
        self.temp_image_dir.mkdir(parents=True, exist_ok=True)

    def read_samples(
        self,
        max_samples: int | None = None,
        sampling_mode: str = "sequential",
        random_seed: int = 42,
        sample_ids_path: str = "",
        export_sample_ids_path: str = "",
    ) -> List[DocVQASample]:
        import pandas as pd

        file_list = self._find_parquet_files()
        sample_metadata_list = self._read_sample_metadata(file_list)
        selected_sample_id_set = self._select_sample_ids(
            sample_metadata_list=sample_metadata_list,
            max_samples=max_samples,
            sampling_mode=sampling_mode,
            random_seed=random_seed,
            sample_ids_path=sample_ids_path,
            export_sample_ids_path=export_sample_ids_path,
        )

        result: List[DocVQASample] = []
        selected_sample_id_set = set(selected_sample_id_set)
        sample_metadata_map = {sample_metadata.sample_id: sample_metadata for sample_metadata in sample_metadata_list}
        ordered_sample_ids = [sample_metadata.sample_id for sample_metadata in sample_metadata_list if sample_metadata.sample_id in selected_sample_id_set]
        for parquetfile in file_list:
            dataframe = pd.read_parquet(parquetfile)
            for line_number, raw_sample in enumerate(dataframe.to_dict(orient="records")):
                sample_id = str(raw_sample.get("id") if raw_sample.get("id") is not None else "")
                if sample_id not in selected_sample_id_set:
                    continue
                sample_metadata = sample_metadata_map.get(sample_id)
                sample = self._convert_sample(
                    raw_sample=raw_sample,
                    index=len(result),
                    shard_name=parquetfile.name,
                    shard_row=line_number,
                    sample_metadata=sample_metadata,
                )
                if sample is None:
                    continue
                result.append(sample)
        result_sort_key = {sample_id: index for index, sample_id in enumerate(ordered_sample_ids)}
        result.sort(key=lambda sample: result_sort_key.get(sample.sample_id, 10**9))
        return result

    def _find_parquet_files(self) -> List[Path]:
        if self.dataset_path.is_file() and self.dataset_path.suffix.lower() == ".parquet":
            return [self.dataset_path]
        direct_shard = sorted(self.dataset_path.glob("*.parquet"))
        datashard = sorted((self.dataset_path / "data").glob("*.parquet"))
        file_list = direct_shard or datashard
        if not file_list:
            raise ValueError(f"directory {self.dataset_path} not_found_under OCRBench-v2 parquet shard.")
        return file_list

    def _read_sample_metadata(self, file_list: List[Path]) -> List[OCRBenchV2SampleMetadata]:
        import pandas as pd

        result: List[OCRBenchV2SampleMetadata] = []
        for parquetfile in file_list:
            dataframe = pd.read_parquet(parquetfile, columns=["id", "type", "dataset_name", "question"])
            for line_number, raw_sample in enumerate(dataframe.to_dict(orient="records")):
                question = raw_sample.get("question")
                if not isinstance(question, str) or not question.strip():
                    continue
                sample_id = str(raw_sample.get("id") if raw_sample.get("id") is not None else len(result))
                result.append(
                    OCRBenchV2SampleMetadata(
                        sample_id=sample_id,
                        task_type=str(raw_sample.get("type") or "unknown"),
                        dataset_name=str(raw_sample.get("dataset_name") or "unknown"),
                        shard_path=str(parquetfile),
                        shard_name=parquetfile.name,
                        shard_row=line_number,
                    )
                )
        return result

    def _select_sample_ids(
        self,
        sample_metadata_list: List[OCRBenchV2SampleMetadata],
        max_samples: int | None,
        sampling_mode: str,
        random_seed: int,
        sample_ids_path: str,
        export_sample_ids_path: str,
    ) -> List[str]:
        if sample_ids_path:
            sample_ids = self._read_sample_ids(sample_ids_path)
            # fixed_id_file_for_fair_reproduction,still_follow_cli
            # A fixed id file enables fair reproduction; max_samples can limit a smoke test.
            if max_samples is not None:
                sample_ids = sample_ids[:max_samples]
        elif sampling_mode == "random":
            sample_ids = self._random_sample_ids(sample_metadata_list, max_samples, random_seed)
        elif sampling_mode == "stratified_by_type":
            sample_ids = self._stratified_sample_ids(sample_metadata_list, max_samples, random_seed)
        else:
            sample_ids = [sample_metadata.sample_id for sample_metadata in sample_metadata_list[:max_samples]]
        if export_sample_ids_path:
            self._export_sample_ids(sample_ids, export_sample_ids_path)
        return sample_ids

    @staticmethod
    def _random_sample_ids(
        sample_metadata_list: List[OCRBenchV2SampleMetadata],
        max_samples: int | None,
        random_seed: int,
    ) -> List[str]:
        sample_ids = [sample_metadata.sample_id for sample_metadata in sample_metadata_list]
        if max_samples is None or max_samples >= len(sample_ids):
            return sample_ids
        random_generator = random.Random(random_seed)
        return random_generator.sample(sample_ids, max_samples)

    @staticmethod
    def _stratified_sample_ids(
        sample_metadata_list: List[OCRBenchV2SampleMetadata],
        max_samples: int | None,
        random_seed: int,
    ) -> List[str]:
        if max_samples is None or max_samples >= len(sample_metadata_list):
            return [sample_metadata.sample_id for sample_metadata in sample_metadata_list]

        random_generator = random.Random(random_seed)
        group_by_type: Dict[str, List[OCRBenchV2SampleMetadata]] = {}
        for sample_metadata in sample_metadata_list:
            group_by_type.setdefault(sample_metadata.task_type, []).append(sample_metadata)

        types = sorted(group_by_type)
        base_quota = max(1, max_samples // len(types))
        selected_sample_ids: List[str] = []
        remaining_candidates: List[OCRBenchV2SampleMetadata] = []

        for task_type in types:
            group_samples = list(group_by_type[task_type])
            random_generator.shuffle(group_samples)
            group_selection_count = min(len(group_samples), base_quota)
            selected_sample_ids.extend(sample.sample_id for sample in group_samples[:group_selection_count])
            remaining_candidates.extend(group_samples[group_selection_count:])

        if len(selected_sample_ids) < max_samples and remaining_candidates:
            random_generator.shuffle(remaining_candidates)
            needed_count = max_samples - len(selected_sample_ids)
            selected_sample_ids.extend(sample.sample_id for sample in remaining_candidates[:needed_count])

        if len(selected_sample_ids) > max_samples:
            selected_sample_ids = selected_sample_ids[:max_samples]
        return selected_sample_ids

    @staticmethod
    def _read_sample_ids(sample_ids_path: str) -> List[str]:
        path_obj = Path(sample_ids_path)
        data = json.loads(path_obj.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("sample_ids", [])
        if not isinstance(data, list):
            raise ValueError(f"sample_id_file_format_error: {path_obj}")
        return [str(sample_id) for sample_id in data]

    @staticmethod
    def _export_sample_ids(sample_ids: List[str], sample_ids_path: str) -> None:
        path_obj = Path(sample_ids_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(
            json.dumps({"sample_ids": sample_ids}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _convert_sample(
        self,
        raw_sample: Dict[str, Any],
        index: int,
        shard_name: str,
        shard_row: int,
        sample_metadata: OCRBenchV2SampleMetadata | None = None,
    ) -> DocVQASample | None:
        question = raw_sample.get("question")
        if not isinstance(question, str) or not question.strip():
            return None

        sample_id = str(raw_sample.get("id") if raw_sample.get("id") is not None else index)
        data_type = sample_metadata.task_type if sample_metadata else str(raw_sample.get("type") or "unknown")
        dataset_name = sample_metadata.dataset_name if sample_metadata else str(raw_sample.get("dataset_name") or "unknown")
        image_path = self._export_image(
            raw_sample.get("image"),
            sample_id=sample_id,
            shard_name=shard_name,
            shard_row=shard_row,
        )
        if image_path is None:
            return None

        answer_field = raw_sample.get("answers", [])
        # parquet usually_read_after numpy.ndarray;must_expand,otherwise_multiple_candidates
        # Parquet answers may be numpy arrays; expand them before evaluation.
        if hasattr(answer_field, "tolist"):
            answer_field = answer_field.tolist()
        if isinstance(answer_field, (list, tuple)):
            ground_truth_answers = [str(answer).strip() for answer in answer_field if str(answer).strip()]
        else:
            ground_truth_answers = [str(answer_field).strip()] if str(answer_field).strip() else []

        return DocVQASample(
            sample_id=sample_id,
            question=str(question),
            image_path=str(image_path),
            ground_truth_answers=ground_truth_answers,
            document_context=f"dataset_name={dataset_name}; task_type={data_type}",
            task_type=data_type,
            data_split="test",
            dataset_name=dataset_name,
        )

    def _export_image(
        self,
        image_field: Any,
        sample_id: str,
        shard_name: str,
        shard_row: int,
    ) -> Path | None:
        if not isinstance(image_field, dict):
            return None
        image_bytes = image_field.get("bytes")
        if image_bytes is None:
            return None
        image_bytes = bytes(image_bytes)
        raw_path = str(image_field.get("path") or "")
        suffix = Path(raw_path).suffix.lower() or ".png"
        summary = hashlib.md5(image_bytes).hexdigest()[:12]
        safe_shard_name = Path(shard_name).stem.replace(" ", "_")
        output_path = self.temp_image_dir / f"{safe_shard_name}_{shard_row}_{sample_id}_{summary}{suffix}"
        if not output_path.exists():
            output_path.write_bytes(image_bytes)
        return output_path
