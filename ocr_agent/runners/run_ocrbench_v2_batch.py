from __future__ import annotations

from typing import List

from ..data_models import DocVQAResult, BackendConfig, SingleRequest, PipelineConfig
from ..datasets.ocrbench_v2 import OCRBenchV2Dataset
from ..core.pipeline import OCRAgentPipeline
from ..model_service import build_backend


def run_ocrbench_v2_batch(
    dataset_path: str,
    backend_config: BackendConfig,
    pipeline_config: PipelineConfig,
    max_samples: int | None = None,
    temp_image_dir: str = "",
    max_iterations: int = 3,
    sampling_mode: str = "sequential",
    random_seed: int = 42,
    sample_ids_path: str = "",
    export_sample_ids_path: str = "",
) -> List[DocVQAResult]:
    datasets = OCRBenchV2Dataset(dataset_path=dataset_path, temp_image_dir=temp_image_dir)
    sample_list = datasets.read_samples(
        max_samples=max_samples,
        sampling_mode=sampling_mode,
        random_seed=random_seed,
        sample_ids_path=sample_ids_path,
        export_sample_ids_path=export_sample_ids_path,
    )
    backend = build_backend(backend_config)
    pipeline = OCRAgentPipeline(backend=backend, config=pipeline_config)

    result_list: List[DocVQAResult] = []
    for sample in sample_list:
        request = SingleRequest(
            question=sample.question,
            image_paths=[sample.image_path],
            document_context=sample.document_context,
            task_type=sample.task_type,
            dataset_name=sample.dataset_name,
            max_iterations=max_iterations,
        )
        pipeline_output = pipeline.run(request).to_dict()
        result_list.append(
            DocVQAResult(
                sample_id=sample.sample_id,
                task_type=sample.task_type,
                data_split=sample.data_split,
                question=sample.question,
                image_path=sample.image_path,
                ground_truth_answers=sample.ground_truth_answers,
                pipeline_output=pipeline_output,
                dataset_name=sample.dataset_name,
            )
        )
    return result_list
