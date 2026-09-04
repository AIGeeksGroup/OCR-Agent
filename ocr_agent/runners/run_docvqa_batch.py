from __future__ import annotations

from typing import List

from ..data_models import DocVQAResult, BackendConfig, SingleRequest, PipelineConfig
from ..model_service import build_backend
from ..datasets.docvqa import DocVQADataset
from ..core.pipeline import OCRAgentPipeline


def run_docvqa_batch(
    annotation_path: str,
    image_root: str,
    backend_config: BackendConfig,
    pipeline_config: PipelineConfig,
    max_samples: int | None = None,
    temp_image_dir: str = r"D:\OCR-Agent\tmp\docvqa_images",
    max_iterations: int = 3,
) -> List[DocVQAResult]:
    datasets = DocVQADataset(
        annotation_path=annotation_path,
        image_root=image_root,
        temp_image_dir=temp_image_dir,
    )
    sample_list = datasets.read_samples(max_samples=max_samples)
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
