from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..data_models import BackendConfig, PipelineConfig
from ..tools.evaluation import export_evaluation, evaluate_docvqa_results
from ..tools.result_export import export_jsonl
from .run_docvqa_batch import run_docvqa_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="DocVQA small_batch_entry")
    parser.add_argument("--backend", choices=["mock", "openai", "gemini"], default="openai")
    parser.add_argument("--protocol", choices=["chat_completions", "responses"], default=os.environ.get("OPENAI_WIRE_API", "chat_completions"))
    parser.add_argument("--variant", choices=["naive", "cot", "self-refine", "capability", "memory", "ocr-agent"], default="ocr-agent")
    parser.add_argument("--datasets", required=True, help="DocVQA annotation_path,support parquet/json file_or parquet directory")
    parser.add_argument("--image_dir", default="", help="DocVQA image_root;if_is parquet embedded_image,optional")
    parser.add_argument("--temp_image_dir", default="", help="from parquet exported_temp_image_dir;use_current_project tmp/docvqa_images")
    parser.add_argument("--max_samples", type=int, default=20, help="max_samples")
    parser.add_argument("--max_iterations", type=int, default=3, help="max_reflection_rounds")
    parser.add_argument("--enable_verification", action="store_true", help="enable_candidate_validation")
    parser.add_argument("--enable_blind_review", action="store_true", help="enable_independent_blind_review")
    parser.add_argument("--enable_answer_filter", action="store_true", help="enable_heuristic_answer_filter")
    parser.add_argument("--request_interval_seconds", type=float, default=-1, help="min_request_interval_seconds")
    parser.add_argument("--max_retries", type=int, default=-1, help="max_retries_on_model_failure")
    parser.add_argument("--temperature", type=float, default=0.0, help="model_decode_temperature")
    parser.add_argument("--model_random_seed", type=int, default=42, help="decoder_seed_for_compatible_backend")
    parser.add_argument("--output", required=True, help="output jsonl path")
    parser.add_argument("--evaluation_output", default="", help="optional;evaluation_summary json path")
    parser.add_argument("--model", default="")
    parser.add_argument("--base_url", default="")
    args = parser.parse_args()

    if not args.model:
        if args.backend == "gemini":
            args.model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        else:
            args.model = os.environ.get("OPENAI_MODEL", "reducto/RolmOCR")

    if not args.base_url:
        if args.backend == "gemini":
            args.base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        else:
            args.base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")

    backend_config = BackendConfig(
        backend_type=args.backend,
        model_name=args.model,
        api_key=(
            os.environ.get("GEMINI_API_KEY")
            if args.backend == "gemini"
            else os.environ.get("OPENAI_API_KEY", "local-123")
        ),
        base_url=args.base_url,
        api_protocol=args.protocol,
        max_retries=2 if args.max_retries < 0 else args.max_retries,
        request_interval_seconds=0.0 if args.request_interval_seconds < 0 else args.request_interval_seconds,
        temperature=args.temperature,
        random_seed=args.model_random_seed,
    )
    pipeline_config = PipelineConfig(
        variant_name=args.variant,
        enable_capability_reflection=args.variant in {"capability", "ocr-agent"},
        enable_memory_reflection=args.variant in {"memory", "ocr-agent"},
        enable_verification=args.enable_verification,
        enable_blind_review=args.enable_blind_review,
        enable_answer_filter=args.enable_answer_filter,
    )

    result_list = run_docvqa_batch(
        annotation_path=args.datasets,
        image_root=args.image_dir,
        backend_config=backend_config,
        pipeline_config=pipeline_config,
        max_samples=args.max_samples,
        temp_image_dir=args.temp_image_dir,
        max_iterations=args.max_iterations,
    )
    export_jsonl(result_list, args.output)
    evaluation_results = evaluate_docvqa_results(result_list)
    if args.evaluation_output:
        export_evaluation(evaluation_results, args.evaluation_output)

    print(
        json.dumps(
            {
                "datasets": str(Path(args.datasets).resolve()),
                "image_dir": str(Path(args.image_dir).resolve()) if args.image_dir else "",
                "temp_image_dir": str(Path(args.temp_image_dir).resolve()) if args.temp_image_dir else "",
                "backend": args.backend,
                "protocol": args.protocol,
                "model": args.model,
                "base_url": args.base_url,
                "variant": args.variant,
                "max_iterations": args.max_iterations,
                "temperature": args.temperature,
                "model_random_seed": args.model_random_seed,
                "result_count": len(result_list),
                "evaluation_results": evaluation_results.to_dict(),
                "output_file": str(Path(args.output).resolve()),
                "evaluation_output_file": str(Path(args.evaluation_output).resolve()) if args.evaluation_output else "",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
