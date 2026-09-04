from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..data_models import BackendConfig, PipelineConfig
from ..tools.evaluation import export_evaluation, evaluate_ocrbench_v2_results
from ..tools.result_export import export_jsonl
from .run_ocrbench_v2_batch import run_ocrbench_v2_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="OCRBench-v2 batch_experiment_entry")
    parser.add_argument("--backend", choices=["mock", "openai", "gemini"], default="openai")
    parser.add_argument("--protocol", choices=["chat_completions", "responses"], default=os.environ.get("OPENAI_WIRE_API", "chat_completions"))
    parser.add_argument("--variant", choices=["naive", "cot", "self-refine", "capability", "memory", "ocr-agent", "visual-tool"], default="ocr-agent")
    parser.add_argument("--datasets", default=r"src\ocr_agent\datasets\OCRBench-v2")
    parser.add_argument("--temp_image_dir", default="")
    parser.add_argument("--visual_tool_dir", default="", help="visual-tool tile_cache_dir;default_system_temp_dir")
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--sampling_mode", choices=["sequential", "random", "stratified_by_type"], default="sequential")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--sample_ids_file", default="", help="optional;from json read_fixed_ids")
    parser.add_argument("--export_sample_ids_file", default="", help="optional;write_selected_sample_ids_to json")
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--enable_verification", action="store_true", help="enable_extra_candidate_validation")
    parser.add_argument("--enable_blind_review", action="store_true", help="enable_extra_independent_blind_review")
    parser.add_argument("--enable_answer_filter", action="store_true", help="enable_heuristic_answer_filter;disabled_in_paper_faithful_mode")
    parser.add_argument("--request_interval_seconds", type=float, default=-1)
    parser.add_argument("--max_retries", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model_random_seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation_output", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--base_url", default="")
    args = parser.parse_args()

    args.model = args.model or os.environ.get(
        "GEMINI_MODEL" if args.backend == "gemini" else "OPENAI_MODEL",
        "gemini-2.5-flash" if args.backend == "gemini" else "/workspace/OCR-Agent/model_weights/RolmOCR",
    )
    args.base_url = args.base_url or os.environ.get(
        "GEMINI_BASE_URL" if args.backend == "gemini" else "OPENAI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta"
        if args.backend == "gemini"
        else "http://127.0.0.1:8000/v1",
    )

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
        enable_visual_tools=args.variant == "visual-tool",
        visual_tool_dir=args.visual_tool_dir,
    )

    result_list = run_ocrbench_v2_batch(
        dataset_path=args.datasets,
        backend_config=backend_config,
        pipeline_config=pipeline_config,
        max_samples=args.max_samples,
        temp_image_dir=args.temp_image_dir,
        max_iterations=args.max_iterations,
        sampling_mode=args.sampling_mode,
        random_seed=args.random_seed,
        sample_ids_path=args.sample_ids_file,
        export_sample_ids_path=args.export_sample_ids_file,
    )
    export_jsonl(result_list, args.output)
    evaluation_results = evaluate_ocrbench_v2_results(result_list)
    if args.evaluation_output:
        export_evaluation(evaluation_results, args.evaluation_output)

    print(
        json.dumps(
            {
                "datasets": str(Path(args.datasets).resolve()),
                "backend": args.backend,
                "model": args.model,
                "base_url": args.base_url,
                "variant": args.variant,
                "sampling_mode": args.sampling_mode,
                "random_seed": args.random_seed,
                "temperature": args.temperature,
                "model_random_seed": args.model_random_seed,
                "sample_ids_file": str(Path(args.sample_ids_file).resolve()) if args.sample_ids_file else "",
                "export_sample_ids_file": str(Path(args.export_sample_ids_file).resolve()) if args.export_sample_ids_file else "",
                "max_iterations": args.max_iterations,
                "result_count": len(result_list),
                "evaluation_results": evaluation_results.to_dict() if hasattr(evaluation_results, "to_dict") else evaluation_results,
                "output_file": str(Path(args.output).resolve()),
                "evaluation_output_file": str(Path(args.evaluation_output).resolve()) if args.evaluation_output else "",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
