from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib import request

try:
    from ..data_models import BackendConfig
    from ..model_service import build_backend
except ImportError:
    import sys

    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from ocr_agent.data_models import BackendConfig
    from ocr_agent.model_service import build_backend


def list_models(base_url: str, api_key: str) -> dict:
    request_object = request.Request(
        url=f"{base_url.rstrip('/')}/models",
        method="GET",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with request.urlopen(request_object, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_available_model_ids(model_list_response: dict) -> list[str]:
    data_list = model_list_response.get("data", [])
    if not isinstance(data_list, list):
        return []
    model_ids: list[str] = []
    for item in data_list:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            model_ids.append(model_id)
    return model_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="check_local RolmOCR vLLM service_available")
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "local-123"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "reducto/RolmOCR"))
    parser.add_argument("--image", default=str((Path(__file__).resolve().parents[3] / "examples" / "invoice.png").resolve()))
    parser.add_argument("--question", default="Extract all visible text from the image. Return plain text only.")
    args = parser.parse_args()

    model_list = list_models(args.base_url, args.api_key)
    available_model_ids = parse_available_model_ids(model_list)
    selected_model = args.model
    if available_model_ids and selected_model not in available_model_ids:
        selected_model = available_model_ids[0]

    backend = build_backend(
        BackendConfig(
            backend_type="openai",
            model_name=selected_model,
            api_key=args.api_key,
            base_url=args.base_url,
            api_protocol="chat_completions",
            max_retries=1,
        )
    )
    output_text = backend.generate("You are a careful OCR assistant.", args.question, [args.image])
    print(
        json.dumps(
            {
                "base_url": args.base_url,
                "requestmodel": args.model,
                "actual_model": selected_model,
                "image": str(Path(args.image).resolve()),
                "model_list_return": model_list,
                "generated_result": output_text,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
