# OCR-Agent: Agentic OCR with Capability and Memory Reflection

OCR-Agent is a lightweight, image-aware OCR reasoning pipeline. It separates the
initial answer from the final conclusion, reflects on visible evidence, filters
actions through explicit capability constraints, and optionally verifies or
blind-reviews a proposed correction.

## Highlights

- Mock, OpenAI-compatible, and Gemini backends behind one interface.
- OCR, visual question answering, translation, parsing, localization, and
  structured-output prompts.
- Variants for controlled ablations: `naive`, `cot`, `self-refine`, `capability`,
  `memory`, `ocr-agent`, and `visual-tool`.
- Optional high-resolution image tiles for small text and layout details.
- JSONL exports with per-round traces, evidence, actions, memory, and decisions.
- DocVQA and OCRBench-v2 readers for JSON, parquet, and parquet directories.
- Exact match, contains-answer, ANLS, initial-vs-final, and reflection-audit
  summaries. OCRBench-v2 task-specific official metrics remain the source of
  truth for paper comparisons.

## Project layout

```text
ocr_agent/
  capability_constraints.py
  data_models.py
  model_service.py
  prompts.py
  visual_tools.py
  core/pipeline.py
  datasets/docvqa.py
  datasets/ocrbench_v2.py
  runners/cli_docvqa.py
  runners/cli_ocrbench_v2.py
  runners/cli_rolmocr_health_check.py
  runners/run_docvqa_batch.py
  runners/run_ocrbench_v2_batch.py
  tools/evaluation.py
  tools/result_export.py
```

## Installation

The package uses Python 3.10 or newer.

```bash
git clone https://github.com/AIGeeksGroup/OCR-Agent.git
cd OCR-Agent
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

No API key is required for the mock backend. Keep real keys in environment
variables and never commit them to the repository.

## DocVQA

The reader accepts a JSON file, a parquet file, or a directory of parquet files.
Each sample should provide `question`, an image path or embedded image bytes, and
`answer` or `answers`.

```bash
python -m ocr_agent.runners.cli_docvqa \
  --backend mock \
  --datasets /path/to/docvqa.json \
  --image_dir /path/to/images \
  --variant ocr-agent \
  --max_samples 20 \
  --max_iterations 3 \
  --output outputs/docvqa.jsonl \
  --evaluation_output outputs/docvqa_metrics.json
```

For an OpenAI-compatible gateway:

```bash
export OPENAI_API_KEY="replace-me"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_MODEL="reducto/RolmOCR"

python -m ocr_agent.runners.cli_docvqa \
  --backend openai \
  --protocol chat_completions \
  --datasets /path/to/docvqa.json \
  --image_dir /path/to/images \
  --output outputs/docvqa.jsonl
```

The `responses` protocol is also supported with `--protocol responses` when the
gateway implements the OpenAI Responses API.

For Gemini, set the API key and model variables and select `--backend gemini`:

```bash
export GEMINI_API_KEY="replace-me"
export GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta"
export GEMINI_MODEL="gemini-2.5-flash"

python -m ocr_agent.runners.cli_docvqa \
  --backend gemini \
  --datasets /path/to/docvqa.json \
  --image_dir /path/to/images \
  --output outputs/docvqa-gemini.jsonl
```

## OCRBench-v2

OCRBench-v2 is read from a parquet shard or a directory containing parquet
shards. Use sequential, seeded random, or type-stratified sampling. A fixed JSON
sample-id file can be exported and reused for fair comparisons.

```bash
python -m ocr_agent.runners.cli_ocrbench_v2 \
  --backend mock \
  --datasets /path/to/OCRBench-v2 \
  --sampling_mode stratified_by_type \
  --random_seed 42 \
  --max_samples 100 \
  --output outputs/ocrbench_v2.jsonl \
  --evaluation_output outputs/ocrbench_v2_metrics.json
```

For visual-tool experiments, select `--variant visual-tool` and optionally set
`--visual_tool_dir`. The generated tiles are passed together with the original
image so the model retains full-page context.

## Health check

Use the RolmOCR health check to query a local OpenAI-compatible `/models`
endpoint and run one image request:

```bash
python -m ocr_agent.runners.cli_rolmocr_health_check \
  --base_url http://127.0.0.1:8000/v1 \
  --api_key local-123 \
  --model reducto/RolmOCR \
  --image /path/to/image.png
```

## Python API

```python
from ocr_agent.core.pipeline import OCRAgentPipeline
from ocr_agent.data_models import BackendConfig, PipelineConfig, SingleRequest
from ocr_agent.model_service import build_backend

backend = build_backend(BackendConfig(backend_type="mock"))
pipeline = OCRAgentPipeline(backend, PipelineConfig(variant_name="ocr-agent"))
result = pipeline.run(
    SingleRequest(question="Read the text.", image_paths=["image.png"])
)
print(result.to_dict())
```

## Output and evaluation

`tools/result_export.py` writes one JSON object per sample to JSONL and a
readable JSON companion file. Each record includes the initial and final answer,
the parsed conclusion, whether it changed, and the complete reflection trace.

`tools/evaluation.py` provides DocVQA exact match, contains-answer, and ANLS
metrics. OCRBench-v2 output additionally includes metrics by task type and
dataset, initial-vs-final counts, and an audit of reflection, revision,
verification, and blind-review decisions.

## Paper

The implementation is associated with the OCR-Agent paper available at
[arXiv:2602.21042](https://arxiv.org/abs/2602.21042).

## License

Add the license required by your intended release and by any upstream dataset or
model weights before redistribution.
