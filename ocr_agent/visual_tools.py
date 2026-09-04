from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Iterable, List

from PIL import Image


class VisualToolExecutor:
    """generate_overlapping_tiles,visual_detail_for_revision."""

    def __init__(self, output_dir: str = "") -> None:
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "ocr-agent-visual-tools"

    def generate_high_res_tiles(self, image_paths: Iterable[str]) -> List[str]:
        output_paths: List[str] = []
        for image_path in image_paths:
            output_paths.extend(self._generate_image_tiles(Path(image_path)))
        return output_paths

    def _generate_image_tiles(self, image_path: Path) -> List[str]:
        try:
            with Image.open(image_path) as original_image:
                image = original_image.convert("RGB")
                width, height = image.size
                if width < 320 or height < 320:
                    return []

                text = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:12]
                output_dir = self.output_dir / text
                output_dir.mkdir(parents=True, exist_ok=True)
                horizontal_overlap = max(16, int(width * 0.08))
                vertical_overlap = max(16, int(height * 0.08))
                center_line_x = width // 2
                center_line_y = height // 2
                regions = [
                    (0, 0, min(width, center_line_x + horizontal_overlap), min(height, center_line_y + vertical_overlap)),
                    (max(0, center_line_x - horizontal_overlap), 0, width, min(height, center_line_y + vertical_overlap)),
                    (0, max(0, center_line_y - vertical_overlap), min(width, center_line_x + horizontal_overlap), height),
                    (max(0, center_line_x - horizontal_overlap), max(0, center_line_y - vertical_overlap), width, height),
                ]
                output_paths: List[str] = []
                for id, region in enumerate(regions, start=1):
                    output_path = output_dir / f"tile_{id}.jpg"
                    if not output_path.exists():
                        image.crop(region).save(output_path, format="JPEG", quality=95, optimize=True)
                    output_paths.append(str(output_path))
                return output_paths
        except Exception:
            # image_tool_optional;bad_sample_does_not_stop_batch.
            return []
