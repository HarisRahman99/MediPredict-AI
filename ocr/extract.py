"""OCR text extraction using EasyOCR."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

from .preprocess import OCRDependencyError, OCRProcessingError


@lru_cache(maxsize=1)
def _reader():
    try:
        import easyocr
    except ImportError as exc:
        raise OCRDependencyError(
            "EasyOCR is not installed."
        ) from exc

    # gpu=False for maximum compatibility
    return easyocr.Reader(["en"], gpu=False)


def extract_text(images: List[object]) -> str:
    if not images:
        raise OCRProcessingError("No report pages were available.")

    reader = _reader()
    extracted_lines = []

    try:
        for i, image in enumerate(images, start=1):
            print(f"\nProcessing page {i}...")

            results = reader.readtext(
                image,
                detail=0,
                paragraph=False,
            )

            print("OCR Output:")
            print(results)

            extracted_lines.extend(results)

    except Exception as exc:
        raise OCRProcessingError(
            f"EasyOCR failed: {exc}"
        ) from exc

    return "\n".join(
        str(line).strip()
        for line in extracted_lines
        if str(line).strip()
    )