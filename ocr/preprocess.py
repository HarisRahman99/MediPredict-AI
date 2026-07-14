"""Image and PDF preprocessing before OCR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}


class OCRProcessingError(Exception):
    """Raised when an uploaded report cannot be prepared for OCR."""


class OCRDependencyError(OCRProcessingError):
    """Raised when a required OCR dependency is not installed."""


@dataclass(frozen=True)
class UploadedReport:
    filename: str
    content: bytes

    @property
    def extension(self) -> str:
        return Path(self.filename or "").suffix.lower().lstrip(".")


def is_allowed_file(filename: str) -> bool:
    return bool(filename and Path(filename).suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS)


def _require_cv2():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise OCRDependencyError(
            "OCR preprocessing requires opencv-python-headless and numpy."
        ) from exc
    return cv2, np


def _decode_image(content: bytes):
    cv2, np = _require_cv2()
    image_bytes = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise OCRProcessingError("The image could not be read. Please upload a clear JPG or PNG report.")
    return image


def _pdf_to_images(content: bytes):
    try:
        import fitz  # type: ignore
    except ImportError:
        fitz = None

    cv2, np = _require_cv2()

    if fitz is not None:
        try:
            document = fitz.open(stream=content, filetype="pdf")
            if document.page_count == 0:
                raise OCRProcessingError("The PDF does not contain any readable pages.")

            images = []
            for page in document:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                arr = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width, pixmap.n
                )
                images.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            return images
        except OCRProcessingError:
            raise
        except Exception as exc:
            raise OCRProcessingError("The PDF appears to be corrupted or unreadable.") from exc

    try:
        from pdf2image import convert_from_bytes  # type: ignore
    except ImportError as exc:
        raise OCRDependencyError(
            "PDF OCR requires PyMuPDF or pdf2image. Install one of them and try again."
        ) from exc

    try:
        pages = convert_from_bytes(content, dpi=220)
    except Exception as exc:
        raise OCRProcessingError("The PDF appears to be corrupted or unreadable.") from exc

    if not pages:
        raise OCRProcessingError("The PDF does not contain any readable pages.")

    images = []
    for page in pages:
        rgb = np.array(page.convert("RGB"))
        images.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return images


def _resize_for_ocr(gray):
    cv2, _ = _require_cv2()
    height, width = gray.shape[:2]
    if width == 0 or height == 0:
        raise OCRProcessingError("The uploaded image is empty.")

    target_width = 1800
    if width >= target_width:
        return gray

    scale = target_width / float(width)
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _deskew(gray):
    cv2, np = _require_cv2()
    coords = np.column_stack(np.where(gray < 245))
    if coords.size == 0:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5 or abs(angle) > 15:
        return gray

    height, width = gray.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image(image):
    """
    Prepare image for PaddleOCR.
    Returns a 3-channel BGR image.
    """

    cv2, _ = _require_cv2()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = _resize_for_ocr(gray)

    gray = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=18,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    gray = _deskew(gray)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    # PaddleOCR expects a 3-channel image
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def preprocess_upload(filename: str, content: bytes) -> List[object]:
    """Validate and preprocess an uploaded report into one or more OCR images."""

    if not content:
        raise OCRProcessingError("The uploaded file is empty.")
    if not is_allowed_file(filename):
        raise OCRProcessingError("Unsupported file type. Upload a PDF, JPG, JPEG, or PNG report.")

    report = UploadedReport(filename=filename, content=content)
    raw_images = _pdf_to_images(content) if report.extension == "pdf" else [_decode_image(content)]
    return [preprocess_image(image) for image in raw_images]

