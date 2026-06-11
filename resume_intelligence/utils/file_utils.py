"""File validation and temporary file handling for resume uploads."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@dataclass
class FileValidationResult:
    """Outcome of validating an uploaded resume file."""

    valid: bool
    extension: str
    error: Optional[str] = None


def validate_resume_file(filename: str, file_bytes: bytes) -> FileValidationResult:
    """
    Validate uploaded resume file type and size.

    Returns a FileValidationResult instead of raising, so callers can
    surface friendly errors in the UI.
    """
    if not filename:
        return FileValidationResult(valid=False, extension="", error="No filename provided.")

    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return FileValidationResult(
            valid=False,
            extension=extension,
            error=f"Unsupported file type '{extension}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
        )

    if not file_bytes:
        return FileValidationResult(valid=False, extension=extension, error="Uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        return FileValidationResult(
            valid=False,
            extension=extension,
            error=f"File too large ({size_mb:.1f} MB). Maximum is 10 MB.",
        )

    return FileValidationResult(valid=True, extension=extension)


def save_upload_to_temp(file_bytes: bytes, extension: str) -> str:
    """Write uploaded bytes to a temporary file and return its path."""
    suffix = extension if extension.startswith(".") else f".{extension}"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(file_bytes)
    except Exception:
        os.unlink(path)
        raise
    return path


def read_binary_upload(uploaded_file: BinaryIO) -> bytes:
    """Read all bytes from a file-like upload object."""
    uploaded_file.seek(0)
    return uploaded_file.read()
