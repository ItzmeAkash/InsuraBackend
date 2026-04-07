from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import fitz  # type: ignore[import-not-found] # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

logger = logging.getLogger(__name__)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_groq_key(*, env_path: str | Path | None = None) -> str:
    """
    Load GROQ_API_KEY from environment (optionally .env).

    Keep this here (not in the parser) so the parser can stay focused on orchestration.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(env_path) if env_path else (repo_root() / ".env"))
    except ImportError:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()


def load_prompt_template_from_yaml(
    yaml_path: str | Path,
    *,
    template_field: str = "template",
) -> str:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt YAML not found: {path}")
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise ImportError("PyYAML is required. Install it via `pip install pyyaml`.") from e

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Prompt YAML must be a mapping object: {path}")
    template = data.get(template_field)
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"Prompt YAML missing non-empty '{template_field}' field: {path}")
    return template


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    if fitz is None:
        raise ImportError("PyMuPDF is required. Install it via `pip install pymupdf`.")
    path = Path(pdf_path)
    doc = fitz.open(path)
    try:
        return "".join(page.get_text() for page in doc)
    finally:
        doc.close()


def split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Character windows with overlap (no extra deps vs RecursiveCharacterTextSplitter)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")

    if len(text) <= chunk_size:
        return [text]
    step = max(chunk_size - chunk_overlap, 1)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def parse_json_from_llm(content: str) -> dict[str, Any] | None:
    """Strip optional markdown fences and parse the first JSON object."""
    text = content.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}\s*$", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.debug("JSON decode failed for extracted brace block")
    return None


def deep_merge_extraction(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge chunk JSON: recurse into dicts; prefer longer strings; fill missing leaves."""
    for key, new_val in incoming.items():
        if new_val is None:
            continue
        if key not in base:
            base[key] = new_val
            continue
        old_val = base[key]
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            deep_merge_extraction(old_val, new_val)
        else:
            base[key] = _merge_scalar_or_replace(old_val, new_val)
    return base


def build_chunker(*, chunk_size: int, chunk_overlap: int) -> Callable[[str], list[str]]:
    return lambda text: split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _is_empty_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _merge_scalar_or_replace(old_val: Any, new_val: Any) -> Any:
    if _is_empty_scalar(old_val):
        return new_val
    if isinstance(old_val, str) and isinstance(new_val, str):
        o, n = old_val.strip(), new_val.strip()
        if not n:
            return old_val
        if not o or len(n) > len(o):
            return n
        return old_val
    if new_val not in (None, "", {}, []):
        return new_val
    return old_val
