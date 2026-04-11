"""Insurance PDF text extraction and structured LLM parsing (Groq)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from .utils import (
    deep_merge_extraction,
    extract_text_from_pdf,
    load_groq_key,
    load_prompt_template_from_yaml,
    parse_json_from_llm,
)

logger = logging.getLogger(__name__)

class InsuranceComparisonParser:
    """Extract structured insurance benefits/pricing from a PDF via Groq.

    Merged JSON includes top-level ``company_name`` (insurer/underwriter as printed on the PDF),
    filled from the chunked extraction prompt when present in the document text.
    """

    DEFAULT_MODEL = "openai/gpt-oss-120b"
    CHUNK_CHARS = 8000
    CHUNK_OVERLAP = 500
    MAX_PROMPT_CHUNK = 8000
    DEFAULT_PROMPT_YAML = Path(__file__).resolve().parent / "prompts" / "uae_health_insurance.yaml"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.0,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        max_prompt_chunk: int | None = None,
        prompt_template: str | None = None,
        prompt_yaml_path: str | Path | None = None,
        text_extractor: Callable[[str | Path], str] = extract_text_from_pdf,
        json_parser: Callable[[str], dict[str, Any] | None] = parse_json_from_llm,
        merge_func: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] = deep_merge_extraction,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._model_name = model_name or self.DEFAULT_MODEL
        self._temperature = temperature
        self._chunk_size = self.CHUNK_CHARS if chunk_size is None else chunk_size
        self._chunk_overlap = self.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self._max_prompt_chunk = self.MAX_PROMPT_CHUNK if max_prompt_chunk is None else max_prompt_chunk
        self._prompt_template = (
            prompt_template
            if isinstance(prompt_template, str) and prompt_template.strip()
            else load_prompt_template_from_yaml(prompt_yaml_path or self.DEFAULT_PROMPT_YAML)
        )
        self._text_extractor = text_extractor
        self._json_parser = json_parser
        self._merge_func = merge_func

    def extract(self, pdf_path: str | Path) -> dict[str, Any]:
        key = self._api_key or load_groq_key()
        if not key:
            raise ValueError(
                "GROQ_API_KEY is missing: add it to .env in the project root or pass api_key=..."
            )

        llm = ChatGroq(
            groq_api_key=key,
            model_name=self._model_name,
            temperature=self._temperature,
        )

        raw_text = self._text_extractor(pdf_path)
        chunks = _split_text(raw_text, chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)

        merged: dict[str, Any] = {}

        for i, chunk in enumerate(chunks):
            prompt = self._prompt_template.format(chunk=chunk[: self._max_prompt_chunk])
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content if isinstance(response.content, str) else str(response.content)
            data = self._json_parser(content)
            if not isinstance(data, dict):
                logger.warning("Chunk %s: no valid JSON, skipped", i)
                continue
            self._merge_func(merged, data)

        return merged


def _split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Kept private to this module: it's part of the parser's "strategy" (chunk sizing) not a global utility.
    """
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pdf = Path(__file__).resolve().parent.parent / "Metlife_SP & Green_461k.pdf"
    parser = InsuranceComparisonParser()
    _ = parser.extract(pdf)
