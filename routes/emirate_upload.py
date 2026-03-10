import io
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, File, HTTPException, UploadFile

from langchain_groq import ChatGroq

from routes.utils import extract_pdf_info1
from utils.helper import valid_emirates_id

try:
    import docx  # type: ignore
except Exception:  # pragma: no cover
    docx = None


router = APIRouter(prefix="/api/emirate_upload", tags=["Emirates ID"])

_EID_PATTERN = re.compile(r"\b784-\d{4}-\d{7}-\d\b")


def _find_emirates_id_in_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = _EID_PATTERN.search(text)
    return m.group(0) if m else None


def _find_emirates_id_in_any(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        return _find_emirates_id_in_text(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            found = _find_emirates_id_in_any(v)
            if found:
                return found
        return None
    if isinstance(obj, (list, tuple)):
        for v in obj:
            found = _find_emirates_id_in_any(v)
            if found:
                return found
        return None
    return None


def _extract_text_from_txt_bytes(b: bytes) -> str:
    # best-effort decoding for user uploads
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="ignore")


def _extract_text_from_docx_bytes(b: bytes) -> str:
    if docx is None:
        raise RuntimeError("python-docx is not installed (required for .docx)")
    d = docx.Document(io.BytesIO(b))
    parts = [p.text for p in d.paragraphs if p.text]
    return "\n".join(parts).strip()


def _llm_extract_emirates_id_from_text(text: str) -> Optional[str]:
    """
    Fallback for txt/docx when regex fails: ask LLM to find Emirates ID.
    Returns a single EID string or None.
    """
    if not text or not text.strip():
        return None
    llm = ChatGroq(
        model=os.getenv("LLM_MODEL"),
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    prompt = f"""
You are extracting an Emirates ID number from text.
Return ONLY the Emirates ID in this exact format: 784-YYYY-NNNNNNN-C
If you cannot find it, return ONLY an empty string.

Text:
---
{text[:20000]}
---
"""
    resp = llm.invoke(prompt)
    content = (resp.content if hasattr(resp, "content") else str(resp)).strip()
    eid = _find_emirates_id_in_text(content) or _find_emirates_id_in_text(text)
    if eid and valid_emirates_id(eid):
        return eid
    return eid


async def _process_pdf_or_image(file: UploadFile) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    file_extension = os.path.splitext(file.filename or "")[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        try:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()

            extracted = await extract_pdf_info1(temp_file.name)
            eid = None
            if isinstance(extracted, dict):
                candidate = extracted.get("id_number") or extracted.get("id") or extracted.get("emirates_id")
                if isinstance(candidate, str) and valid_emirates_id(candidate.strip()):
                    eid = candidate.strip()
                else:
                    eid = _find_emirates_id_in_any(extracted)
            return extracted, eid
        finally:
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass


def _process_txt_or_docx(filename: str, content: bytes) -> Tuple[Dict[str, Any], Optional[str]]:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".txt":
        text = _extract_text_from_txt_bytes(content)
    elif ext == ".docx":
        text = _extract_text_from_docx_bytes(content)
    else:
        raise ValueError(f"Unsupported text file type: {ext}")

    eid = _find_emirates_id_in_text(text)
    if not eid:
        eid = _llm_extract_emirates_id_from_text(text)
    return {"text": text}, eid


@router.post("/extract")
async def extract_emirates_id_from_documents(
    files: List[UploadFile] = File(...),
):
    """
    Upload multiple documents (pdf, jpg/jpeg/png, txt, docx) and extract Emirates ID (EID)
    per document. PDFs/images use the vision extraction flow; txt/docx are parsed as text.
    """
    if not files:
        raise HTTPException(status_code=422, detail="No files provided")

    results: List[Dict[str, Any]] = []
    for f in files:
        filename = f.filename or "uploaded"
        ext = os.path.splitext(filename)[1].lower()
        try:
            if ext in (".pdf", ".jpg", ".jpeg", ".png"):
                extracted, eid = await _process_pdf_or_image(f)
                results.append(
                    {
                        "filename": filename,
                        "emirates_id": eid,
                        "valid_emirates_id": bool(eid and valid_emirates_id(eid)),
                        "extracted": extracted,
                        "error": None,
                    }
                )
            elif ext in (".txt", ".docx"):
                content = await f.read()
                extracted, eid = _process_txt_or_docx(filename, content)
                results.append(
                    {
                        "filename": filename,
                        "emirates_id": eid,
                        "valid_emirates_id": bool(eid and valid_emirates_id(eid)),
                        "extracted": extracted,
                        "error": None,
                    }
                )
            else:
                results.append(
                    {
                        "filename": filename,
                        "emirates_id": None,
                        "valid_emirates_id": False,
                        "extracted": None,
                        "error": f"Unsupported file type: {ext}. Supported: .pdf, .jpg, .jpeg, .png, .txt, .docx",
                    }
                )
        except Exception as e:
            results.append(
                {
                    "filename": filename,
                    "emirates_id": None,
                    "valid_emirates_id": False,
                    "extracted": None,
                    "error": str(e),
                }
            )

    return {"documents": results, "count": len(results)}

