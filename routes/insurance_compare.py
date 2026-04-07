import asyncio
import os
import tempfile
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from documentcomparison_parser import InsuranceComparisonParser

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

SECTION_KEY_RENAMES = {
    "section_1_policy_scope": "benefits_section",
    "inpatient_benefits": "section_two_inpatient_benefits",
    "outpatient_benefits": "section_three_outpatient_benefits",
    "additional_benefits": "section_four_additional_benefits",
}


def _rename_section_keys(data: dict[str, Any]) -> dict[str, Any]:
    for old_key, new_key in SECTION_KEY_RENAMES.items():
        if old_key not in data:
            continue
        if new_key not in data:
            data[new_key] = data[old_key]
        data.pop(old_key, None)
    return data


async def _parse_one_pdf(
    *,
    file: UploadFile,
    parser: InsuranceComparisonParser,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    filename = file.filename or "unknown.pdf"

    if not filename.lower().endswith(".pdf"):
        return {"filename": filename, "ok": False, "error": "Only PDF files are allowed"}

    tmp_path: str | None = None
    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            return {"filename": filename, "ok": False, "error": "File too large (max 10MB)"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name

        async with semaphore:
            data = await asyncio.to_thread(parser.extract, tmp_path)
            if isinstance(data, dict):
                data = _rename_section_keys(data)

        return {"filename": filename, "ok": True, "data": data}
    except Exception as e:
        return {"filename": filename, "ok": False, "error": str(e)}
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


@router.post("/insurance/compare-pdfs/", tags=["Insurance Comparison"])
async def compare_insurance_pdfs(
    files: list[UploadFile] = File(...),
):
    """
    Upload multiple PDFs and parse each in parallel.

    Returns per-file results so one failure doesn't fail the whole batch.
    """
    if not files:
        raise HTTPException(status_code=422, detail="No files provided")

    parser = InsuranceComparisonParser()
    cpu = os.cpu_count() or 4
    max_concurrency = max(1, min(20, len(files), cpu))
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        _parse_one_pdf(file=f, parser=parser, semaphore=semaphore)
        for f in files
    ]
    results = await asyncio.gather(*tasks)

    return {"count": len(results), "results": results}

