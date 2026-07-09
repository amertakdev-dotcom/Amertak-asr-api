from __future__ import annotations

# ─────────────────────────────────────────────────────────────
#  Silence TensorFlow / ONEDNN noise before any other import
# ─────────────────────────────────────────────────────────────
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import importlib.util
import logging
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from libs.asr import predict, is_model_loaded, load_model

# ─────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("khmer-asr")

# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".opus", ".aac"}
MAX_FILE_SIZE_MB   = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

TEMP_DIR = Path(tempfile.gettempdir()) / "khmer_asr_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"

# ─────────────────────────────────────────────────────────────
#  Load InverseText (same logic as original app.py)
# ─────────────────────────────────────────────────────────────
def load_inverse_text_function() -> Callable[[str], str]:
    inverse_file = Path("./libs/inverse-text.py")

    if not inverse_file.exists():
        raise FileNotFoundError(f"could not find inverse text file: {inverse_file}")

    spec = importlib.util.spec_from_file_location(
        "inverse_text_module",
        str(inverse_file),
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"could not load inverse text module from: {inverse_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "InverseText"):
        raise AttributeError(f"{inverse_file} must contain function InverseText(text)")

    return module.InverseText


InverseText = load_inverse_text_function()
logger.info("InverseText function loaded successfully")

# ─────────────────────────────────────────────────────────────
#  FastAPI App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Infinity Khmer ASR API",
    description="Production Khmer Speech Recognition API — powered by Infinity Khmer ASR (SoyVitou/infinity-khmer-asr)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────────────────────
#  CORS
# ─────────────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "*",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
#  Serve Frontend static files (index.html, style.css, script.js)
# ─────────────────────────────────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    logger.info(f"Serving static files from: {STATIC_DIR}")

# ─────────────────────────────────────────────────────────────
#  Startup: pre-load model
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("=== Infinity Khmer ASR API starting up ===")
    logger.info(f"MAX_FILE_SIZE_MB : {MAX_FILE_SIZE_MB} MB")
    logger.info(f"TEMP_DIR         : {TEMP_DIR}")
    logger.info("Pre-loading ASR model on startup...")
    try:
        load_model()
        logger.info("ASR model pre-loaded successfully ✓")
    except Exception as e:
        logger.error(f"Model pre-load failed: {e}")
        logger.warning("Model will be loaded on first /transcribe request instead")

# ─────────────────────────────────────────────────────────────
#  Helper: validate & save upload to temp file
# ─────────────────────────────────────────────────────────────
async def save_upload_to_temp(file: UploadFile) -> Path:
    # Check extension
    original_name = file.filename or "audio"
    ext = Path(original_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail={
                "success": False,
                "error": f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            },
        )

    # Read content
    content = await file.read()

    # Check file size
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "success": False,
                "error": f"File too large ({len(content) / 1024 / 1024:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB",
            },
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": "Uploaded file is empty"},
        )

    # Save to unique temp path
    temp_path = TEMP_DIR / f"{uuid.uuid4()}{ext}"
    temp_path.write_bytes(content)

    logger.info(f"Saved upload: {original_name} → {temp_path} ({len(content)} bytes)")
    return temp_path

# ─────────────────────────────────────────────────────────────
#  Helper: cleanup temp file
# ─────────────────────────────────────────────────────────────
def cleanup_temp(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
            logger.debug(f"Cleaned up temp file: {path}")
    except Exception as e:
        logger.warning(f"Could not delete temp file {path}: {e}")

# ─────────────────────────────────────────────────────────────
#  GET /  → serve frontend index.html (or JSON health if no static)
# ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")

    return JSONResponse({
        "status": "online",
        "service": "Khmer ASR API",
        "version": "1.0.0",
        "model": "SoyVitou/infinity-khmer-asr",
        "model_loaded": is_model_loaded(),
        "endpoints": {
            "health": "GET /health",
            "transcribe": "POST /transcribe",
            "docs": "GET /docs",
        },
    })

# ─────────────────────────────────────────────────────────────
#  GET /health
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return JSONResponse({
        "status": "online",
        "service": "Khmer ASR API",
        "model_loaded": is_model_loaded(),
    })

# ─────────────────────────────────────────────────────────────
#  POST /transcribe   ← MAIN ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Transcribe Khmer audio to text.

    - **audio**: audio file (wav / mp3 / m4a / webm / ogg / flac / opus / aac)

    Returns transcribed Khmer text + inverse-normalized text + processing time.
    """
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] /transcribe request: file={audio.filename}, content_type={audio.content_type}")

    temp_path: Path | None = None
    start_time = time.perf_counter()

    try:
        # Save upload
        temp_path = await save_upload_to_temp(audio)

        # Run ASR inference (original predict from libs/asr.py — 100% preserved)
        result = predict(str(temp_path))

        total_time_ms = (time.perf_counter() - start_time) * 1000

        if not result.get("success"):
            error_msg = result.get("error", "unknown ASR error")
            logger.error(f"[{request_id}] ASR failed: {error_msg}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": error_msg,
                    "request_id": request_id,
                },
            )

        transcription = result.get("transcription", "") or ""

        # Run InverseText (same logic as original app.py transcribe_easier)
        try:
            inverse_text = InverseText(transcription)
        except Exception as inv_err:
            inverse_text = transcription
            logger.warning(f"[{request_id}] InverseText error: {inv_err}")

        logger.info(
            f"[{request_id}] Done — "
            f"transcript_len={len(transcription)}, "
            f"asr_ms={result.get('processing_time_ms', 0):.1f}, "
            f"total_ms={total_time_ms:.1f}, "
            f"device={result.get('device')}"
        )

        return JSONResponse({
            "success": True,
            "text": transcription,
            "inverse_text": inverse_text,
            "language": "km",
            "processing_time_ms": result.get("processing_time_ms", 0),
            "total_time_ms": round(total_time_ms, 2),
            "device": result.get("device"),
            "request_id": request_id,
        })

    except HTTPException:
        raise

    except Exception as e:
        total_time_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(f"[{request_id}] Unexpected error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "request_id": request_id,
            },
        )

    finally:
        if temp_path:
            cleanup_temp(temp_path)

# ─────────────────────────────────────────────────────────────
#  Global exception handler
# ─────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"},
    )

# ─────────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info",
        timeout_keep_alive=120,
        timeout_graceful_shutdown=30,
    )
