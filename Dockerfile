# ─────────────────────────────────────────────────────────────
#  Infinity Khmer ASR — Production Dockerfile (CPU)
#  Compatible with Render Free/Starter plan
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies for audio processing (pydub / ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install PyTorch CPU first (smaller image, avoids CUDA bloat) ──
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining Python dependencies ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ──
COPY . .

# ── Create temp upload directory ──
RUN mkdir -p /tmp/khmer_asr_uploads

# ── Cache directory for Hugging Face models ──
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV USE_TF=0
ENV TRANSFORMERS_NO_TF=1

# ── Expose port ──
EXPOSE 8000

# ── Run app ──
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]
