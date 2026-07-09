# 🎙️ Infinity Khmer ASR — Production API

ប្រព័ន្ធ API បំលែងសម្ដីភាសាខ្មែរទៅជាអក្សរ (Speech-to-Text)  
ផ្អែកលើ Model `SoyVitou/infinity-khmer-asr` — trained with 200 hours of Khmer audio.

---

## 📁 File Structure

```
infinity-khmer-asr/
├── main.py                ← FastAPI app (entry point)
├── requirements.txt       ← Python dependencies
├── render.yaml            ← Render deployment config
├── Dockerfile             ← Optional Docker build
├── libs/
│   ├── asr.py             ← ASR model (100% original)
│   └── inverse-text.py    ← InverseText logic (100% original)
├── static/
│   ├── index.html         ← Frontend UI
│   ├── style.css          ← Glassmorphism design
│   └── script.js          ← WaveSurfer + Transcribe logic
└── test/
    └── *.wav              ← Audio test files
```

---

## 🔑 API Endpoints

| Method | Path         | Description                    |
|--------|--------------|-------------------------------|
| GET    | `/`          | Frontend UI / Health check    |
| GET    | `/health`    | Service status + model status |
| POST   | `/transcribe`| Upload audio → Khmer text     |
| GET    | `/docs`      | Swagger UI                    |
| GET    | `/redoc`     | ReDoc UI                      |

### POST `/transcribe`

```bash
curl -X POST https://your-app.onrender.com/transcribe \
  -F "audio=@test/sample.wav"
```

**Response (Success):**
```json
{
  "success": true,
  "text": "ខ្ញុំចង់ផ្ញើប្រាក់",
  "inverse_text": "ខ្ញុំចង់ផ្ញើប្រាក់",
  "language": "km",
  "processing_time_ms": 1842.5,
  "total_time_ms": 1950.2,
  "device": "cpu",
  "request_id": "a1b2c3d4"
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "audio file not found",
  "request_id": "a1b2c3d4"
}
```

---

## 💻 ដំឡើងដំណើរការ Locally

### ១. Clone / Extract Project

```bash
git clone https://github.com/your-username/infinity-khmer-asr.git
cd infinity-khmer-asr
```

### ២. បង្កើត Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### ៣. ដំឡើង PyTorch (CPU)

```bash
pip install torch==2.5.1+cpu torchaudio==2.5.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu
```

> ⚠️ ប្រើ CPU version ដើម្បីកំហែងទំហំ RAM លើ Render Free tier

### ៤. ដំឡើង Dependencies ផ្សេងៗ

```bash
pip install -r requirements.txt
```

### ៥. កំណត់ HF_TOKEN

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

> HF Token ចាំបាច់ ដើម្បី download model `SoyVitou/infinity-khmer-asr`  
> ទទួល token នៅ: https://huggingface.co/settings/tokens

### ៦. Start Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### ៧. Test API

```bash
# Health check
curl http://localhost:8000/health

# Transcribe audio
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@test/20260321_061453_385188.wav"
```

### ៨. Open Frontend

```
http://localhost:8000
```

---

## 🌐 Deploy ទៅ Render

### ជំហាន ១ — រៀបចំ GitHub

```bash
# Initialize git (ប្រសិនបើមិនទាន់មាន)
git init
git add .
git commit -m "feat: Khmer ASR Production API"

# Push ទៅ GitHub
git remote add origin https://github.com/your-username/infinity-khmer-asr.git
git branch -M main
git push -u origin main
```

> **ចំណាំ:** ត្រូវមាន `.gitignore` ដើម្បីមិន push `venv/`, `.cache/`, `__pycache__/`

**`.gitignore` ដែលត្រូវបង្កើត:**
```
venv/
__pycache__/
*.pyc
.cache/
*.egg-info/
dist/
build/
.env
```

### ជំហាន ២ — បង្កើត Render Service

1. ចូល [render.com](https://render.com) → **New** → **Web Service**
2. ភ្ជាប់ GitHub repository របស់អ្នក
3. Render នឹងរក `render.yaml` ដោយស្វ័យប្រវត្តិ

### ជំហាន ៣ — Environment Variables

ក្នុង Render Dashboard → **Environment** → **Add Environment Variable:**

| Key | Value | Note |
|-----|-------|------|
| `HF_TOKEN` | `hf_xxxx...` | ⚠️ Secret — Hugging Face token |
| `MAX_FILE_SIZE_MB` | `50` | អតិបរមាទំហំ file |
| `CORS_ORIGINS` | `*` | ឬដាក់ domain ជាក់លាក់ |

### ជំហាន ៤ — Build & Start Commands

| Field | Value |
|-------|-------|
| **Build Command** | `pip install torch==2.5.1+cpu torchaudio==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple && pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 120` |

### ជំហាន ៥ — Persistent Disk (Optional, Free tier)

Render Free tier ផ្តល់ Disk 1GB ។ ប្រើ disk ដើម្បី cache HF model:

```yaml
disk:
  name: model-cache
  mountPath: /opt/render/.cache
  sizeGB: 10
```

> ⚠️ **Render Free tier** — Service spin down ក្រោយ 15 min idle  
> ការ download model ដំបូង (~500MB) អាចចំណាយ 5-10 min

### ជំហាន ៦ — Test Endpoint

```bash
# Health
curl https://your-app.onrender.com/health

# Transcribe
curl -X POST https://your-app.onrender.com/transcribe \
  -F "audio=@test/sample.wav" \
  | python3 -m json.tool
```

---

## ⚙️ Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (required) | Hugging Face API token |
| `PORT` | `8000` | Server port (Render ផ្ដល់ auto) |
| `HOST` | `0.0.0.0` | Server host |
| `MAX_FILE_SIZE_MB` | `50` | Max upload size in MB |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `ASR_DEVICE` | auto | Force device: `cpu` or `cuda:0` |
| `HF_HOME` | system cache | Hugging Face cache directory |
| `TRANSFORMERS_CACHE` | system cache | Transformers model cache |

---

## 🐳 Docker Deployment (Alternative)

```bash
# Build
docker build -t khmer-asr .

# Run
docker run -p 8000:8000 \
  -e HF_TOKEN="hf_xxxx" \
  khmer-asr
```

---

## 📊 Supported Audio Formats

| Format | Extension |
|--------|-----------|
| WAV | `.wav` |
| MP3 | `.mp3` |
| M4A | `.m4a` |
| WebM | `.webm` |
| OGG | `.ogg` |
| FLAC | `.flac` |
| Opus | `.opus` |
| AAC | `.aac` |

- **Max file size:** 50MB
- **Max audio duration:** 30 seconds (ត្រូវបន្ថែម/កាត់ ដោយ model)
- **Sample rate:** auto-resampled to 16kHz

---

## 🔧 Troubleshooting

### Model fails to load
```
RuntimeError: HF_TOKEN not found
```
→ ត្រូវ set `HF_TOKEN` environment variable

### CORS error from frontend
→ set `CORS_ORIGINS=https://your-frontend.vercel.app` ក្នុង Render env

### Out of Memory on Render Free
→ Render Free tier RAM 512MB ។ Model ~300MB ។ ប្រើ CPU-only torch
→ ប្រើ Starter plan ($7/month) ប្រសិនបើ OOM error

### Render timeout (15 minutes idle)
→ ប្រើ UptimeRobot ដើម្បី ping `/health` រៀង 10 min

---

## 👨‍💻 Developer

**Amertak Network** — Building tools for the Khmer community  
Model by [@SoyVitou](https://huggingface.co/SoyVitou) — trained with 200 hours Khmer audio

License: Apache 2.0
