# Khmer ASR Server

Production-ready Khmer Automatic Speech Recognition for Render.com.

## Deploy in 1 Click

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## Quick Start (Local)

1. Install dependencies:
   npm install

2. Install FFmpeg (required for audio conversion):
   macOS: brew install ffmpeg
   Ubuntu: sudo apt-get install ffmpeg

3. Copy environment file:
   cp .env.example .env
   Edit .env and add your HF_API_TOKEN

4. Run server:
   npm start

Server runs at http://localhost:3000

## Render.com Deployment

### Method 1: One-Click (Blueprint)
1. Push this code to a GitHub repo
2. Click the Deploy to Render button above
3. In Render dashboard, add your HF_API_TOKEN as an environment variable
4. Deploy

### Method 2: Manual
1. Create a new Web Service on Render
2. Connect your GitHub repo
3. Set Runtime: Node
4. Set Build Command: npm install
5. Set Start Command: npm start
6. Add Environment Variables:
   - DEFAULT_BACKEND: hf-api
   - HF_API_TOKEN: hf_your_token_here
   - HF_API_MODEL: ksoky/whisper-large-khmer-asr
7. Deploy

## API Endpoints

GET /        - Web UI
GET /health  - Render health check
POST /transcribe - Upload audio file (multipart)
POST /transcribe/url - Transcribe from audio URL
GET /models  - List configured models

## Important Notes for Render

| Tier | RAM | Recommendation |
|------|-----|----------------|
| Starter   | 512MB | Use hf-api backend only |
| Standard  | 2GB   | Can run whisper-small locally |
| Pro+      | 4GB+  | Can run whisper-large locally |

Get free token at https://huggingface.co/settings/tokens

## Models

| Model | Size | Best For |
|-------|------|----------|
| ksoky/whisper-large-khmer-asr | 1.5GB | Best accuracy (API) |
| seanghay/whisper-small-khmer | 250MB | Local/lower RAM |
| seanghay/Qwen3-ASR-0.6B-Khmer | 600MB | Modern, fast |

## License
MIT
