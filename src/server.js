import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { transcribeAudio } from './transcribe.js';
import { config } from './config.js';
import { ensureDir } from './audio-utils.js';

const app = express();
const upload = multer({ dest: config.UPLOAD_DIR });

ensureDir(config.UPLOAD_DIR);
ensureDir('public');

app.use(express.json());
app.use(express.static('public'));

// Render health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'khmer-asr', timestamp: new Date().toISOString() });
});

// Serve web UI
app.get('/', (req, res) => {
  res.sendFile(path.resolve('public/index.html'));
});

// Transcribe uploaded file
app.post('/transcribe', upload.single('audio'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No audio file uploaded' });
    }

    const backend = req.body.backend || config.DEFAULT_BACKEND;
    console.log('[' + new Date().toISOString() + '] Transcribing: ' + req.file.originalname + ' via ' + backend);

    const result = await transcribeAudio(req.file.path, backend);

    if (fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);

    res.json(result);
  } catch (error) {
    console.error('Transcription error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Transcribe from URL
app.post('/transcribe/url', async (req, res) => {
  try {
    const { url, backend } = req.body;
    if (!url) return res.status(400).json({ error: 'URL is required' });

    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to download audio');

    const ext = path.extname(new URL(url).pathname) || '.audio';
    const fileName = 'dl_' + Date.now() + ext;
    const filePath = path.join(config.UPLOAD_DIR, fileName);

    const buffer = Buffer.from(await response.arrayBuffer());
    fs.writeFileSync(filePath, buffer);

    const result = await transcribeAudio(filePath, backend || config.DEFAULT_BACKEND);
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);

    res.json(result);
  } catch (error) {
    console.error('URL transcription error:', error);
    res.status(500).json({ error: error.message });
  }
});

// List models
app.get('/models', (req, res) => {
  res.json({
    local: config.TRANSFORMERS_MODEL,
    api: config.HF_API_MODEL,
    current_backend: config.DEFAULT_BACKEND
  });
});

const PORT = config.PORT;
app.listen(PORT, '0.0.0.0', () => {
  console.log('Khmer ASR Server listening on http://0.0.0.0:' + PORT);
  console.log('Environment: ' + (process.env.NODE_ENV || 'development'));
  console.log('Backend: ' + config.DEFAULT_BACKEND);
});
