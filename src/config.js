import dotenv from 'dotenv';
dotenv.config();

export const config = {
  PORT: process.env.PORT || 3000,
  HF_API_TOKEN: process.env.HF_API_TOKEN || '',
  DEFAULT_BACKEND: process.env.DEFAULT_BACKEND || 'hf-api',
  TRANSFORMERS_MODEL: process.env.TRANSFORMERS_MODEL || 'seanghay/whisper-small-khmer',
  HF_API_MODEL: process.env.HF_API_MODEL || 'ksoky/whisper-large-khmer-asr',
  UPLOAD_DIR: './uploads',
  CACHE_DIR: './cache',
  SUPPORTED_FORMATS: ['.wav', '.mp3', '.mp4', '.m4a', '.ogg', '.flac', '.webm']
};
