import { pipeline } from '@huggingface/transformers';
import { config } from '../config.js';

let transcriber = null;

export async function initLocalModel() {
  if (transcriber) return transcriber;

  console.log('Loading local model: ' + config.TRANSFORMERS_MODEL);
  console.log('First run downloads and caches the model (may take 1-5 minutes)');

  transcriber = await pipeline(
    'automatic-speech-recognition',
    config.TRANSFORMERS_MODEL,
    {
      dtype: 'fp32',
      device: 'cpu'
    }
  );

  console.log('Local model loaded successfully');
  return transcriber;
}

export async function transcribeLocal(audioData) {
  const model = await initLocalModel();
  const result = await model(audioData, {
    language: 'khmer',
    task: 'transcribe',
    return_timestamps: false
  });

  return {
    text: (result.text || '').trim(),
    language: 'km',
    backend: 'transformers-local'
  };
}
