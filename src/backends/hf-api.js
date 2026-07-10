import { HfInference } from '@huggingface/inference';
import { config } from '../config.js';

const hf = new HfInference(config.HF_API_TOKEN);

export async function transcribeApi(audioBuffer) {
  if (!config.HF_API_TOKEN) {
    throw new Error('HF_API_TOKEN is required for hf-api backend. Set it in Environment variables.');
  }

  const result = await hf.automaticSpeechRecognition({
    model: config.HF_API_MODEL,
    data: new Blob([audioBuffer])
  });

  return {
    text: (result.text || '').trim(),
    language: 'km',
    backend: 'hf-api'
  };
}
