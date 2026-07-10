import { config } from '../config.js';

export async function transcribeApi(audioBuffer) {
  if (!config.HF_API_TOKEN) {
    throw new Error('HF_API_TOKEN មិនទាន់បានដាក់ — ចូល Render Environment Variables');
  }
  
  const response = await fetch(
    `https://api-inference.huggingface.co/models/${config.HF_API_MODEL}`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${config.HF_API_TOKEN}`,
        'Content-Type': 'application/octet-stream',
      },
      body: audioBuffer,
    }
  );
  
  if (!response.ok) {
    const err = await response.text();
    throw new Error(`HuggingFace API error ${response.status}: ${err}`);
  }
  
  const result = await response.json();
  
  return {
    text: (result.text || '').trim(),
    language: 'km',
    backend: 'hf-api'
  };
}