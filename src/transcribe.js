import { convertToWav, readWavFile, needsConversion, ensureDir } from './audio-utils.js';
import { transcribeLocal } from './backends/transformers-local.js';
import { transcribeApi } from './backends/hf-api.js';
import { config } from './config.js';
import fs from 'fs';
import path from 'path';

export async function transcribeAudio(filePath, backend = config.DEFAULT_BACKEND) {
  const startTime = Date.now();
  ensureDir(config.UPLOAD_DIR);

  let wavPath = filePath;
  if (needsConversion(filePath)) {
    const fileName = path.basename(filePath, path.extname(filePath)) + '_16k.wav';
    wavPath = path.join(config.UPLOAD_DIR, fileName);
    await convertToWav(filePath, wavPath);
  }

  const audioData = readWavFile(wavPath);
  let result;

  if (backend === 'hf-api') {
    const buffer = fs.readFileSync(wavPath);
    result = await transcribeApi(buffer);
  } else {
    result = await transcribeLocal(audioData);
  }

  const duration = Date.now() - startTime;

  if (wavPath !== filePath && fs.existsSync(wavPath)) {
    fs.unlinkSync(wavPath);
  }

  return {
    ...result,
    duration: duration + 'ms',
    audioLength: audioData.length + ' samples'
  };
}
