import fs from 'fs';
import path from 'path';
import wavefileModule from 'wavefile';
import ffmpeg from 'fluent-ffmpeg';

const { WaveFile } = wavefileModule;

/**
 * Convert any audio to 16kHz mono WAV (required by Whisper/ASR)
 */
export function convertToWav(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .toFormat('wav')
      .audioChannels(1)
      .audioFrequency(16000)
      .audioCodec('pcm_s16le')
      .on('end', () => resolve(outputPath))
      .on('error', (err) => reject(err))
      .save(outputPath);
  });
}

/**
 * Read WAV file into Float32Array
 */
export function readWavFile(wavPath) {
  const buffer = fs.readFileSync(wavPath);
  const wav = new WaveFile(buffer);
  wav.toBitDepth('32f');
  wav.toSampleRate(16000);
  let audioData = wav.getSamples();
  if (Array.isArray(audioData)) audioData = audioData[0];
  return audioData;
}

export function needsConversion(filePath) {
  return path.extname(filePath).toLowerCase() !== '.wav';
}

export function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}