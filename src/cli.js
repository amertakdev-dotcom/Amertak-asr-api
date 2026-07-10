import { transcribeAudio } from './transcribe.js';
import { config } from './config.js';
import fs from 'fs';

async function main() {
  const args = process.argv.slice(2);
  const filePath = args[0];

  if (!filePath || !fs.existsSync(filePath)) {
    console.log(`
Khmer ASR CLI

Usage:
  node src/cli.js <audio-file> [options]

Options:
  --backend=hf-api        Use Hugging Face API (default on Render)
  --backend=transformers  Use local Transformers.js

Examples:
  node src/cli.js ./audio.mp3
  node src/cli.js ./audio.wav --backend=transformers
`);
    process.exit(1);
  }

  const backendArg = args.find(a => a.startsWith('--backend='));
  const backend = backendArg ? backendArg.split('=')[1] : config.DEFAULT_BACKEND;

  console.log('Transcribing: ' + filePath);
  console.log('Backend: ' + backend);

  try {
    const result = await transcribeAudio(filePath, backend);
    console.log(`
Result:
-------------------------`);
    console.log('Text: ' + result.text);
    console.log('Backend: ' + result.backend);
    console.log('Duration: ' + result.duration);
    console.log('-------------------------');
  } catch (error) {
    console.error('Error: ' + error.message);
    process.exit(1);
  }
}

main();
