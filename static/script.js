/* ─────────────────────────────────────────────
   Infinity Khmer ASR — Frontend Script
   WaveSurfer · Upload · Record · Transcribe
   ───────────────────────────────────────────── */

"use strict";

// ─── API endpoint (auto-detect same origin) ───────────────────
const API_URL = window.location.origin;

// ─── State ────────────────────────────────────────────────────
let wavesurfer    = null;
let currentAudio  = null;   // { blob, url, name }
let mediaRecorder = null;
let recChunks     = [];
let recInterval   = null;
let recSeconds    = 0;
let recStream     = null;
let analyserNode  = null;
let animFrame     = null;

// ─── DOM shortcuts ────────────────────────────────────────────
const $  = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

// ─── Init ─────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  initRecBars();
  setInterval(checkHealth, 30_000);
});

// ─── Health check ─────────────────────────────────────────────
async function checkHealth() {
  const dot  = $("statusDot");
  const text = $("statusText");
  try {
    const res  = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    if (data.status === "online") {
      dot.className  = "dot online";
      text.textContent = data.model_loaded ? "Online · Model Loaded" : "Online · Loading Model…";
    } else {
      throw new Error("not online");
    }
  } catch {
    dot.className  = "dot offline";
    text.textContent = "Offline";
  }
}

// ─── Tab switch ───────────────────────────────────────────────
function switchTab(name, el) {
  $$(".tab").forEach(t => t.classList.remove("active"));
  $$(".tab-panel").forEach(p => p.classList.add("hidden"));
  el.classList.add("active");
  $(`panel-${name}`).classList.remove("hidden");
}

// ─────────────────────────────────────────────────────────────
//  UPLOAD
// ─────────────────────────────────────────────────────────────
function handleDragOver(e) {
  e.preventDefault();
  $("dropzone").classList.add("over");
}
function handleDragLeave(e) {
  $("dropzone").classList.remove("over");
}
function handleDrop(e) {
  e.preventDefault();
  $("dropzone").classList.remove("over");
  const file = e.dataTransfer?.files?.[0];
  if (file) processFile(file);
}
function handleFileSelect(e) {
  const file = e.target.files?.[0];
  if (file) processFile(file);
  e.target.value = "";
}

function processFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  const allowed = ["wav","mp3","m4a","webm","ogg","flac","opus","aac"];
  if (!allowed.includes(ext)) {
    showToast(`ប្រភេទឯកសារ .${ext} មិនត្រូវបានគាំទ្រ`, "error");
    return;
  }
  if (file.size > 50 * 1024 * 1024) {
    showToast("ឯកសារធំជាង 50MB", "error");
    return;
  }
  const url = URL.createObjectURL(file);
  setAudio({ blob: file, url, name: file.name });
}

// ─────────────────────────────────────────────────────────────
//  RECORD
// ─────────────────────────────────────────────────────────────
function initRecBars() {
  const container = $("recBars");
  container.innerHTML = "";
  for (let i = 0; i < 28; i++) {
    const bar = document.createElement("div");
    bar.className = "rec-bar";
    bar.style.animationDelay = `${(i % 7) * 0.07}s`;
    container.appendChild(bar);
  }
}

async function toggleRecord() {
  if (mediaRecorder && mediaRecorder.state === "recording") return;

  try {
    recStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    showToast("មិនអាចចូលដំណើរការ Microphone បាន", "error");
    return;
  }

  // Analyser for bar animation
  const ctx    = new AudioContext();
  const source = ctx.createMediaStreamSource(recStream);
  analyserNode = ctx.createAnalyser();
  analyserNode.fftSize = 64;
  source.connect(analyserNode);
  animateRecBars();

  recChunks  = [];
  recSeconds = 0;

  mediaRecorder = new MediaRecorder(recStream, { mimeType: getSupportedMime() });
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recChunks.push(e.data); };
  mediaRecorder.onstop = finalizeRecord;
  mediaRecorder.start(100);

  recInterval = setInterval(() => {
    recSeconds++;
    $("recTimer").textContent = formatTime(recSeconds);
  }, 1000);

  $("btnRecord").style.display = "none";
  $("btnStop").style.display   = "flex";
  $("recHint").textContent     = "កំពុងថតសំឡេង... ចុច ■ ដើម្បីបញ្ឈប់";
  $$(".rec-bar").forEach(b => b.classList.add("active"));
}

function stopRecord() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  recStream?.getTracks().forEach(t => t.stop());
  clearInterval(recInterval);
  cancelAnimationFrame(animFrame);
  $$(".rec-bar").forEach(b => b.classList.remove("active"));
  $("btnRecord").style.display = "flex";
  $("btnStop").style.display   = "none";
  $("recHint").textContent     = "ចុចប៊ូតុងក្រហម ដើម្បីចាប់ផ្ដើមថត";
}

function finalizeRecord() {
  const mime = getSupportedMime();
  const ext  = mime.includes("webm") ? "webm" : mime.includes("ogg") ? "ogg" : "wav";
  const blob = new Blob(recChunks, { type: mime });
  const url  = URL.createObjectURL(blob);
  const name = `recorded-${Date.now()}.${ext}`;
  setAudio({ blob, url, name });
  // Switch to upload tab to show waveform
  switchTab("upload", $$(".tab")[0]);
  showToast("ថតបានរួច! ចុច \"បំលែងជាអក្សរ\"", "success");
}

function animateRecBars() {
  if (!analyserNode) return;
  const data = new Uint8Array(analyserNode.frequencyBinCount);
  analyserNode.getByteFrequencyData(data);
  const bars = $$(".rec-bar");
  bars.forEach((bar, i) => {
    const val = data[i % data.length] / 255;
    bar.style.height = `${10 + val * 90}%`;
  });
  animFrame = requestAnimationFrame(animateRecBars);
}

function getSupportedMime() {
  const types = ["audio/webm;codecs=opus","audio/webm","audio/ogg;codecs=opus","audio/ogg"];
  for (const t of types) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "audio/webm";
}

function formatTime(s) {
  const m = Math.floor(s / 60).toString().padStart(2,"0");
  const sec = (s % 60).toString().padStart(2,"0");
  return `${m}:${sec}`;
}

// ─────────────────────────────────────────────────────────────
//  SET AUDIO (shared between upload + record)
// ─────────────────────────────────────────────────────────────
function setAudio(audio) {
  // Release previous
  if (currentAudio?.url.startsWith("blob:")) URL.revokeObjectURL(currentAudio.url);
  currentAudio = audio;

  // Show file info
  $("fileName").textContent = audio.name;
  $("fileSize").textContent = formatBytes(audio.blob.size);
  $("fileInfo").style.display = "flex";

  // Enable transcribe
  $("btnTranscribe").disabled = false;

  // Load WaveSurfer
  loadWaveform(audio.url);
}

function clearFile() {
  if (currentAudio?.url.startsWith("blob:")) URL.revokeObjectURL(currentAudio.url);
  currentAudio = null;
  $("fileInfo").style.display   = "none";
  $("waveformWrap").style.display = "none";
  $("btnTranscribe").disabled   = true;
  $("resultPanel").style.display = "none";
  wavesurfer?.destroy();
  wavesurfer = null;
}

// ─────────────────────────────────────────────────────────────
//  WAVESURFER
// ─────────────────────────────────────────────────────────────
function loadWaveform(url) {
  $("waveformWrap").style.display = "block";

  if (wavesurfer) {
    wavesurfer.load(url);
    return;
  }

  wavesurfer = WaveSurfer.create({
    container: "#waveform",
    waveColor: "rgba(167,139,250,0.5)",
    progressColor: "rgba(56,189,248,0.9)",
    cursorColor: "#f59e0b",
    barWidth: 3,
    barGap: 2,
    barRadius: 3,
    height: 80,
    normalize: true,
    backend: "WebAudio",
  });

  wavesurfer.on("ready", () => {
    $("waveDuration").textContent = secToTime(wavesurfer.getDuration());
    $("waveCurrent").textContent  = "0:00";
  });

  wavesurfer.on("audioprocess", () => {
    $("waveCurrent").textContent = secToTime(wavesurfer.getCurrentTime());
  });

  wavesurfer.on("finish", () => {
    $("iconPlay").style.display  = "block";
    $("iconPause").style.display = "none";
  });

  wavesurfer.load(url);
}

function togglePlay() {
  if (!wavesurfer) return;
  wavesurfer.playPause();
  const playing = wavesurfer.isPlaying();
  $("iconPlay").style.display  = playing ? "none"  : "block";
  $("iconPause").style.display = playing ? "block" : "none";
}

function setVolume(val) {
  wavesurfer?.setVolume(parseFloat(val));
}

function secToTime(s) {
  if (!s || isNaN(s)) return "0:00";
  const m   = Math.floor(s / 60);
  const sec = Math.floor(s % 60).toString().padStart(2,"0");
  return `${m}:${sec}`;
}

// ─────────────────────────────────────────────────────────────
//  TRANSCRIBE
// ─────────────────────────────────────────────────────────────
async function runTranscribe() {
  if (!currentAudio) return;

  setLoading(true);
  $("resultPanel").style.display = "none";

  try {
    const formData = new FormData();
    formData.append("audio", currentAudio.blob, currentAudio.name);

    const startTime = Date.now();
    const res = await fetch(`${API_URL}/transcribe`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);

    if (!res.ok || !data.success) {
      showToast(data.error || "មានបញ្ហាក្នុងការបំលែង", "error");
      return;
    }

    // Display results
    $("transcriptText").textContent = data.text || "(គ្មានអត្ថបទ)";
    $("inverseText").textContent    = data.inverse_text || "(គ្មានអត្ថបទ)";
    $("resultMeta").textContent     = `⏱ ${data.total_time_ms ? (data.total_time_ms/1000).toFixed(2) : elapsed}s  ·  Device: ${data.device || "cpu"}  ·  ID: ${data.request_id || "—"}`;

    $("resultPanel").style.display = "block";
    $("resultPanel").scrollIntoView({ behavior: "smooth", block: "start" });
    showToast("បំលែងជោគជ័យ! ✓", "success");

  } catch (err) {
    console.error(err);
    showToast("មិនអាចភ្ជាប់ Server បាន", "error");
  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  $("btnTranscribe").disabled = on;
  $("btnLabel").style.display   = on ? "none"  : "flex";
  $("btnLoading").style.display = on ? "flex"  : "none";
}

// ─────────────────────────────────────────────────────────────
//  COPY / DOWNLOAD
// ─────────────────────────────────────────────────────────────
async function copyText(id) {
  const text = $(id)?.textContent?.trim();
  if (!text || text === "—") return;
  try {
    await navigator.clipboard.writeText(text);
    showToast("បានចម្លងអត្ថបទ ✓", "success");
  } catch {
    showToast("មិនអាចចម្លងបាន", "error");
  }
}

function downloadTxt(id, filename) {
  const text = $(id)?.textContent?.trim();
  if (!text || text === "—") return;
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast("កំពុង Download... ✓", "success");
}

// ─────────────────────────────────────────────────────────────
//  HELPERS
// ─────────────────────────────────────────────────────────────
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes/1024).toFixed(1)} KB`;
  return `${(bytes/1024/1024).toFixed(2)} MB`;
}

let toastTimer = null;
function showToast(msg, type = "") {
  const el = $("toast");
  el.textContent = msg;
  el.className   = `toast${type ? " " + type : ""} show`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = "toast"; }, 2800);
}
