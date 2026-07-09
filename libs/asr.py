from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

# reduce noisy TensorFlow logs if transformers checks TensorFlow
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import importlib.util

import numpy as np
import sentencepiece as spm
import torch
import torchaudio
from huggingface_hub import hf_hub_download, snapshot_download
from pydub import AudioSegment
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration


# ---------------------------------------------------------------------
# Hugging Face private repo config
# ---------------------------------------------------------------------
INFINITY_ASR_MODEL = "SoyVitou/infinity-khmer-asr"

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN not found. please set your Hugging Face token first:\n"
        "export HF_TOKEN='your_token_here'"
    )


def import_private_module(repo_id: str, filename: str, token: str):
    """
    optional helper if later your private repo contains a custom .py file.
    example:
        mod = import_private_module(INFINITY_ASR_MODEL, 'custom_model.py', HF_TOKEN)
    """
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
    )

    spec = importlib.util.spec_from_file_location("private_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import private module from: {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class KhmerWhisperTokenizer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.sp_model = spm.SentencePieceProcessor()

        ok = self.sp_model.load(model_path)
        if not ok:
            raise FileNotFoundError(f"failed to load tokenizer model: {model_path}")

        self.vocab_size = self.sp_model.vocab_size()

        self.special_tokens = {
            "<|startoftranscript|>": self._required_piece_id(
                "<|startoftranscript|>"
            ),
            "<|endoftranscript|>": self._required_piece_id(
                "<|endoftranscript|>"
            ),
        }

    def _required_piece_id(self, token: str) -> int:
        token_id = self.sp_model.piece_to_id(token)

        if token_id < 0 or self.sp_model.id_to_piece(token_id) != token:
            raise ValueError(
                f"required special token {token!r} is missing in tokenizer model: "
                f"{self.model_path}"
            )

        return int(token_id)

    def encode(self, text: str) -> List[int]:
        return self.sp_model.encode(text)

    def decode(self, tokens: List[int]) -> str:
        cleaned = []

        for token in tokens:
            token = int(token)

            if token < 0 or token >= self.vocab_size:
                continue

            piece = self.sp_model.id_to_piece(token)

            if piece in {"<|startoftranscript|>", "<|endoftranscript|>"}:
                continue

            if piece.startswith("<|") and piece.endswith("|>"):
                continue

            cleaned.append(token)

        if not cleaned:
            return ""

        return self.sp_model.decode(cleaned).strip()

    def batch_decode(
        self,
        token_sequences,
        skip_special_tokens: bool = False,
    ) -> List[str]:
        results = []

        for tokens in token_sequences:
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()

            results.append(self.decode(tokens))

        return results

    def get_decoder_prompt_ids(self, language=None, task=None):
        return [[0, self.special_tokens["<|startoftranscript|>"]]]


class WhisperASR:
    def __init__(
        self,
        repo_id: str = INFINITY_ASR_MODEL,
        hf_token: str = HF_TOKEN,
        device: Optional[str] = None,
    ):
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.device_name = device or os.getenv("ASR_DEVICE", "").strip() or None

        self.local_repo_dir: Optional[Path] = None
        self.model: Optional[WhisperForConditionalGeneration] = None
        self.tokenizer: Optional[KhmerWhisperTokenizer] = None
        self.feature_extractor: Optional[WhisperFeatureExtractor] = None
        self.device: Optional[torch.device] = None

    def _download_repo(self) -> Path:
        """
        download required files from private Hugging Face repo.

        expected repo files:
            - config.json
            - generation_config.json
            - model.safetensors
            - khmer_bpe_base.model
        """
        print(f"[INFO] downloading/loading private HF repo: {self.repo_id}")

        local_dir = snapshot_download(
            repo_id=self.repo_id,
            token=self.hf_token,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "khmer_bpe_base.model",
                "preprocessor_config.json",
            ],
        )

        return Path(local_dir)

    def _select_device(self) -> torch.device:
        if self.device_name:
            return torch.device(self.device_name)

        if torch.cuda.is_available():
            return torch.device("cuda:0")

        return torch.device("cpu")

    def is_loaded(self) -> bool:
        return (
            self.model is not None
            and self.tokenizer is not None
            and self.feature_extractor is not None
            and self.device is not None
        )

    def load_model(self) -> bool:
        if self.is_loaded():
            print("[INFO] model already loaded")
            return True

        self.local_repo_dir = self._download_repo()

        tokenizer_path = self.local_repo_dir / "khmer_bpe_base.model"
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"khmer_bpe_base.model not found in downloaded repo: "
                f"{self.local_repo_dir}"
            )

        self.device = self._select_device()

        print(f"[INFO] loading model on device: {self.device}")
        print(f"[INFO] local repo dir: {self.local_repo_dir}")
        print(f"[INFO] tokenizer path: {tokenizer_path}")

        self.tokenizer = KhmerWhisperTokenizer(str(tokenizer_path))
        print(f"[INFO] tokenizer vocab size: {self.tokenizer.vocab_size}")

        # your repo list does not show preprocessor_config.json.
        # if it exists, load it. otherwise use default Whisper feature extractor.
        try:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                str(self.local_repo_dir),
                local_files_only=True,
            )
            print("[INFO] loaded feature extractor from private repo")
        except Exception as error:
            print(f"[WARN] feature extractor load failed: {error}")
            print("[WARN] using default WhisperFeatureExtractor()")

            self.feature_extractor = WhisperFeatureExtractor(
                feature_size=80,
                sampling_rate=16000,
                hop_length=160,
                chunk_length=30,
                n_fft=400,
                padding_value=0.0,
                return_attention_mask=False,
            )

        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(self.local_repo_dir),
            local_files_only=True,
            use_safetensors=True,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        start_id = self.tokenizer.special_tokens["<|startoftranscript|>"]
        eos_id = self.tokenizer.special_tokens["<|endoftranscript|>"]

        # align model config
        self.model.config.decoder_start_token_id = start_id
        self.model.config.bos_token_id = start_id
        self.model.config.eos_token_id = eos_id
        self.model.config.pad_token_id = eos_id
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = None
        self.model.config.begin_suppress_tokens = None
        self.model.config.use_cache = False
        self.model.config.language = None
        self.model.config.task = None

        if hasattr(self.model.config, "lang_to_id"):
            self.model.config.lang_to_id = {}

        if hasattr(self.model.config, "task_to_id"):
            try:
                delattr(self.model.config, "task_to_id")
            except Exception:
                pass

        # align generation config
        self.model.generation_config.decoder_start_token_id = start_id
        self.model.generation_config.bos_token_id = start_id
        self.model.generation_config.eos_token_id = eos_id
        self.model.generation_config.pad_token_id = eos_id
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.suppress_tokens = None
        self.model.generation_config.begin_suppress_tokens = None
        self.model.generation_config.do_sample = False
        self.model.generation_config.early_stopping = True
        self.model.generation_config.return_timestamps = False
        self.model.generation_config.language = None
        self.model.generation_config.task = None

        if hasattr(self.model.generation_config, "lang_to_id"):
            self.model.generation_config.lang_to_id = {}

        if hasattr(self.model.generation_config, "task_to_id"):
            try:
                delattr(self.model.generation_config, "task_to_id")
            except Exception:
                pass

        if hasattr(self.model.generation_config, "is_multilingual"):
            self.model.generation_config.is_multilingual = False

        print("[INFO] model initialized successfully")
        return True

    def unload_model(self) -> bool:
        try:
            if not self.is_loaded():
                print("[INFO] model already unloaded")
                return True

            print("[INFO] unloading model...")

            try:
                if self.model is not None:
                    self.model.to("cpu")
            except Exception as error:
                print(f"[WARN] could not move model to CPU before unload: {error}")

            self.model = None
            self.tokenizer = None
            self.feature_extractor = None
            self.device = None

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            print("[INFO] model unloaded successfully")
            return True

        except Exception as error:
            print(f"[ERROR] failed to unload model: {error}")

            self.model = None
            self.tokenizer = None
            self.feature_extractor = None
            self.device = None

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return False

    def _load_audio_with_fallback(
        self,
        audio_file_path: str,
        target_sr: int = 16000,
    ):
        """
        load audio using torchaudio first.
        fallback to pydub if codec/backend is unsupported.
        """
        try:
            waveform, sample_rate = torchaudio.load(audio_file_path)
            return waveform, sample_rate

        except Exception as error:
            print(f"[WARN] torchaudio.load failed: {error}")
            print("[WARN] falling back to pydub...")

        audio = AudioSegment.from_file(audio_file_path)

        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)

        samples = np.array(audio.get_array_of_samples())

        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).mean(axis=1)

        max_val = float(1 << (8 * audio.sample_width - 1))
        waveform = torch.tensor(samples.astype(np.float32) / max_val).unsqueeze(0)
        sample_rate = audio.frame_rate

        return waveform, sample_rate

    def predict(self, audio_file_path: str) -> Dict:
        try:
            if not self.is_loaded():
                self.load_model()

            assert self.model is not None
            assert self.tokenizer is not None
            assert self.feature_extractor is not None
            assert self.device is not None

            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"audio file not found: {audio_file_path}")

            start_time = time.perf_counter()

            waveform, sample_rate = self._load_audio_with_fallback(
                audio_file_path,
                target_sr=16000,
            )

            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=16000,
                )(waveform)
                sample_rate = 16000

            # convert stereo to mono
            if waveform.ndim == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if waveform.ndim == 2:
                waveform = waveform.squeeze(0)

            # match Whisper training max length: 30 seconds
            max_samples = 30 * 16000

            if waveform.shape[0] > max_samples:
                waveform = waveform[:max_samples]
            elif waveform.shape[0] < max_samples:
                waveform = torch.nn.functional.pad(
                    waveform,
                    (0, max_samples - waveform.shape[0]),
                )

            inputs = self.feature_extractor(
                waveform.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
            )

            input_features = inputs.input_features.to(self.device)

            decoder_input_ids = torch.tensor(
                [[self.tokenizer.special_tokens["<|startoftranscript|>"]]],
                device=self.device,
            )

            eos_id = self.tokenizer.special_tokens["<|endoftranscript|>"]

            with torch.inference_mode():
                predicted_ids = self.model.generate(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    max_new_tokens=128,
                    num_beams=3,
                    do_sample=False,
                    eos_token_id=eos_id,
                    pad_token_id=eos_id,
                    forced_decoder_ids=None,
                    suppress_tokens=None,
                    begin_suppress_tokens=None,
                    language=None,
                    task=None,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    return_timestamps=False,
                )

            text = self.tokenizer.decode(predicted_ids[0].tolist())

            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            return {
                "success": True,
                "transcription": text,
                "processing_time_ms": round(response_time_ms, 2),
                "device": str(self.device),
                "audio_path": audio_file_path,
                "repo_id": self.repo_id,
                "local_repo_dir": str(self.local_repo_dir),
                "raw_ids": predicted_ids[0].tolist(),
            }

        except Exception as error:
            return {
                "success": False,
                "error": str(error),
                "transcription": "",
                "processing_time_ms": 0,
                "device": str(self.device) if self.device else None,
                "audio_path": audio_file_path,
                "repo_id": self.repo_id,
                "local_repo_dir": str(self.local_repo_dir)
                if self.local_repo_dir
                else None,
            }


# ---------------------------------------------------------------------
# singleton helpers for app.py
# app.py can simply do:
#   from libs.asr import predict
# ---------------------------------------------------------------------
_asr = WhisperASR()


def load_model() -> bool:
    return _asr.load_model()


def unload_model() -> bool:
    return _asr.unload_model()


def predict(audio_file_path: str) -> Dict:
    return _asr.predict(audio_file_path)


def is_model_loaded() -> bool:
    return _asr.is_loaded()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    args = parser.parse_args()

    result = predict(args.audio)

    print("\n===== ASR RESULT =====")
    print("success:", result.get("success"))

    if result.get("success"):
        print("transcription:", result.get("transcription"))
        print("processing_time_ms:", result.get("processing_time_ms"))
        print("device:", result.get("device"))
    else:
        print("error:", result.get("error"))