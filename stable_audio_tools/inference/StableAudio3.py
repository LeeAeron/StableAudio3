"""Stable Audio 3 Portable
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_COMPILE_BACKEND"] = "none"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Root directory of the launcher (bat file location = working directory at startup)
# The script itself may live deep inside venv subfolders, so we use cwd, not __file__.
LAUNCHER_DIR = os.getcwd()

import json
import sys
import time
import gc
import random
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from huggingface_hub import snapshot_download
    _HAS_HF_HUB = True
except ImportError:
    _HAS_HF_HUB = False
    print("[WARN] huggingface_hub not installed. Auto-download disabled.")

import gradio as gr
import numpy as np
import soundfile as sf

import subprocess
import shutil

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.reset()

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.attention.flex_attention")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.nn.utils.weight_norm"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*expandable_segments not supported.*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*local_dir_use_symlinks.*"
)

# CONFIG MANAGER — saves/loads all UI settings to JSON near the launcher

class ConfigManager:
    """Manages saving and loading UI settings to a JSON file.

    The config file is created in the same folder as the running script (bat file),
    i.e. next to app.py.
    """

    def __init__(self):
        # Config file path: next to the bat launcher (LAUNCHER_DIR = cwd at startup)
        self.config_path = os.path.join(
            LAUNCHER_DIR,
            "stable_audio_3_settings.json"
        )
        self.defaults = self._build_defaults()
        self.current = {}  # Initialize empty first
        self.current = self.load()

    def _build_defaults(self):
        """All default values for all UI components."""
        return {
            # === Simple tab ===
            "simple_variant": "medium",
            "simple_steps": 8,
            "simple_cfg_scale": 1.0,
            "simple_sampler_type": "pingpong",
            "simple_seed": 0,
            "simple_duration": 349,
            "simple_prompt": "",
            # Audio Export (Simple)
            "simple_fmt": "WAV32F",
            "simple_target_sr": "48000",
            "simple_bitrate": 320,
            "simple_ogg_quality": 5,
            "simple_normalize": False,
            "simple_normalize_level": -15,

            # === Advanced tab ===
            "adv_variant": "medium",
            "adv_steps": 8,
            "adv_cfg": 1.0,
            "adv_sampler": "pingpong",
            "adv_seed": -1,
            "adv_sigma_max": 1.0,
            "adv_apg": 1.0,
            "adv_dur_padding": 6.0,
            "adv_seconds_total": 349,
            "adv_prompt": "",
            "adv_negative": "",
            # Init audio
            "adv_init_noise": 0.9,
            # Inpainting
            "adv_mask_start": 0.0,
            "adv_mask_end": 0.0,
            # Output params
            "adv_preview_every": 0,
            "adv_cut_to_total": True,
            # Audio Export (Advanced)
            "adv_fmt": "WAV32F",
            "adv_target_sr": "48000",
            "adv_bitrate": 320,
            "adv_ogg_quality": 5,
            "adv_normalize": False,
            "adv_normalize_level": -15,
        }

    def load(self):
        """Load config from file or return defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                # Merge: saved + defaults for missing keys
                merged = dict(self.defaults)
                merged.update(saved)
                print(f"[INFO] Config loaded from: {self.config_path}")
                return merged
            except Exception as e:
                print(f"[WARN] Failed to load config: {e}. Using defaults.")
                return dict(self.defaults)
        else:
            print(f"[INFO] Config not found, creating default: {self.config_path}")
            # Write defaults directly without calling save() to avoid self.current access
            try:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(self.defaults, f, indent=2, ensure_ascii=False)
                print(f"[INFO] Default config created: {self.config_path}")
            except Exception as e:
                print(f"[WARN] Failed to create default config: {e}")
            return dict(self.defaults)

    def save(self, settings=None):
        """Save current settings to JSON file."""
        if settings is not None:
            self.current.update(settings)
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.current, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Config saved to: {self.config_path}")
        except Exception as e:
            print(f"[WARN] Failed to save config: {e}")

    def get(self, key, default=None):
        return self.current.get(key, default)

    def set(self, key, value):
        self.current[key] = value
        self.save()


# Global config manager
_cfg = ConfigManager()


# GPU Auto-Detection & Optimal Config
def get_optimal_config():
    """determine the optimal configuration for specific hardware."""
    config = {
        "device": "cpu",
        "dtype": torch.float32,
        "amp_dtype": torch.float32,
        "has_tensor_cores": False,
        "supports_bf16": False,
        "compute_capability": (0, 0),
        "gpu_name": "CPU",
        "vram_gb": 0.0,
        "offload_to_cpu": False,
        "use_autocast": False,
        "fallback_to_fp32": False,
    }

    if not torch.cuda.is_available():
        print("[INFO] CUDA not available, using CPU")
        return config

    gpu_name = torch.cuda.get_device_name(0)
    gpu_name_lower = gpu_name.lower()
    cc = torch.cuda.get_device_capability(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)

    # Tensor Cores: SM >= 7.0 + not GTX 16xx
    has_tensor_cores = cc[0] >= 7 and not any(x in gpu_name_lower for x in [
        "gtx 16", "gtx 1650", "gtx 1660", "gtx 1670"
    ])

    # BF16: Ampere+ (SM >= 8.0)
    supports_bf16 = cc[0] >= 8

    # Offload for gpu <8GB VRAM
    offload_to_cpu = vram_gb < 8

    # Choosing a dtype by priority
    if supports_bf16:
        dtype = torch.bfloat16
        amp_dtype = torch.bfloat16
    elif has_tensor_cores:
        dtype = torch.float16
        amp_dtype = torch.float16
    elif cc[0] >= 7:  # Turing without Tensor Cores (GTX 16xx)
        # save in FP16 (less VRAM), processing in FP32
        dtype = torch.float16
        amp_dtype = torch.float32
        config["fallback_to_fp32"] = True
    else:
        dtype = torch.float32
        amp_dtype = torch.float32

    use_autocast = dtype != torch.float32

    config.update({
        "device": "cuda",
        "dtype": dtype,
        "amp_dtype": amp_dtype,
        "has_tensor_cores": has_tensor_cores,
        "supports_bf16": supports_bf16,
        "compute_capability": cc,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "offload_to_cpu": offload_to_cpu,
        "use_autocast": use_autocast,
    })

    return config


OPTIMAL = get_optimal_config()

# CUDA Optimizations — auto-detected based on GPU compute capability
# These are applied AFTER get_optimal_config() so we know the hardware capabilities

# TF32 (TensorFloat-32): only beneficial on Ampere+ (SM >= 8.0)
# On older GPUs these flags are ignored, but we gate them explicitly for clarity.
if OPTIMAL["device"] == "cuda":
    cc_major = OPTIMAL["compute_capability"][0]

    # TF32: Ampere+ only (SM >= 8.0)
    supports_tf32 = cc_major >= 8
    torch.backends.cuda.matmul.allow_tf32 = supports_tf32
    torch.backends.cudnn.allow_tf32 = supports_tf32
    if supports_tf32:
        print("[INFO] TF32 enabled (Ampere+ GPU)")
    else:
        print(f"[INFO] TF32 disabled (CC {cc_major}.x < 8.0)")

    # cuDNN tuning
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Turing (SM 7.x, e.g., RTX 20xx, GTX 16xx) — FP32 accumulation for FP16 matmul
    if cc_major == 7:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        print("[INFO] FP16 reduced-precision reduction disabled (Turing GPU)")

    # PyTorch 2.0+ SDPA optimizations — auto-detect flash attention support
    try:
        # Flash Attention requires SM80+ (Ampere+). Check both compute capability
        # and whether PyTorch was actually built with flash attention support.
        flash_available = torch.backends.cuda.is_flash_attention_available()
        supports_flash = cc_major >= 8 and flash_available

        # Memory-efficient attention works on SM50+ (Maxwell+), so always safe to enable
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Math fallback — always safe
        torch.backends.cuda.enable_math_sdp(True)

        # Flash attention — only on supported hardware
        torch.backends.cuda.enable_flash_sdp(supports_flash)

        if supports_flash:
            print("[INFO] Flash Attention (SDPA) enabled")
        else:
            reason = []
            if cc_major < 8:
                reason.append(f"CC {cc_major}.x < 8.0")
            if not flash_available:
                reason.append("PyTorch build lacks flash attention")
            print(f"[INFO] Flash Attention (SDPA) disabled ({', '.join(reason)})")
            print("[INFO] Falling back to memory-efficient + math SDPA backends")

    except AttributeError:
        # PyTorch < 2.0 doesn't have these APIs
        print("[INFO] PyTorch < 2.0 — SDPA backend controls unavailable")
else:
    print("[INFO] CPU mode — CUDA optimizations skipped")
# Imports after config
import torchaudio
import torchaudio.transforms as T
from einops import rearrange
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

try:
    from stable_audio_tools import create_model_from_config
    from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint
except ImportError:
    print("[ERROR] stable-audio-tools not installed! Run StableAudio3.bat -> Install.")
    sys.exit(1)

# Device

device = torch.device(OPTIMAL["device"])
print(f"[INFO] Using device: {device}")
if device.type == "cuda":
    print(f"[INFO] GPU: {OPTIMAL['gpu_name']}")
    print(f"[INFO] Compute Capability: {OPTIMAL['compute_capability']}")
    print(f"[INFO] VRAM: {OPTIMAL['vram_gb']:.1f} GB")
    print(f"[INFO] Optimal dtype: {OPTIMAL['dtype']}")
    print(f"[INFO] AMP dtype: {OPTIMAL['amp_dtype']}")
    print(f"[INFO] Tensor Cores: {OPTIMAL['has_tensor_cores']}")
    print(f"[INFO] CPU Offload: {OPTIMAL['offload_to_cpu']}")


# Paths
script_dir = LAUNCHER_DIR  # bat launcher directory (cwd at startup)
models_dir = os.path.join(LAUNCHER_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

print(f"[INFO] Models directory: {models_dir}")

# HuggingFace repo mapping: local_folder_name -> repo_id
HF_REPOS = {
    "stable-audio-3-medium": "LeeAeron/stable-audio-3-medium",
    "stable-audio-3-small-music": "LeeAeron/stable-audio-3-small-music",
    "stable-audio-3-small-sfx": "LeeAeron/stable-audio-3-small-sfx",
    "t5gemma-b-b-ul2": "LeeAeron/t5gemma-b-b-ul2",
}


def ensure_model_downloaded(local_dir: str, repo_id: str) -> None:
    """Checks for the local availability of the model; if not, downloads it from HF."""
    config_path = os.path.join(local_dir, "model_config.json")
    if os.path.exists(config_path):
        print(f"[INFO] Model found locally: {local_dir}")
        return

    if not _HAS_HF_HUB:
        raise RuntimeError(
            f"Model not found locally: {local_dir}\n"
            f"huggingface_hub is not installed. Cannot auto-download from {repo_id}.\n"
            f"Install: pip install huggingface_hub"
        )

    print(f"[INFO] Model not found locally: {local_dir}")
    print(f"[INFO] Downloading from HuggingFace: {repo_id} -> {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"[INFO] Download complete: {local_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to download {repo_id}: {e}")
        raise gr.Error(f"Failed to download model {repo_id}: {str(e)}")


def _check_and_download_t5gemma():
    """Checking and downloading the T5Gemma model at startup."""
    t5_dir = os.path.join(models_dir, "t5gemma-b-b-ul2")
    t5_repo = HF_REPOS.get("t5gemma-b-b-ul2")
    if t5_repo:
        ensure_model_downloaded(t5_dir, t5_repo)



@dataclass
class Variant:
    key: str
    path: str
    label: str
    default_duration: int
    placeholder: str


VARIANTS: list[Variant] = [
    Variant(
        key="medium",
        path=os.path.join(models_dir, "stable-audio-3-medium"),
        label="Medium — general audio (largest)",
        default_duration=349,
        placeholder="A dream-like Synthpop instrumental that would accompany a dream-sequence in a surrealist movie 120 BPM",
    ),
    Variant(
        key="small-music",
        path=os.path.join(models_dir, "stable-audio-3-small-music"),
        label="Small Music — 0.6B, music-focused",
        default_duration=120,
        placeholder="Cinematic neo-soul groove with electric piano, brushed drums, walking upright bass, smoky vibe 92 BPM",
    ),
    Variant(
        key="small-sfx",
        path=os.path.join(models_dir, "stable-audio-3-small-sfx"),
        label="Small SFX — 0.6B, sound effects",
        default_duration=7,
        placeholder="Chugging train coming into station with horn",
    ),
]



@dataclass
class VariantInfo:
    sample_rate: int
    sample_size: int
    max_seconds: int
    config: dict


VARIANT_INFO: dict[str, VariantInfo] = {}

for v in VARIANTS:
    config_path = os.path.join(v.path, "model_config.json")
    if not os.path.exists(config_path):
        # try downloading a model from HuggingFace.
        folder_name = os.path.basename(v.path)
        repo_id = HF_REPOS.get(folder_name)
        if repo_id and _HAS_HF_HUB:
            try:
                ensure_model_downloaded(v.path, repo_id)
            except Exception as e:
                print(f"[WARN] Auto-download failed for {v.key}: {e}")
        else:
            print(f"[WARN] Config NOT found for {v.key}: {config_path}")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        sr = int(cfg.get("sample_rate", 48000))
        ss = int(cfg.get("sample_size", sr * 60))
        VARIANT_INFO[v.key] = VariantInfo(
            sample_rate=sr,
            sample_size=ss,
            max_seconds=ss // sr,
            config=cfg,
        )
        print(f"[INFO] Config loaded for {v.key}: sr={sr}, max={ss//sr}s")
    else:
        print(f"[WARN] Config still NOT found for {v.key}: {config_path}")



# Checking and downloading the T5Gemma model at startup
_check_and_download_t5gemma()


@dataclass
class LoadedVariant:
    variant: Variant
    model: object
    sample_rate: int
    sample_size: int
    max_seconds: int


# store only one active model
_current_loaded: Optional[LoadedVariant] = None
_current_key: Optional[str] = None


SAMPLERS = ["pingpong", "euler", "rk4", "dpmpp"]


# Audio Export Constants
LOSSY_FORMATS = {"MP3", "OGG", "OPUS", "AAC", "WMA"}
LOSSLESS_FORMATS = {"WAV32F", "WAV32", "WAV24", "WAV", "FLAC", "ALAC", "AIFF"}
ALL_FORMATS = ["WAV32F", "WAV32", "WAV24", "WAV", "FLAC", "ALAC", "AIFF", "MP3", "OGG", "OPUS", "AAC", "WMA"]
SAMPLE_RATES = [44100, 48000, 96000, 192000]
BITRATE_OPTIONS = [64, 96, 128, 160, 192, 224, 256, 320]
OGG_QUALITY_OPTIONS = list(range(-1, 11))
NORMALIZATION_LEVELS = list(range(-20, 11))

def unload_current_model():
    """Unload the current model from memory."""
    global _current_loaded, _current_key
    if _current_loaded is not None:
        print(f"[INFO] Unloading model {_current_key}...")
        del _current_loaded.model
        del _current_loaded
        _current_loaded = None
        _current_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[INFO] Model unloaded, memory cleared.")


def load_local_model(model_dir: str):
    """Load the model from the local folder."""
    config_path = os.path.join(model_dir, "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"model_config.json not found in {model_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = create_model_from_config(config)

    safetensors_path = os.path.join(model_dir, "model.safetensors")
    ckpt_path = os.path.join(model_dir, "model.ckpt")

    if os.path.exists(safetensors_path):
        print(f"[INFO] Loading weights from {safetensors_path}")
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=False)
    elif os.path.exists(ckpt_path):
        print(f"[INFO] Loading weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(
            f"No model weights found in {model_dir}. "
            f"Expected model.safetensors or model.ckpt"
        )

    return model, config


def move_model_to_device(model, target_device, dtype):
    """Moving the model to a device with memory control."""
    if target_device.type == "cpu":
        return model.to("cpu").to(dtype)

    print(f"[INFO] Moving model to {target_device} with dtype={dtype}...")

    # First, on the CPU in the desired dtype
    model = model.to("cpu").to(dtype)

    # Clear the cache before loading onto the GPU
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # We move the GPU modules by modules (to control peak VRAM consumption)
    def move_module(module, device):
        try:
            module.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                module.to(device)  # Retry
            else:
                raise

    # move the main components one by one
    if hasattr(model, "model"):
        move_module(model.model, target_device)
    if hasattr(model, "conditioner"):
        move_module(model.conditioner, target_device)
    if hasattr(model, "pretransform") and model.pretransform is not None:
        move_module(model.pretransform, target_device)
    if hasattr(model, "diffusion"):
        move_module(model.diffusion, target_device)

    # Everything else
    move_module(model, target_device)

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    print(f"[INFO] VRAM after load: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

    return model


def load_variant(variant_key: str) -> LoadedVariant:
    """Load a model after unloading the previous one.."""
    global _current_loaded, _current_key

    # If the same model is already loaded, return it.
    if _current_key == variant_key and _current_loaded is not None:
        return _current_loaded

    # If another model is loaded, unload it.
    if _current_loaded is not None:
        unload_current_model()

    v = next((x for x in VARIANTS if x.key == variant_key), None)
    if not v:
        raise gr.Error(f"Unknown variant {variant_key!r}")

    # Automatically download a model from HuggingFace if it is not available locally
    folder_name = os.path.basename(v.path)
    repo_id = HF_REPOS.get(folder_name)
    if repo_id:
        ensure_model_downloaded(v.path, repo_id)
    elif not os.path.exists(v.path):
        raise gr.Error(
            f"Model folder not found: {v.path}\n\n"
            f"Please download model files from HuggingFace and place them in:\n"
            f"{v.path}\n\n"
            f"Required files:\n"
            f"  - model_config.json\n"
            f"  - model.safetensors (or model.ckpt)\n"
            f"  - preprocessor_config.json"
        )

    info = VARIANT_INFO.get(variant_key)
    if info is None:
        raise gr.Error(f"model_config.json missing for {variant_key}")

    print(f"[startup] loading weights for {v.key} ...", flush=True)
    print(f"[startup] dtype={OPTIMAL['dtype']}, device={OPTIMAL['device']}")
    t0 = time.time()

    try:
        # Loading onto the CPU
        model, config = load_local_model(v.path)

        # Move to the target device with the optimal dtype
        if OPTIMAL["offload_to_cpu"]:
            # For cards with low VRAM: try loading on the GPU, fallback on the CPU
            print("[INFO] Loading model to CPU first (low VRAM mode)...")
            model = model.to("cpu").to(OPTIMAL["dtype"])

            # trying to move it to the GPU in parts.
            try:
                model = move_model_to_device(model, device, OPTIMAL["dtype"])
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[WARN] OOM during GPU transfer, keeping model on CPU: {e}")
                    model = model.to("cpu")
                else:
                    raise
        else:
            # For cards with sufficient VRAM, full GPU load
            model = move_model_to_device(model, device, OPTIMAL["dtype"])

    except Exception as e:
        print(f"[ERROR] Failed to load {v.path}: {e}")
        raise gr.Error(f"Cannot load model from {v.path}: {str(e)}")

    lv = LoadedVariant(
        variant=v,
        model=model,
        sample_rate=info.sample_rate,
        sample_size=info.sample_size,
        max_seconds=info.max_seconds,
    )
    _current_loaded = lv
    _current_key = variant_key

    print(
        f"[startup] {v.key} ready in {time.time() - t0:.1f}s * "
        f"sr={lv.sample_rate} * max={lv.max_seconds}s",
        flush=True,
    )
    return lv


# Spectrogram
def _power_to_db(spec: np.ndarray, amin: float = 1e-10) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(amin, spec))


def audio_spectrogram_image(
    waveform: torch.Tensor,
    sample_rate: int,
    db_range=(35, 120),
    figsize=(5, 4),
) -> Image.Image:
    """Render a Mel spectrogram (left channel) as a PIL image."""
    import warnings
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    n_fft = 1024
    hop_length = n_fft // 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mel_op = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=None,
            hop_length=hop_length, center=True, pad_mode="reflect", power=2.0,
            norm="slaney", onesided=True, n_mels=128, mel_scale="htk",
        )
        melspec = mel_op(waveform.float())[0]  # left channel
    fig = Figure(figsize=figsize, dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.imshow(_power_to_db(melspec.numpy()), origin="lower", aspect="auto",
              vmin=db_range[0], vmax=db_range[1])
    ax.set_ylabel("mel bins (log freq)")
    ax.set_xlabel("frame")
    ax.set_title("MelSpectrogram")
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba()))



# Audio utils
def _gradio_audio_to_tensor(
    audio_in: Optional[Tuple[int, np.ndarray]],
) -> Optional[Tuple[int, torch.Tensor]]:
    """Convert a numpy value to tuple
    that ``generate_diffusion_cond_inpaint`` expects. Accepts mono or stereo."""
    if audio_in is None:
        return None
    sr, arr = audio_in
    if arr is None or (hasattr(arr, "size") and arr.size == 0):
        return None
    arr = np.asarray(arr)
    if arr.dtype.kind in ("i", "u"):
        max_val = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / max_val
    else:
        arr = arr.astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]                       # (1, N)
    else:
        # gr.Audio returns (N, C); transpose to (C, N)
        arr = arr.T if arr.shape[0] > arr.shape[1] else arr
    return int(sr), torch.from_numpy(arr)



def normalize_audio(waveform: np.ndarray, target_level_db: float) -> np.ndarray:
    """Normalize audio to target RMS level in dB."""
    if waveform.size == 0:
        return waveform
    if waveform.dtype == np.int16:
        waveform_float = waveform.astype(np.float32) / 32768.0
    else:
        waveform_float = waveform.astype(np.float32)
    rms = np.sqrt(np.mean(waveform_float ** 2))
    if rms < 1e-10:
        return waveform
    target_rms = 10 ** (target_level_db / 20.0)
    gain = target_rms / rms
    normalized = np.clip(waveform_float * gain, -1.0, 1.0)
    if waveform.dtype == np.int16:
        return (normalized * 32767).astype(np.int16)
    return normalized


def save_audio_with_ffmpeg(
    waveform: np.ndarray,
    sample_rate: int,
    output_path: str,
    fmt: str = "WAV32F",
    target_sr: int = 48000,
    bitrate: int = 320,
    ogg_quality: int = 5,
    normalize: bool = False,
    normalize_level_db: float = -15.0,
) -> str:
    """Export audio via FFmpeg with format selection, resampling, and normalization."""
    fmt = fmt.lower().replace(".", "")

    # guarantie that sample_rate and target_sr are numbers
    try:
        sample_rate = int(str(sample_rate).replace("Hz", "").strip())
    except Exception:
        sample_rate = 48000
    try:
        target_sr = int(str(target_sr).replace("Hz", "").strip())
    except Exception:
        target_sr = 48000

    # Normalize if requested
    if normalize:
        waveform = normalize_audio(waveform, normalize_level_db)

    # Convert to float32 for processing
    if waveform.dtype == np.int16:
        wav_float = waveform.astype(np.float32) / 32768.0
    else:
        wav_float = waveform.astype(np.float32)
    wav_float = np.clip(wav_float, -1.0, 1.0)

    # save temp WAV of source SR
    tmp_wav = os.path.join(script_dir, "temp_ffmpeg_input.wav")
    os.makedirs(os.path.dirname(tmp_wav) if os.path.dirname(tmp_wav) else ".", exist_ok=True)
    sf.write(tmp_wav, wav_float, sample_rate, subtype="PCM_16")

    try:
        cmd = ["ffmpeg", "-y", "-i", tmp_wav]
        # FFmpeg resample
        cmd.extend(["-ar", str(target_sr), "-resampler", "soxr", "-precision", "28"])

        if fmt == "mp3":
            cmd.extend(["-c:a", "libmp3lame", "-b:a", f"{bitrate}k"])
        elif fmt == "ogg":
            cmd.extend(["-c:a", "libvorbis", "-q:a", str(ogg_quality)])
        elif fmt == "opus":
            cmd.extend(["-c:a", "libopus", "-b:a", f"{bitrate}k", "-vbr", "on"])
        elif fmt == "aac":
            cmd.extend(["-c:a", "aac", "-b:a", f"{bitrate}k"])
        elif fmt == "wma":
            cmd.extend(["-c:a", "wmav2", "-b:a", f"{bitrate}k"])
        elif fmt == "flac":
            cmd.extend(["-c:a", "flac"])
        elif fmt == "alac":
            cmd.extend(["-c:a", "alac"])
        if fmt == "wav24":
            cmd.extend(["-c:a", "pcm_s24le"])
            output_path = output_path.rsplit(".", 1)[0] + ".wav"
        elif fmt == "wav32":
            cmd.extend(["-c:a", "pcm_s32le"])
            output_path = output_path.rsplit(".", 1)[0] + ".wav"
        elif fmt == "wav32f":
            cmd.extend(["-c:a", "pcm_f32le"])
            output_path = output_path.rsplit(".", 1)[0] + ".wav"
        elif fmt == "aiff":
            cmd.extend(["-c:a", "pcm_s16be"])
        else:
            cmd.extend(["-c:a", "pcm_s16le"])


        cmd.append(output_path)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[FFmpeg] Error: {result.stderr}")
            # Fallback to WAV
            output_path = output_path.rsplit(".", 1)[0] + ".wav"
            shutil.copy(tmp_wav, output_path)
        else:
            print(f"[FFmpeg] Saved: {output_path} ({target_sr}Hz, {fmt.upper()})")
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

    return output_path


def _tensor_to_audio(
    output: torch.Tensor,
    sample_rate: int,
    duration_seconds: Optional[int],
    out_dir: Optional[str] = None,
    fmt: str = "WAV32F",
    target_sr: int = 48000,
    bitrate: int = 320,
    ogg_quality: int = 5,
    normalize: bool = False,
    normalize_level_db: float = -15.0,
) -> Tuple[str, torch.Tensor]:
    """Pack a (B, C, N) generation tensor to audio, optionally cut to duration,
    export via FFmpeg with format/resample/normalization, and return (path, int16-tensor).
    """
    # Rearrange and normalize to int16
    output = rearrange(output, "b d n -> d (b n)")
    output = (
        output.to(torch.float32)
        .div(torch.max(torch.abs(output)).clamp(min=1e-9))
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .cpu()
    )
    if duration_seconds is not None:
        output = output[:, : int(duration_seconds) * sample_rate]

    # Output directory
    if out_dir is None:
        out_dir = os.path.join(LAUNCHER_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    now = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"sa3_{now}"

    # Convert to numpy for FFmpeg
    audio_np = output.numpy().T  # (N, C) or (N,)
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]  # (N, 1) for mono

    # Export via FFmpeg with all options
    ext = fmt.lower().replace(".", "")
    filename = f"{base_name}.{ext}"
    out_path = os.path.join(out_dir, filename)

    out_path = save_audio_with_ffmpeg(
        waveform=audio_np,
        sample_rate=sample_rate,
        output_path=out_path,
        fmt=fmt,
        target_sr=target_sr,
        bitrate=bitrate,
        ogg_quality=ogg_quality,
        normalize=normalize,
        normalize_level_db=normalize_level_db,
    )

    return out_path, output


# Inference
def _run_inference(
    variant_key: str,
    prompt: str,
    negative_prompt: str = "",
    duration: int = 60,
    steps: int = 8,
    cfg_scale: float = 1.0,
    sampler_type: str = "pingpong",
    seed: int = 0,
    sigma_max: float = 1.0,
    apg_scale: float = 1.0,
    duration_padding_sec: float = 6.0,
    cut_to_seconds_total: bool = True,
    init_audio: Optional[Tuple[int, np.ndarray]] = None,
    init_noise_level: float = 0.9,
    inpaint_audio: Optional[Tuple[int, np.ndarray]] = None,
    mask_start_sec: float = 0.0,
    mask_end_sec: float = 0.0,
    preview_every: int = 0,
    return_spectrogram: bool = True,
    # Audio Export Parameters
    fmt: str = "WAV32F",
    target_sr: int = 48000,
    bitrate: int = 320,
    ogg_quality: int = 5,
    normalize: bool = False,
    normalize_level_db: float = -15.0,
    progress: gr.Progress = gr.Progress(),
):
    """Full-featured generation. Returns (audio_path, [spectrogram_img, *previews])
    when ``return_spectrogram`` is True, else just ``audio_path``."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    lv = load_variant(variant_key)
    duration = max(1, min(int(duration), lv.max_seconds))

    progress(0.05, desc=f"[{variant_key}] preparing conditioning")
    conditioning = [{"prompt": prompt, "seconds_total": int(duration)}]
    negative_conditioning = None
    neg = (negative_prompt or "").strip()
    if neg:
        negative_conditioning = [{"prompt": neg, "seconds_total": int(duration)}]

    # The pretransform encoder is fp16 (cast the whole model at startup),
    # but prepare_audio's torchaudio Resample uses an fp32 kernel. Pre-resample
    # in fp32 here so prepare_audio's resample is a no-op, then cast to the
    # model dtype so the encoder doesn't see a dtype mismatch.
    model_dtype = next(lv.model.parameters()).dtype

    def _prep(tup):
        if tup is None:
            return None
        sr, t = tup
        t = t.float()
        if sr != lv.sample_rate:
            t = torchaudio.functional.resample(t, sr, lv.sample_rate)
        return lv.sample_rate, t.to(model_dtype)

    init_audio_t = _prep(_gradio_audio_to_tensor(init_audio))
    inpaint_audio_t = _prep(_gradio_audio_to_tensor(inpaint_audio))

    # Inpaint mask: only enable if mask_end > mask_start AND we have either
    # inpaint_audio or init_audio (otherwise the mask wraps zero content).
    mask_start = max(0.0, float(mask_start_sec))
    mask_end = min(float(duration), float(mask_end_sec))
    use_mask = (
        inpaint_audio_t is not None
        and mask_end > mask_start
    )

    # Seed handling: fix for Windows + NumPy int32 overflow
    if seed is not None and int(seed) > 0:
        seed_val = int(seed)
    else:
        seed_val = random.randint(0, 2**31 - 1)   # 2147483647 — max for int32
        print(f"[INFO] Generated random seed: {seed_val}")

    preview_images: list = []
    callback = None
    if preview_every and int(preview_every) > 0:
        every = int(preview_every)

        def _cb(info):
            i = info["i"]
            if i % every != 0:
                return
            denoised = info["denoised"]
            try:
                if lv.model.pretransform is not None:
                    denoised = lv.model.pretransform.decode(denoised)
                d = rearrange(denoised, "b d n -> d (b n)")
                d = d.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                img = audio_spectrogram_image(d, sample_rate=lv.sample_rate)
                preview_images.append((img, f"Step {i + 1}"))
            except Exception as e:
                print(f"[preview] skipped step {i}: {e}", flush=True)
        callback = _cb

    gen_kwargs: dict = dict(
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        sample_size=lv.sample_size,
        sampler_type=sampler_type,
        seed=seed_val,
        device=str(device),
        sigma_max=float(sigma_max),
        apg_scale=float(apg_scale),
        duration_padding_sec=float(duration_padding_sec),
    )
    if init_audio_t is not None:
        gen_kwargs["init_audio"] = init_audio_t
        gen_kwargs["init_noise_level"] = float(init_noise_level)
    if inpaint_audio_t is not None:
        gen_kwargs["inpaint_audio"] = inpaint_audio_t
    if use_mask:
        gen_kwargs["inpaint_mask_start_seconds"] = mask_start
        gen_kwargs["inpaint_mask_end_seconds"] = mask_end
    if callback is not None:
        gen_kwargs["callback"] = callback

    progress(0.25, desc=f"[{variant_key}] sampling {steps} steps with {sampler_type}")
    t0 = time.time()

    # autocast for mixed precision ===
    autocast_enabled = OPTIMAL["use_autocast"]
    autocast_dtype = OPTIMAL["amp_dtype"]

    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
        output = generate_diffusion_cond_inpaint(lv.model, **gen_kwargs)

    print(f"[infer/{variant_key}] sampling done in {time.time() - t0:.1f}s", flush=True)

    progress(0.92, desc="Normalising & saving")
    cut_dur = int(duration) if cut_to_seconds_total else None
    out_path, int16_audio = _tensor_to_audio(output, lv.sample_rate, cut_dur, fmt=fmt, target_sr=target_sr, bitrate=bitrate, ogg_quality=ogg_quality, normalize=normalize, normalize_level_db=normalize_level_db)

    if not return_spectrogram:
        return out_path

    spec_img = audio_spectrogram_image(int16_audio, sample_rate=lv.sample_rate)
    return out_path, [spec_img, *preview_images]


def infer(
    variant_key: str,
    prompt: str,
    duration: int = 60,
    steps: int = 8,
    cfg_scale: float = 1.0,
    sampler_type: str = "pingpong",
    seed: int = 0,
    # Audio Export (Simple tab)
    fmt: str = "WAV32F",
    target_sr: int = 48000,
    bitrate: int = 320,
    ogg_quality: int = 5,
    normalize: bool = False,
    normalize_level_db: float = -15.0,
    progress: gr.Progress = gr.Progress(),
):
    """Slim handler used by the Simple tab and the Examples cache."""
    return _run_inference(
        variant_key=variant_key,
        prompt=prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_type=sampler_type,
        seed=seed,
        return_spectrogram=False,
        fmt=fmt,
        target_sr=target_sr,
        bitrate=bitrate,
        ogg_quality=ogg_quality,
        normalize=normalize,
        normalize_level_db=normalize_level_db,
        progress=progress,
    )


def infer_advanced(
    variant_key: str,
    prompt: str,
    negative_prompt: str,
    duration: int,
    steps: int,
    cfg_scale: float,
    sampler_type: str,
    seed: int,
    sigma_max: float,
    apg_scale: float,
    duration_padding_sec: float,
    cut_to_seconds_total: bool,
    init_audio: Optional[Tuple[int, np.ndarray]],
    init_noise_level: float,
    inpaint_audio: Optional[Tuple[int, np.ndarray]],
    mask_start_sec: float,
    mask_end_sec: float,
    preview_every: int,
    # Audio Export Parameters
    fmt: str = "WAV32F",
    target_sr: int = 48000,
    bitrate: int = 320,
    ogg_quality: int = 5,
    normalize: bool = False,
    normalize_level_db: float = -15.0,
    progress: gr.Progress = gr.Progress(),
):
    """Full-featured handler used by the Advanced tab."""
    return _run_inference(
        variant_key=variant_key,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_type=sampler_type,
        seed=seed,
        sigma_max=sigma_max,
        apg_scale=apg_scale,
        duration_padding_sec=duration_padding_sec,
        cut_to_seconds_total=cut_to_seconds_total,
        init_audio=init_audio,
        init_noise_level=init_noise_level,
        inpaint_audio=inpaint_audio,
        mask_start_sec=mask_start_sec,
        mask_end_sec=mask_end_sec,
        preview_every=preview_every,
        return_spectrogram=True,
        fmt=fmt,
        target_sr=target_sr,
        bitrate=bitrate,
        ogg_quality=ogg_quality,
        normalize=normalize,
        normalize_level_db=normalize_level_db,
        progress=progress,
    )



# UI helpers
def _get_variant_max_seconds(key: str) -> int:
    info = VARIANT_INFO.get(key)
    return info.max_seconds if info else 60


def _variant_change_simple(variant_key: str):
    lv = load_variant(variant_key)
    # Clamp duration to new model's max
    _new_dur = min(lv.variant.default_duration, lv.max_seconds)
    return (
        gr.update(maximum=lv.max_seconds, value=_new_dur,
                  label=f"Duration (s) * model max {lv.max_seconds}s"),
        gr.update(placeholder=lv.variant.placeholder),
    )


def _variant_change_advanced(variant_key: str):
    lv = load_variant(variant_key)
    dur = min(lv.variant.default_duration, lv.max_seconds)
    return (
        gr.update(maximum=lv.max_seconds, value=dur,
                  label=f"Seconds total * model max {lv.max_seconds}s"),
        gr.update(placeholder=lv.variant.placeholder),
        gr.update(maximum=float(lv.max_seconds), value=0.0),
        gr.update(maximum=float(lv.max_seconds), value=float(dur)),
    )




# UI defaults (used before any model is loaded)
_first_key = VARIANTS[0].key
_first_max = _get_variant_max_seconds(_first_key)

VARIANT_CHOICES = [(v.label, v.key) for v in VARIANTS]


theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.Color(
        name="indigo",
        c50="#eef2ff",
        c100="#e0e7ff",
        c200="#c7d2fe",
        c300="#a5b4fc",
        c400="#818cf8",
        c500="#667eea",
        c600="#5b6fd6",
        c700="#4f5fbf",
        c800="#444fa8",
        c900="#3a3f91",
        c950="#2d2f6e",
    ),
    secondary_hue=gr.themes.colors.Color(
        name="purple",
        c50="#faf5ff",
        c100="#f3e8ff",
        c200="#e9d5ff",
        c300="#d8b4fe",
        c400="#c084fc",
        c500="#a855f7",
        c600="#9333ea",
        c700="#7e22ce",
        c800="#6b21a8",
        c900="#581c87",
        c950="#3b0764",
    ),
    neutral_hue=gr.themes.colors.Color(
        name="slate",
        c50="#f8fafc",
        c100="#f1f5f9",
        c200="#e2e8f0",
        c300="#cbd5e1",
        c400="#94a3b8",
        c500="#64748b",
        c600="#475569",
        c700="#334155",
        c800="#1e293b",
        c900="#0f172a",
        c950="#020617",
    ),
    font=["Inter", "Arial", "sans-serif"],
    font_mono=["ui-monospace", "Consolas", "monospace"],
)
css = """
    /* === LAYOUT UTILITIES === */
    .square-btn {width: 40px !important; min-width: 40px !important; padding: 0 !important; height: 40px !important;}
    .voice-controls-row {align-items: center !important;}
    .voice-controls-row button {height: 40px !important; margin-top: 24px !important;}
    .generate-btn-row {margin-bottom: 10px !important;}
    .seed-row {align-items: center !important; gap: 8px !important;}
    .seed-row button {height: 40px !important; margin-top: 24px !important; min-width: 40px !important;}
    .seed-row .form {margin-bottom: 0 !important;}
    .chunking-row {align-items: center !important; gap: 8px !important;}
    .chunking-row .form {margin-bottom: 0 !important;}

    /* === SCROLLBAR === */
    ::-webkit-scrollbar {width: 8px; height: 8px;}
    ::-webkit-scrollbar-track {background: #0f172a;}
    ::-webkit-scrollbar-thumb {background: linear-gradient(#667eea, #764ba2); border-radius: 4px;}
    ::-webkit-scrollbar-thumb:hover {background: #667eea;}
    
    .btn-generate {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        border-color: #059669 !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }
    .btn-generate:hover {
        background: linear-gradient(135deg, #34d399, #10b981) !important;
    }

    .btn-restart {
        background: linear-gradient(135deg, #ef4444, #dc2626) !important;
        border-color: #dc2626 !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }
    .btn-restart:hover {
        background: linear-gradient(135deg, #f87171, #ef4444) !important;
    }

    .btn-gray {
        background: #334155 !important;
        border-color: #475569 !important;
        color: #e2e8f0 !important;
        font-size: 11px !important;
        padding: 4px 8px !important;
    }
    .btn-gray:hover {
        background: #475569 !important;
    }
    """



with gr.Blocks(title="StableAudio 3 Portable") as demo:
    gr.Markdown("<div align='center'><h1>StableAudio 3 Portable</h1></div>")

    # GPU Status
    if torch.cuda.is_available():
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gr.Markdown(f'<div style="text-align:center;padding:10px;border-radius:5px;">🟢 GPU: {torch.cuda.get_device_name(0)} VRAM: {vram_total:.1f}GB</div>')
    else:
        gr.Markdown('<div style="text-align:center;padding:10px;border-radius:5px;">⚪ CPU Mode</div>')


    with gr.Tabs():
        # Simple tab
        with gr.Tab("Simple"):
            with gr.Row():
                # Left column (2/5)
                with gr.Column(scale=2):
                    variant = gr.Dropdown(
                        choices=VARIANT_CHOICES,
                        value=_cfg.get("simple_variant", _first_key),
                        label="Model",
                    )
                    with gr.Accordion("Advanced settings", open=True):
                        steps = gr.Slider(1, 50, value=_cfg.get("simple_steps", 8), step=1, label="Steps")
                        cfg_scale = gr.Slider(0.5, 8.0, value=_cfg.get("simple_cfg_scale", 1.0), step=0.1, label="CFG scale")
                        sampler_type = gr.Dropdown(SAMPLERS, value=_cfg.get("simple_sampler_type", "pingpong"), label="Sampler")
                        seed = gr.Number(value=_cfg.get("simple_seed", 0), precision=0, label="Seed (0 = random)")
                    # Audio Export
                    with gr.Accordion("🔊 Audio Export", open=False):
                        simple_fmt = gr.Dropdown(
                            choices=ALL_FORMATS, value=_cfg.get("simple_fmt", "WAV32F"), label="Format",
                        )
                        simple_target_sr = gr.Dropdown(
                            choices=[f"{sr}" for sr in SAMPLE_RATES],
                            value=_cfg.get("simple_target_sr", "48000"), label="Sample Rate, Hz",
                        )
                        simple_bitrate = gr.Slider(
                            minimum=64, maximum=320, step=32, value=_cfg.get("simple_bitrate", 320),
                            label="Bitrate kbps (MP3/AAC/WMA)",
                        )
                        simple_ogg_quality = gr.Slider(
                            minimum=-1, maximum=10, step=1, value=_cfg.get("simple_ogg_quality", 5),
                            label="Quality -1..10 (OGG/OPUS)",
                        )
                        simple_normalize = gr.Checkbox(
                            label="Normalize Audio", value=_cfg.get("simple_normalize", False),
                            info="Normalize to target RMS level",
                        )
                        simple_normalize_level = gr.Dropdown(
                            choices=NORMALIZATION_LEVELS,
                            value=_cfg.get("simple_normalize_level", -15),
                            label="Normalization Level (dB)",
                        )

                # Right column (3/5)
                with gr.Column(scale=3):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value=_cfg.get("simple_prompt", ""),
                        placeholder=VARIANTS[0].placeholder,
                        lines=6,
                    )
                    with gr.Row():
                        simple_clear_btn = gr.Button("🗑️ Clear", elem_classes="btn-gray", scale=1)
                        simple_paste_btn = gr.Button("📋 Paste", elem_classes="btn-gray", scale=1)
                        simple_copy_btn = gr.Button("📄 Copy", elem_classes="btn-gray", scale=1)
                    _simple_dur = _cfg.get("simple_duration", VARIANTS[0].default_duration)
                    _simple_dur = max(1, min(int(_simple_dur), _first_max))
                    duration = gr.Slider(
                        1, _first_max,
                        value=_simple_dur, step=1,
                        label=f"Duration (s) * model max {_first_max}s",
                    )
                    with gr.Row(elem_classes="generate-btn-row"):
                            run_btn = gr.Button("▶ GENERATE", variant="primary", elem_classes="btn-generate", scale=3)
                    audio_out = gr.Audio(label="Output", type="filepath", autoplay=True)

            variant.change(
                fn=_variant_change_simple,
                inputs=[variant],
                outputs=[duration, prompt],
            )

            # Button click handlers
            simple_clear_btn.click(lambda: "", inputs=None, outputs=prompt)
            simple_paste_btn.click(
                None, inputs=None, outputs=prompt,
                js="""
                async () => {
                    try {
                        const text = await navigator.clipboard.readText();
                        return text;
                    } catch (err) {
                        alert("Error accessing clipboard: " + err);
                        return "";
                    }
                }
                """
            )
            simple_copy_btn.click(
                None, inputs=[prompt], outputs=None,
                js="(text) => navigator.clipboard.writeText(text)"
            )

            run_btn.click(
                fn=infer,
                inputs=[
                    variant, prompt, duration, steps, cfg_scale, sampler_type, seed,
                    simple_fmt, simple_target_sr, simple_bitrate, simple_ogg_quality,
                    simple_normalize, simple_normalize_level,
                ],
                outputs=[audio_out],
            )

        # Advanced tab
        with gr.Tab("Advanced"):
            with gr.Row():
                # Left column (2/5)
                with gr.Column(scale=2):
                    adv_variant = gr.Dropdown(
                        choices=VARIANT_CHOICES,
                        value=_cfg.get("adv_variant", _first_key),
                        label="Model",
                    )

                    with gr.Accordion("Init audio", open=False):
                        adv_init_audio = gr.Audio(
                            label="Init audio",
                            type="numpy",
                        )
                        adv_init_noise = gr.Slider(
                            minimum=0.01, maximum=1.0, step=0.01, value=_cfg.get("adv_init_noise", 0.9),
                            label="Init noise level",
                        )

                    with gr.Accordion("Inpainting", open=False):
                        adv_inpaint_audio = gr.Audio(
                            label="Inpaint audio",
                            type="numpy",
                        )
                        _mask_start = _cfg.get("adv_mask_start", 0.0)
                        _mask_start = max(0.0, min(float(_mask_start), float(_first_max)))
                        adv_mask_start = gr.Slider(
                            minimum=0.0,
                            maximum=float(_first_max),
                            step=0.1, value=_mask_start, label="Mask start (sec)",
                        )
                        _mask_end = _cfg.get("adv_mask_end", 0.0)
                        _mask_end = max(0.0, min(float(_mask_end), float(_first_max)))
                        adv_mask_end = gr.Slider(
                            minimum=0.0,
                            maximum=float(_first_max),
                            step=0.1, value=_mask_end, label="Mask end (sec)",
                        )


                    with gr.Accordion("Advanced settings", open=True):
                            adv_steps = gr.Slider(
                                minimum=1, maximum=500, step=1, value=_cfg.get("adv_steps", 8), label="Steps"
                            )
                            adv_cfg = gr.Slider(
                                minimum=0.0, maximum=25.0, step=0.1, value=_cfg.get("adv_cfg", 1.0),
                                label="CFG scale",
                            )

                    with gr.Accordion("Sampler params", open=False):
                        with gr.Row():
                            adv_seed = gr.Number(
                                label="Seed (-1 = random)",
                                value=_cfg.get("adv_seed", -1), precision=0,
                            )
                            adv_sampler = gr.Dropdown(
                                SAMPLERS, label="Sampler type", value=_cfg.get("adv_sampler", "pingpong"),
                            )
                        adv_sigma_max = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.01, value=_cfg.get("adv_sigma_max", 1.0),
                            label="Sigma max",
                        )
                        adv_apg = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.1, value=_cfg.get("adv_apg", 1.0),
                            label="APG scale", info="1.0=full APG, 0.0=vanilla CFG",
                        )
                        adv_dur_padding = gr.Slider(
                            minimum=0.0, maximum=30.0, step=0.5, value=_cfg.get("adv_dur_padding", 6.0),
                            label="Duration padding (sec)",
                        )

                    with gr.Accordion("Output params", open=False):
                        adv_preview_every = gr.Slider(
                            minimum=0, maximum=100, step=1, value=_cfg.get("adv_preview_every", 0),
                            label="Spec preview every N steps (0 = off)",
                        )
                        adv_cut_to_total = gr.Checkbox(
                            label="Cut to seconds total", value=_cfg.get("adv_cut_to_total", True),
                        )

                    with gr.Accordion("🔊 Audio Export", open=False):
                        with gr.Row():
                            adv_fmt = gr.Dropdown(
                                choices=ALL_FORMATS,
                                value=_cfg.get("adv_fmt", "WAV32F"),
                                label="Format",
                            )
                            adv_target_sr = gr.Dropdown(
                                choices=[f"{sr}" for sr in SAMPLE_RATES],
                                value=_cfg.get("adv_target_sr", "48000"),
                                label="Sample Rate",
                            )
                        adv_bitrate = gr.Slider(
                            minimum=64, maximum=320, step=32, value=_cfg.get("adv_bitrate", 320),
                            label="Bitrate kbps (MP3/AAC/WMA)",
                        )
                        adv_ogg_quality = gr.Slider(
                            minimum=-1, maximum=10, step=1, value=_cfg.get("adv_ogg_quality", 5),
                            label="Quality -1..10 (OGG/OPUS)",
                        )
                        with gr.Row():
                            adv_normalize = gr.Checkbox(
                                label="Normalize Audio", value=_cfg.get("adv_normalize", False),
                                info="Normalize to target RMS level",
                            )
                            adv_normalize_level = gr.Dropdown(
                                choices=NORMALIZATION_LEVELS,
                                value=_cfg.get("adv_normalize_level", -15),
                                label="Normalization Level (dB)",
                            )

                # Right column (3/5)
                with gr.Column(scale=3):
                    adv_prompt = gr.Textbox(
                        label="Prompt",
                        value=_cfg.get("adv_prompt", ""),
                        placeholder=VARIANTS[0].placeholder,
                        lines=3,
                    )
                    with gr.Row():
                        adv_clear_btn = gr.Button("🗑️ Clear", elem_classes="btn-gray", scale=1)
                        adv_paste_btn = gr.Button("📋 Paste", elem_classes="btn-gray", scale=1)
                        adv_copy_btn = gr.Button("📄 Copy", elem_classes="btn-gray", scale=1)
                    adv_negative = gr.Textbox(
                        label="Negative prompt",
                        value=_cfg.get("adv_negative", ""),
                        placeholder="Negative prompt",
                        lines=2,
                    )
                    _adv_dur = _cfg.get("adv_seconds_total", VARIANTS[0].default_duration)
                    _adv_dur = max(1, min(int(_adv_dur), _first_max))
                    adv_seconds_total = gr.Slider(
                        minimum=1,
                        maximum=_first_max,
                        step=1,
                        value=_adv_dur,
                        label=f"Duration (s) * model max {_first_max}s",
                    )
                    
                    with gr.Row(elem_classes="generate-btn-row"):
                            adv_generate = gr.Button("▶ GENERATE", variant="primary", elem_classes="btn-generate", scale=3)
                    adv_audio_out = gr.Audio(
                        label="Output audio", type="filepath", autoplay=False,
                        sources=[],
                    )
                    adv_spec_gallery = gr.Gallery(
                        label="Output spectrogram", show_label=True, columns=2,
                    )
                    with gr.Row():
                        send_to_init_btn = gr.Button("Send to init audio", scale=1)
                        send_to_inpaint_btn = gr.Button("Send to inpaint audio", scale=1)

            send_to_init_btn.click(
                fn=lambda a: a, inputs=[adv_audio_out], outputs=[adv_init_audio]
            )
            send_to_inpaint_btn.click(
                fn=lambda a: a, inputs=[adv_audio_out], outputs=[adv_inpaint_audio]
            )

            # Keep the inpaint mask bounded by the current duration.
            def _update_mask_max(seconds_total):
                m = max(float(seconds_total), 1.0)
                return (
                    gr.update(maximum=m),
                    gr.update(maximum=m),
                )
            adv_seconds_total.change(
                _update_mask_max,
                inputs=[adv_seconds_total],
                outputs=[adv_mask_start, adv_mask_end],
            )

            adv_variant.change(
                fn=_variant_change_advanced,
                inputs=[adv_variant],
                outputs=[adv_seconds_total, adv_prompt, adv_mask_start, adv_mask_end],
            )

            # Button click handlers
            adv_clear_btn.click(lambda: "", inputs=None, outputs=adv_prompt)
            adv_paste_btn.click(
                None, inputs=None, outputs=adv_prompt,
                js="""
                async () => {
                    try {
                        const text = await navigator.clipboard.readText();
                        return text;
                    } catch (err) {
                        alert("Error accessing clipboard: " + err);
                        return "";
                    }
                }
                """
            )
            adv_copy_btn.click(
                None, inputs=[adv_prompt], outputs=None,
                js="(text) => navigator.clipboard.writeText(text)"
            )

            adv_generate.click(
                fn=infer_advanced,
                inputs=[
                    adv_variant,
                    adv_prompt,
                    adv_negative,
                    adv_seconds_total,
                    adv_steps,
                    adv_cfg,
                    adv_sampler,
                    adv_seed,
                    adv_sigma_max,
                    adv_apg,
                    adv_dur_padding,
                    adv_cut_to_total,
                    adv_init_audio,
                    adv_init_noise,
                    adv_inpaint_audio,
                    adv_mask_start,
                    adv_mask_end,
                    adv_preview_every,
                    # Audio Export inputs
                    adv_fmt,
                    adv_target_sr,
                    adv_bitrate,
                    adv_ogg_quality,
                    adv_normalize,
                    adv_normalize_level,
                ],
                outputs=[adv_audio_out, adv_spec_gallery],
            )




    # CONFIG SAVE HANDLERS

    def _save_simple_settings(variant_val, steps_val, cfg_val, sampler_val, seed_val,
                               duration_val, prompt_val, fmt_val, target_sr_val, bitrate_val,
                               ogg_q_val, norm_val, norm_level_val):
        """Saves Simple tab settings into config file."""
        _cfg.save({
            "simple_variant": variant_val,
            "simple_steps": steps_val,
            "simple_cfg_scale": cfg_val,
            "simple_sampler_type": sampler_val,
            "simple_seed": seed_val,
            "simple_duration": duration_val,
            "simple_prompt": prompt_val,
            "simple_fmt": fmt_val,
            "simple_target_sr": target_sr_val,
            "simple_bitrate": bitrate_val,
            "simple_ogg_quality": ogg_q_val,
            "simple_normalize": norm_val,
            "simple_normalize_level": norm_level_val,
        })
        return

    def _save_adv_settings(variant_val, steps_val, cfg_val, sampler_val, seed_val,
                            sigma_max_val, apg_val, dur_pad_val, seconds_val, prompt_val,
                            neg_val, init_noise_val, mask_start_val, mask_end_val,
                            preview_val, cut_val, fmt_val, target_sr_val, bitrate_val,
                            ogg_q_val, norm_val, norm_level_val):
        """Saves Advanced tab settings into config file."""
        _cfg.save({
            "adv_variant": variant_val,
            "adv_steps": steps_val,
            "adv_cfg": cfg_val,
            "adv_sampler": sampler_val,
            "adv_seed": seed_val,
            "adv_sigma_max": sigma_max_val,
            "adv_apg": apg_val,
            "adv_dur_padding": dur_pad_val,
            "adv_seconds_total": seconds_val,
            "adv_prompt": prompt_val,
            "adv_negative": neg_val,
            "adv_init_noise": init_noise_val,
            "adv_mask_start": mask_start_val,
            "adv_mask_end": mask_end_val,
            "adv_preview_every": preview_val,
            "adv_cut_to_total": cut_val,
            "adv_fmt": fmt_val,
            "adv_target_sr": target_sr_val,
            "adv_bitrate": bitrate_val,
            "adv_ogg_quality": ogg_q_val,
            "adv_normalize": norm_val,
            "adv_normalize_level": norm_level_val,
        })
        return

    # Simple tab autosave handlers
    variant.change(fn=_save_simple_settings, inputs=[
        variant, steps, cfg_scale, sampler_type, seed, duration, prompt,
        simple_fmt, simple_target_sr, simple_bitrate, simple_ogg_quality,
        simple_normalize, simple_normalize_level,
    ], outputs=[])

    steps.change(fn=_save_simple_settings, inputs=[
        variant, steps, cfg_scale, sampler_type, seed, duration, prompt,
        simple_fmt, simple_target_sr, simple_bitrate, simple_ogg_quality,
        simple_normalize, simple_normalize_level,
    ], outputs=[])

    cfg_scale.change(fn=_save_simple_settings, inputs=[
        variant, steps, cfg_scale, sampler_type, seed, duration, prompt,
        simple_fmt, simple_target_sr, simple_bitrate, simple_ogg_quality,
        simple_normalize, simple_normalize_level,
    ], outputs=[])

    sampler_type.change(fn=_save_simple_settings, inputs=[
        variant, steps, cfg_scale, sampler_type, seed, duration, prompt,
        simple_fmt, simple_target_sr, simple_bitrate, simple_ogg_quality,
        simple_normalize, simple_normalize_level,
    ], outputs=[])

    seed.change(fn=_save_simple_settings, inputs=[
        variant, steps, cfg_scale, sampler_type, seed, duration, prompt,
        simple_fmt, simple_target_sr, simple_bitrate, simple_ogg_quality,
        simple_normalize, simple_normalize_level,
    ], outputs=[])

    duration.change(fn=_save_simple_settings, inputs=[
        variant, steps, cfg_scale, sampler_type, seed, duration, prompt,
        simple_fmt, simple_target_sr, simple_bitrate, simple_ogg_quality,
        simple_normalize, simple_normalize_level,
    ], outputs=[])

    prompt.change(fn=_save_simple_settings, inputs=[
        variant, steps, cfg_scale, sampler_type, seed, duration, prompt,
        simple_fmt, simple_target_sr, simple_bitrate, simple_ogg_quality,
        simple_normalize, simple_normalize_level,
    ], outputs=[])

    simple_fmt.change(fn=_save_simple_settings, inputs=[
        variant, steps, cfg_scale, sampler_type, seed, duration, prompt,
        simple_fmt, simple_target_sr, simple_bitrate, simple_ogg_quality,
        simple_normalize, simple_normalize_level,
    ], outputs=[])

    simple_target_sr.change(fn=_save_simple_settings, inputs=[
        variant, steps, cfg_scale, sampler_type, seed, duration, prompt,
        simple_fmt, simple_target_sr, simple_bitrate, simple_ogg_quality,
        simple_normalize, simple_normalize_level,
    ], outputs=[])

    simple_bitrate.change(fn=_save_simple_settings, inputs=[
        variant, steps, cfg_scale, sampler_type, seed, duration, prompt,
        simple_fmt, simple_target_sr, simple_bitrate, simple_ogg_quality,
        simple_normalize, simple_normalize_level,
    ], outputs=[])

    simple_ogg_quality.change(fn=_save_simple_settings, inputs=[
        variant, steps, cfg_scale, sampler_type, seed, duration, prompt,
        simple_fmt, simple_target_sr, simple_bitrate, simple_ogg_quality,
        simple_normalize, simple_normalize_level,
    ], outputs=[])

    simple_normalize.change(fn=_save_simple_settings, inputs=[
        variant, steps, cfg_scale, sampler_type, seed, duration, prompt,
        simple_fmt, simple_target_sr, simple_bitrate, simple_ogg_quality,
        simple_normalize, simple_normalize_level,
    ], outputs=[])

    simple_normalize_level.change(fn=_save_simple_settings, inputs=[
        variant, steps, cfg_scale, sampler_type, seed, duration, prompt,
        simple_fmt, simple_target_sr, simple_bitrate, simple_ogg_quality,
        simple_normalize, simple_normalize_level,
    ], outputs=[])

    # Advanced tab autosave handlers
    adv_variant.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_steps.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_cfg.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_sampler.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_seed.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_sigma_max.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_apg.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_dur_padding.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_seconds_total.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_prompt.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_negative.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_init_noise.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_mask_start.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_mask_end.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_preview_every.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_cut_to_total.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_fmt.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_target_sr.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_bitrate.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_ogg_quality.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_normalize.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])

    adv_normalize_level.change(fn=_save_adv_settings, inputs=[
        adv_variant, adv_steps, adv_cfg, adv_sampler, adv_seed, adv_sigma_max, adv_apg,
        adv_dur_padding, adv_seconds_total, adv_prompt, adv_negative, adv_init_noise,
        adv_mask_start, adv_mask_end, adv_preview_every, adv_cut_to_total,
        adv_fmt, adv_target_sr, adv_bitrate, adv_ogg_quality, adv_normalize, adv_normalize_level,
    ], outputs=[])



def on_exit():
    unload_current_model()
    print("[INFO] Clean exit.")

import atexit
atexit.register(on_exit)


if __name__ == "__main__":
    print("[INFO] Starting Stable Audio 3 on http://127.0.0.1:8990")
    demo.launch(
        server_name="127.0.0.1",
        server_port=8990,
        share=False,
        inbrowser=True,
        theme=theme,
        css=css
    )
