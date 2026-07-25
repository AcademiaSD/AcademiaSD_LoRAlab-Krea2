import os
import gc
import math
import time
import random
import json
import signal
import sys
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import bitsandbytes as bnb
from safetensors.torch import save_file, load, load_file
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors import safe_open
from bitsandbytes.functional import QuantState

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HF_REPO_ID = "AcademiaSD/Krea-2-NF4-for-LoRA-Training"

DEFAULTS = {
    "model_id": "Krea-2-NF4",
    "cache_dir": "./cached_data_krea2",
    "output_dir": "./krea2_lora_output",
    "total_steps": 500,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 1e-4,
    "min_lr_ratio": 0.1,
    "warmup_steps": 100,
    "lora_rank": 16,
    "lora_alpha": 32,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "save_every": 25,
    "seed": 42,
    "timestep_sampling": "krea2_shift",
    "preview_every": 0,
    "preview_steps": 8,
    "preview_cfg": 0.0,
    "preview_caption_mode": "first",
    "preview_custom_prompt": "",
    "use_turbo": True,
    "turbo_lora_strength": 1.0,
    "use_filter_bypass": False,
    "filter_bypass_strength": 10.0,
    "project_name": "",
    "trigger_word": "",
}

CONFIG_PATH = "train_settings.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"[OK] Configuration loaded from {CONFIG_PATH} / Configuración cargada desde {CONFIG_PATH}")
else:
    cfg = {}
    print(f"[!] {CONFIG_PATH} not found, using default values / No se encontró {CONFIG_PATH}, usando valores por defecto.")

MODEL_ID          = cfg.get("model_id",          DEFAULTS["model_id"])
TOTAL_STEPS       = cfg.get("total_steps",       DEFAULTS["total_steps"])
BATCH_SIZE        = cfg.get("batch_size",        DEFAULTS["batch_size"])
GRAD_ACCUM_STEPS  = cfg.get("grad_accum_steps",  DEFAULTS["grad_accum_steps"])
LR                = cfg.get("lr",                DEFAULTS["lr"])
MIN_LR_RATIO      = cfg.get("min_lr_ratio",      DEFAULTS["min_lr_ratio"])
WARMUP_STEPS      = cfg.get("warmup_steps",      DEFAULTS["warmup_steps"])
LORA_RANK         = cfg.get("lora_rank",         DEFAULTS["lora_rank"])
LORA_ALPHA        = cfg.get("lora_alpha",        DEFAULTS["lora_alpha"])
WEIGHT_DECAY      = cfg.get("weight_decay",      DEFAULTS["weight_decay"])
MAX_GRAD_NORM     = cfg.get("max_grad_norm",     DEFAULTS["max_grad_norm"])
SAVE_EVERY        = cfg.get("save_every",        DEFAULTS["save_every"])
SEED              = cfg.get("seed",              DEFAULTS["seed"])
TIMESTEP_SAMPLING = cfg.get("timestep_sampling", DEFAULTS["timestep_sampling"])
PREVIEW_EVERY     = cfg.get("preview_every",     DEFAULTS["preview_every"])
PREVIEW_STEPS     = cfg.get("preview_steps",     DEFAULTS["preview_steps"])
PREVIEW_CFG       = cfg.get("preview_cfg",       DEFAULTS["preview_cfg"])
PREVIEW_CAPTION_MODE  = cfg.get("preview_caption_mode",  DEFAULTS["preview_caption_mode"])
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"]).strip()

USE_TURBO            = cfg.get("use_turbo",            DEFAULTS["use_turbo"])
TURBO_LORA_STRENGTH  = cfg.get("turbo_lora_strength",  DEFAULTS["turbo_lora_strength"])
TURBO_LORA_PATH      = os.path.join(MODEL_ID, "LoRAs", "krea2_turbo_lora_rank_64_bf16.safetensors")

USE_FILTER_BYPASS      = cfg.get("use_filter_bypass",      DEFAULTS["use_filter_bypass"])
FILTER_BYPASS_STRENGTH = cfg.get("filter_bypass_strength", DEFAULTS["filter_bypass_strength"])
FILTER_BYPASS_PATH     = os.path.join(MODEL_ID, "LoRAs", "krea2filterbypass3.safetensors")

TRIGGER_WORD      = cfg.get("trigger_word", "")
PROJECT_NAME      = cfg.get("project_name", "").strip()

if PROJECT_NAME:
    CACHE_DIR  = f"./cached_data_krea2_{PROJECT_NAME}"
    OUTPUT_DIR = f"./krea2_lora_output_{PROJECT_NAME}"
else:
    CACHE_DIR  = cfg.get("cache_dir",  DEFAULTS["cache_dir"])
    OUTPUT_DIR = cfg.get("output_dir", DEFAULTS["output_dir"])

print(f"  Model ID / ID Modelo     : {MODEL_ID}")
print(f"  Project / Proyecto       : {PROJECT_NAME if PROJECT_NAME else '(Default)'}")
print(f"  Trigger Word / Palabra   : {TRIGGER_WORD}")
print(f"  Cache Dir / Carpeta Caché: {CACHE_DIR}")
print(f"  Output Dir / Salida      : {OUTPUT_DIR}")
print(f"  Total Steps / Pasos      : {TOTAL_STEPS}")
print(f"  Learning Rate / LR       : {LR}")
print(f"  LoRA Rank/Alpha          : {LORA_RANK}/{LORA_ALPHA}")
print(f"  Batch / Grad Accum       : {BATCH_SIZE}/{GRAD_ACCUM_STEPS}")
print(f"  Preview Mode / Prompt    : Mode={PREVIEW_CAPTION_MODE} | Custom='{PREVIEW_CUSTOM_PROMPT}'")
print(f"  Preview Every / Steps / CFG: {PREVIEW_EVERY} / {PREVIEW_STEPS} / {PREVIEW_CFG}")
print(f"  Seed Configured / Semilla: {SEED} ({'RANDOM' if SEED <= 0 else 'FIXED'})")
print(f"  Turbo LoRA Accelerated   : {'ON (Strength=' + str(TURBO_LORA_STRENGTH) + ')' if USE_TURBO else 'OFF'}")
print(f"  Filter Bypass LoRA       : {'ON (Strength=' + str(FILTER_BYPASS_STRENGTH) + ')' if USE_FILTER_BYPASS else 'OFF'}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
RESUME_DIR = os.path.join(OUTPUT_DIR, "resume_checkpoint")
OPT_FILE   = os.path.join(OUTPUT_DIR, "optimizer.pt")
STEP_FILE  = os.path.join(OUTPUT_DIR, "current_step.txt")

if SEED > 0:
    torch.manual_seed(SEED)
    random.seed(SEED)


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


def get_hf_token():
    if os.path.exists("HF_token.json"):
        try:
            with open("HF_token.json", "r", encoding="utf-8") as f:
                token_data = json.load(f)
                token = token_data.get("token", "").strip()
                if token:
                    print("✓ Using HF Token / Usando token de HF")
                    return token
        except Exception:
            pass
    return None


def enable_hf_file_progress():
    try:
        import tqdm
        import tqdm.auto
        import tqdm.std

        def patch_tqdm(cls):
            orig_init = cls.__init__
            def new_init(self, *args, **kwargs):
                kwargs['disable'] = False
                kwargs['mininterval'] = 0.5
                orig_init(self, *args, **kwargs)
            cls.__init__ = new_init

        patch_tqdm(tqdm.std.tqdm)
        patch_tqdm(tqdm.auto.tqdm)
        if hasattr(tqdm, 'tqdm'):
            patch_tqdm(tqdm.tqdm)

        from huggingface_hub.utils import enable_progress_bars
        enable_progress_bars()
        logging.getLogger("huggingface_hub").setLevel(logging.INFO)
    except Exception:
        pass

    os.environ["TQDM_DISABLE"] = "0"
    os.environ["TQDM_MININTERVAL"] = "0.5"


def ensure_file_downloaded(local_path, repo_id, filename_in_repo):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    print(f"⚠ Missing LoRA file / Archivo LoRA no encontrado: {local_path}")
    print(f"  Downloading from HF / Descargando desde Hugging Face: {filename_in_repo}")

    enable_hf_file_progress()
    hf_token = get_hf_token()

    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename_in_repo,
            local_dir=MODEL_ID,
            local_dir_use_symlinks=False,
            token=hf_token
        )
        print(f"✓ LoRA downloaded successfully / Descargado con éxito: {downloaded}")
        return downloaded
    except Exception as e:
        print(f"[!] Warning downloading {filename_in_repo} from HF: {e}")
        return local_path


def ensure_model_downloaded(local_path, repo_id):
    if os.path.exists(local_path) and os.path.isdir(local_path):
        has_content = any(
            os.path.exists(os.path.join(local_path, f))
            for f in ["index.json", "model_index.json", "config.json"]
        ) or len(os.listdir(local_path)) > 0
        if has_content:
            print(f"[OK] Local model found at / Modelo local encontrado en: {local_path}")
            return local_path

    print(f"⚠ Local model not found at / No se encontró modelo local en: {local_path}")
    print(f"  Downloading from Hugging Face / Descargando desde Hugging Face: {repo_id}")

    enable_hf_file_progress()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with pip install huggingface_hub")

    hf_token = get_hf_token()

    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=hf_token,
        max_workers=2,
    )

    print(f"[OK] Model downloaded to / Modelo descargado en: {downloaded_path}")
    return downloaded_path


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=6400,
                    base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def sample_sigma(batch_size, image_seq_len, device, shift_cfg):
    if TIMESTEP_SAMPLING == "logit_normal":
        u = torch.sigmoid(torch.randn(batch_size, device=device))
    else:
        u = torch.rand(batch_size, device=device)
    mu = calculate_shift(image_seq_len, *shift_cfg)
    e_mu = math.exp(mu)
    sigma = e_mu / (e_mu + (1.0 / u.clamp(1e-6, 1 - 1e-6) - 1.0))
    return sigma.clamp(1e-4, 1.0 - 1e-4)


def pack_latents(x):
    B, C, H, W = x.shape
    H = H - (H % 2)
    W = W - (W % 2)
    x = x[:, :, :H, :W]
    x = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // 2) * (W // 2), C * 4)


def unpack_latents(x, H, W):
    B, _, C = x.shape
    x = x.view(B, H // 2, W // 2, C // 4, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, C // 4, H, W)


def prepare_position_ids(text_seq_len, grid_h, grid_w, device):
    text_ids = torch.zeros(text_seq_len, 3, device=device)
    image_ids = torch.zeros(grid_h, grid_w, 3, device=device)
    image_ids[..., 1] = torch.arange(grid_h, device=device)[:, None]
    image_ids[..., 2] = torch.arange(grid_w, device=device)[None, :]
    return torch.cat([text_ids, image_ids.reshape(grid_h * grid_w, 3)], dim=0)


SKIP_QUANT = ("img_in", "time_embed", "time_mod_proj", "txt_in", "final_layer")


def quantize_to_nf4_(module, prefix=""):
    from bitsandbytes.nn import Linear4bit, Params4bit
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear) and not any(s in full for s in SKIP_QUANT):
            w = child.weight.data.float().contiguous()
            new_layer = Linear4bit(
                child.in_features, child.out_features,
                bias=child.bias is not None, quant_type="nf4",
                compute_dtype=torch.bfloat16,
            )
            new_layer.weight = Params4bit(w, requires_grad=False, quant_type="nf4")
            if child.bias is not None:
                new_layer.bias = torch.nn.Parameter(child.bias.data, requires_grad=False)
            setattr(module, name, new_layer)
            del child, w
        else:
            quantize_to_nf4_(child, full)


def load_nf4_cache_(transformer, cache_dir):
    index_path = os.path.join(cache_dir, "index.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index.json not found in NF4 cache / No existe index.json en caché NF4: {cache_dir}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    quantized = index.get("quantized", {})
    weights_dir = os.path.join(cache_dir, "weights")
    replaced = 0

    def get_parent_module(root, module_name):
        parts = module_name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    for name, info in quantized.items():
        filepath = os.path.join(weights_dir, info["file"])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"NF4 weight file not found / No existe archivo NF4: {filepath}")

        parent, child_name = get_parent_module(transformer, name)

        with safe_open(filepath, framework="pt", device="cpu") as f:
            weight_data = f.get_tensor("weight")
            bias_data = None
            if info.get("bias", False):
                bias_data = f.get_tensor("bias")

            qs_dict = {}
            for key in f.keys():
                if not key.startswith("quant_state."):
                    continue
                qs_key = key[len("quant_state."):]
                qs_dict[qs_key] = f.get_tensor(key)

            packed_qs = {
                "absmax": qs_dict["absmax"],
                "nested_absmax": qs_dict["nested_absmax"],
                "nested_quant_map": qs_dict["nested_quant_map"],
                "quant_map": qs_dict["quant_map"],
                "quant_state.bitsandbytes__nf4": qs_dict["quant_state.bitsandbytes__nf4"],
            }

            quant_state = QuantState.from_dict(packed_qs, device="cpu")

        new_weight = Params4bit(
            weight_data,
            requires_grad=False,
            quant_type="nf4",
            quant_storage=torch.uint8,
        )
        new_weight.quant_state = quant_state
        new_weight.bnb_quantized = True

        new_layer = Linear4bit(
            info["in_features"],
            info["out_features"],
            bias=info["bias"],
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
        )
        new_layer.weight = new_weight
        if bias_data is not None:
            new_layer.bias = torch.nn.Parameter(bias_data, requires_grad=False)

        setattr(parent, child_name, new_layer)
        replaced += 1

    print(f"Reconstructed NF4 layers / Capas NF4 reconstruidas: {replaced}")

    verified = 0
    for name, layer in transformer.named_modules():
        if isinstance(layer, Linear4bit):
            if getattr(layer.weight, "bnb_quantized", False):
                if layer.weight.quant_state is not None:
                    verified += 1

    print(f"Verified NF4 layers / Capas NF4 verificadas: {verified}")

    if verified != replaced:
        raise RuntimeError("NF4 Verification mismatch / La verificación NF4 no coincide")

    print("[OK] NF4 cache loaded successfully / Caché NF4 cargada correctamente.")
    return transformer


def _export_lora(model, path):
    clean = {}
    for k, v in model.state_dict().items():
        if "lora_" not in k:
            continue
        new_key = "transformer." + k.replace("base_model.model.", "")
        clean[new_key] = v.to(torch.bfloat16).cpu().contiguous()
    save_file(clean, path, metadata={"format": "pt"})


class VaeHolder:
    vae = None
    @classmethod
    def get(cls):
        if cls.vae is None:
            from diffusers import AutoencoderKLQwenImage
            cls.vae = AutoencoderKLQwenImage.from_pretrained(
                MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16)
        return cls.vae


def clean_module_prefix(name):
    if not isinstance(name, str):
        return ""
    name = name.strip()

    prefixes = (
        "base_model.model.",
        "base_model.",
        "model.",
        "transformer.",
        "diffusion_model.",
        "lora_unet_",
    )

    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if name.startswith(p):
                name = name[len(p):]
                changed = True

    return name


TURBO_KREA2_ALIASES = {
    "attn.wq": "attn.to_q",
    "attn.wk": "attn.to_k",
    "attn.wv": "attn.to_v",
    "attn.wo": "attn.to_out.0",
    "attn.gate": "attn.to_gate",
    "mlp.gate": "ff.gate",
    "mlp.up": "ff.up",
    "mlp.down": "ff.down",
}


def _normalize_dot_name(name):
    name = clean_module_prefix(name)
    return name.replace("-", "_").strip(".").lower()


def _turbo_to_krea2_dot_name(name):
    s = _normalize_dot_name(name)

    if s.startswith("transformer_blocks."):
        return s

    if s.startswith("blocks."):
        s = "transformer_blocks." + s[len("blocks."):]
        for old, new in sorted(TURBO_KREA2_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
            suffix = "." + old
            if s.endswith(suffix):
                s = s[:-len(suffix)] + "." + new
                break
        return s

    TOP_LEVEL_MAP = {
        "first": "img_in",
        "last.linear": "final_layer",
        "tmlp.0": "time_embed.linear_1",
        "tmlp.2": "time_embed.linear_2",
        "tproj.1": "time_mod_proj",
        "txtmlp.1": "txt_in.linear_1",
        "txtmlp.3": "txt_in.linear_2",
    }
    if s in TOP_LEVEL_MAP:
        return TOP_LEVEL_MAP[s]

    if s.startswith("txtfusion."):
        s = "text_fusion." + s[len("txtfusion."):]
        for old, new in sorted(TURBO_KREA2_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
            suffix = "." + old
            if s.endswith(suffix):
                s = s[:-len(suffix)] + "." + new
                break
        return s

    return s


def get_candidate_keys(base_k):
    clean = _normalize_dot_name(base_k)
    mapped = _turbo_to_krea2_dot_name(base_k)

    candidates_dot = [
        mapped,
        clean,
    ]

    if clean.startswith("blocks."):
        candidates_dot.append(mapped.replace("transformer_blocks.", "single_transformer_blocks.", 1))
        candidates_dot.append(mapped.replace("transformer_blocks.", "double_blocks.", 1))
        candidates_dot.append(mapped.replace("transformer_blocks.", "single_blocks.", 1))

    candidates = []
    for cand in candidates_dot:
        cand = cand.strip(".")
        candidates.extend([
            cand,
            "base_model.model." + cand,
            "base_model." + cand,
            "model." + cand,
            "transformer." + cand,
        ])

    for cand in list(candidates_dot):
        u = cand.replace(".", "_")
        candidates.extend([
            u,
            "base_model_model_" + u,
            "base_model_" + u,
        ])

    return list(dict.fromkeys(candidates))


def _extract_external_lora_pairs(sd, device, dtype):
    lora_pairs = {}
    direct_deltas = {}

    for k, v in sd.items():
        is_lora = False

        for pattern in (".lora_A", ".lora_down", ".lora.down", "_lora_A", "_lora_down"):
            if pattern in k:
                base_k = k.split(pattern, 1)[0]
                lora_pairs.setdefault(base_k, {})["A"] = v.to(device=device, dtype=dtype)
                is_lora = True
                break

        if is_lora:
            continue

        for pattern in (".lora_B", ".lora_up", ".lora.up", "_lora_B", "_lora_up"):
            if pattern in k:
                base_k = k.split(pattern, 1)[0]
                lora_pairs.setdefault(base_k, {})["B"] = v.to(device=device, dtype=dtype)
                is_lora = True
                break

        if not is_lora:
            base_k = k
            for suffix in (".diff_b", ".diff", ".weight", ".bias"):
                if base_k.endswith(suffix):
                    base_k = base_k[:-len(suffix)]
                    break
            direct_deltas[base_k] = v.to(device=device, dtype=dtype)

    return lora_pairs, direct_deltas


def _build_peft_target_map(model):
    module_map = {}
    entries = []

    for name, mod in model.named_modules():
        target_obj = None

        if hasattr(mod, "base_layer"):
            target_obj = mod.base_layer
        elif isinstance(mod, (torch.nn.Linear, bnb.nn.Linear4bit)):
            target_obj = mod
        else:
            continue

        if not isinstance(target_obj, (torch.nn.Linear, bnb.nn.Linear4bit)):
            continue

        canonical = _normalize_dot_name(name)
        canonical_no_prefix = canonical
        if canonical_no_prefix.startswith("base_model.model."):
            canonical_no_prefix = canonical_no_prefix[len("base_model.model."):]

        dims = (int(target_obj.out_features), int(target_obj.in_features))

        entry = {
            "wrapper_name": name,
            "canonical": canonical_no_prefix,
            "module": target_obj,
            "dims": dims,
        }

        entries.append(entry)
        module_map[canonical_no_prefix] = entry
        module_map[canonical] = entry

    return module_map, entries


def _resolve_external_lora_target(base_k, a_w, b_w, module_map, entries):
    candidates = get_candidate_keys(base_k)

    for cand in candidates:
        canonical = _normalize_dot_name(cand)
        if canonical in module_map:
            entry = module_map[canonical]
            expected_in, expected_out = int(a_w.shape[1]), int(b_w.shape[0])
            if entry["dims"][1] == expected_in and entry["dims"][0] == expected_out:
                return entry, "alias_exact"

    turbo = _normalize_dot_name(base_k)

    if turbo.startswith("blocks."):
        parts = turbo.split(".")
        if len(parts) >= 4:
            try:
                block_idx = int(parts[1])
            except ValueError:
                block_idx = None

            semantic = ".".join(parts[2:])
            mapped_semantic = semantic
            for old, new in sorted(TURBO_KREA2_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
                if semantic == old:
                    mapped_semantic = new
                    break

            expected_canonical = f"transformer_blocks.{block_idx}.{mapped_semantic}" if block_idx is not None else None
            if expected_canonical:
                entry = module_map.get(expected_canonical)
                if entry is not None and entry["dims"][1] == int(a_w.shape[1]) and entry["dims"][0] == int(b_w.shape[0]):
                    return entry, "explicit_structural"

    if turbo.startswith("blocks."):
        parts = turbo.split(".")
        if len(parts) >= 4:
            try:
                block_idx = int(parts[1])
            except ValueError:
                block_idx = None

            semantic = ".".join(parts[2:])
            mapped_semantic = semantic
            for old, new in sorted(TURBO_KREA2_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
                if semantic == old:
                    mapped_semantic = new
                    break

            if block_idx is not None:
                suffix = f"transformer_blocks.{block_idx}.{mapped_semantic}"
                matches = [e for e in entries if e["canonical"] == suffix and e["dims"][1] == int(a_w.shape[1]) and e["dims"][0] == int(b_w.shape[0])]
                if len(matches) == 1:
                    return matches[0], "structural"

    if turbo.startswith("blocks."):
        parts = turbo.split(".")
        if len(parts) >= 2:
            try:
                block_idx = int(parts[1])
            except ValueError:
                block_idx = None

            if block_idx is not None:
                prefix = f"transformer_blocks.{block_idx}."
                matches = [e for e in entries if e["canonical"].startswith(prefix) and e["dims"][1] == int(a_w.shape[1]) and e["dims"][0] == int(b_w.shape[0])]
                if len(matches) == 1:
                    return matches[0], "block_dimension"

    expected_in, expected_out = int(a_w.shape[1]), int(b_w.shape[0])
    matches = [e for e in entries if e["dims"] == (expected_out, expected_in)]
    if len(matches) == 1:
        return matches[0], "global_dimension"

    return None, "not_found"


def _make_external_lora_hook(a_w, b_w, strength):
    def hook(module, input_args, output):
        if not input_args:
            return output
        x = input_args[0]
        if not torch.is_tensor(x) or x.shape[-1] != a_w.shape[1]:
            return output

        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1]).to(device=a_w.device, dtype=a_w.dtype)
        inter = F.linear(x_flat, a_w)
        delta = F.linear(inter, b_w) * strength
        delta = delta.reshape(*original_shape[:-1], delta.shape[-1])
        return output + delta.to(device=output.device, dtype=output.dtype)

    return hook


def _make_direct_delta_hook(d_tensor, strength):
    def hook(module, input_args, output):
        if not input_args:
            return output
        x = input_args[0]
        if not torch.is_tensor(x):
            return output

        if d_tensor.ndim == 2:
            d = d_tensor
            if x.shape[-1] != d.shape[1]:
                if x.shape[-1] == d.shape[0]:
                    d = d.T.contiguous()
                else:
                    return output

            original_shape = x.shape
            x_flat = x.reshape(-1, x.shape[-1]).to(device=d.device, dtype=d.dtype)
            delta = F.linear(x_flat, d) * strength
            delta = delta.reshape(*original_shape[:-1], delta.shape[-1])
            return output + delta.to(device=output.device, dtype=output.dtype)

        if d_tensor.ndim == 1:
            if d_tensor.shape[0] == output.shape[-1]:
                return output + (d_tensor * strength).to(device=output.device, dtype=output.dtype)
            return output

        try:
            return output + (d_tensor * strength).to(device=output.device, dtype=output.dtype)
        except Exception:
            return output

    return hook


def _apply_layerwise_scale_hooks(model, scale_tensor, layer_indices, strength, tag):
    scales = scale_tensor.flatten().cpu().to(torch.float32).tolist()
    if len(scales) != len(layer_indices):
        return []

    block_map = {}
    for name, mod in model.named_modules():
        for pattern in ("transformer_blocks.", "blocks.", "single_transformer_blocks.", "double_blocks."):
            if pattern in name:
                suffix = name.split(pattern, 1)[1]
                part = suffix.split(".")[0]
                try:
                    idx = int(part)
                    if idx not in block_map or len(name) < len(block_map[idx][0]):
                        block_map[idx] = (name, mod)
                except ValueError:
                    pass

    hooks = []
    for idx, base_scale in zip(layer_indices, scales):
        if idx not in block_map:
            continue

        final_scale = 1.0 + (base_scale - 1.0) * strength
        name, mod = block_map[idx]

        def _make_scale_hook(s):
            def hook(module, input_args, output):
                if torch.is_tensor(output):
                    return output * s
                if isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[0]):
                    return (output[0] * s,) + output[1:]
                return output
            return hook

        handle = mod.register_forward_hook(_make_scale_hook(final_scale))
        hooks.append(handle)

    return hooks


def apply_preview_lora_hooks(
    model,
    lora_path,
    tag="Preview LoRA",
    strength=1.0,
    device="cuda",
    dtype=torch.bfloat16,
):
    if not os.path.exists(lora_path):
        print(f"  [!] {tag} file not found at: {lora_path}")
        return []

    try:
        sd = load_file(lora_path, device="cpu")
    except Exception as e:
        print(f"  [!] Failed to load {tag}: {e}")
        return []

    module_map, entries = _build_peft_target_map(model)
    lora_pairs, direct_deltas = _extract_external_lora_pairs(sd, device, dtype)

    hooks = []
    stats = {
        "alias_exact": 0,
        "explicit_structural": 0,
        "structural": 0,
        "block_dimension": 0,
        "global_dimension": 0,
        "not_found": 0,
        "dimension_error": 0,
        "direct": 0,
        "layer_scale": 0,
    }

    for base_k, weights in lora_pairs.items():
        if "A" not in weights or "B" not in weights:
            continue

        a_w, b_w = weights["A"], weights["B"]
        if a_w.ndim != 2 or b_w.ndim != 2:
            stats["dimension_error"] += 1
            continue

        target_entry, reason = _resolve_external_lora_target(base_k, a_w, b_w, module_map, entries)
        if target_entry is None:
            stats["not_found"] += 1
            continue

        expected_in, expected_out = target_entry["dims"][1], target_entry["dims"][0]
        if (a_w.shape[1] != expected_in or b_w.shape[0] != expected_out or a_w.shape[0] != b_w.shape[1]):
            if (a_w.shape[0] == expected_in and b_w.shape[1] == expected_out and a_w.shape[1] == b_w.shape[0]):
                a_w, b_w = a_w.T.contiguous(), b_w.T.contiguous()
            else:
                stats["dimension_error"] += 1
                continue

        try:
            hook_fn = _make_external_lora_hook(a_w, b_w, strength)
            handle = target_entry["module"].register_forward_hook(hook_fn)
            hooks.append(handle)
            stats[reason] = stats.get(reason, 0) + 1
        except Exception:
            pass

    for base_k, tensor in list(direct_deltas.items()):
        is_scale_vector = False
        if tensor.numel() == 12 and tensor.ndim <= 2:
            candidates = get_candidate_keys(base_k)
            maps_to_linear = any(_normalize_dot_name(cand) in module_map for cand in candidates)
            if not maps_to_linear:
                is_scale_vector = True

        if is_scale_vector:
            layer_indices = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
            scale_hooks = _apply_layerwise_scale_hooks(model, tensor, layer_indices, strength, tag)
            if scale_hooks:
                hooks.extend(scale_hooks)
                stats["layer_scale"] += 1
                del direct_deltas[base_k]
                continue

        target_entry = None
        candidates = get_candidate_keys(base_k)
        for cand in candidates:
            canonical = _normalize_dot_name(cand)
            if canonical in module_map:
                target_entry = module_map[canonical]
                break

        if target_entry is None and tensor.ndim == 2:
            matches = [e for e in entries if e["dims"] == (tensor.shape[0], tensor.shape[1])]
            if len(matches) == 1:
                target_entry = matches[0]
            else:
                matches_t = [e for e in entries if e["dims"] == (tensor.shape[1], tensor.shape[0])]
                if len(matches_t) == 1:
                    target_entry = matches_t[0]
                    tensor = tensor.T.contiguous()

        if target_entry is None:
            continue

        t = tensor
        if t.ndim == 2:
            expected_out, expected_in = target_entry["dims"]
            if t.shape == (expected_in, expected_out):
                t = t.T.contiguous()
            if t.shape != (expected_out, expected_in):
                continue

        try:
            hook_fn = _make_direct_delta_hook(t, strength)
            handle = target_entry["module"].register_forward_hook(hook_fn)
            hooks.append(handle)
            stats["direct"] += 1
        except Exception:
            pass

    total_injected = len(hooks)
    print(f"  [{tag}] Injected {total_injected} hooks (Strength={strength}).")

    return hooks


def run_preview(model, scheduler, embed, mask, neg, size, step, shift_cfg):
    H, W = size
    gh, gw = H // 16, W // 16
    device = "cuda"
    was_training = model.training
    model.eval()

    if SEED <= 0:
        actual_seed = random.randint(1, 2147483647)
    else:
        actual_seed = SEED

    print(f"  ↳ Preview Seed used / Semilla utilizada: {actual_seed}")

    preview_hooks = []
    if USE_TURBO:
        ensure_file_downloaded(TURBO_LORA_PATH, HF_REPO_ID, "LoRAs/krea2_turbo_lora_rank_64_bf16.safetensors")
        preview_hooks.extend(apply_preview_lora_hooks(model, TURBO_LORA_PATH, tag="🚀 Turbo LoRA", strength=TURBO_LORA_STRENGTH, device=device))
    if USE_FILTER_BYPASS:
        ensure_file_downloaded(FILTER_BYPASS_PATH, HF_REPO_ID, "LoRAs/krea2filterbypass3.safetensors")
        preview_hooks.extend(apply_preview_lora_hooks(model, FILTER_BYPASS_PATH, tag="🔓 Filter Bypass", strength=FILTER_BYPASS_STRENGTH, device=device))

    try:
        g = torch.Generator(device=device).manual_seed(actual_seed)
        latents = torch.randn((1, 16, H // 8, W // 8), generator=g, device=device, dtype=torch.bfloat16)
        latents = pack_latents(latents)
        pos_ids = prepare_position_ids(embed.shape[1], gh, gw, device)
        embed, mask = embed.to(device), mask.to(device)
        if neg is not None:
            neg = (neg[0].to(device), neg[1].to(device))

        sigmas = np.linspace(1.0, 1.0 / PREVIEW_STEPS, PREVIEW_STEPS)
        mu = calculate_shift(latents.shape[1], *shift_cfg)
        scheduler.set_timesteps(PREVIEW_STEPS, device=device, sigmas=sigmas, mu=mu)

        with torch.no_grad():
            for t in scheduler.timesteps:
                tt = (t / scheduler.config.num_train_timesteps).expand(1).to(torch.bfloat16)
                pred = model(hidden_states=latents, encoder_hidden_states=embed, timestep=tt,
                             position_ids=pos_ids, encoder_attention_mask=mask, return_dict=False)[0]
                if neg is not None and PREVIEW_CFG > 1.0:
                    pred_u = model(hidden_states=latents, encoder_hidden_states=neg[0], timestep=tt,
                                   position_ids=pos_ids, encoder_attention_mask=neg[1], return_dict=False)[0]
                    pred = pred + PREVIEW_CFG * (pred - pred_u)
                latents = scheduler.step(pred, t, latents, return_dict=False)[0]

            vae = VaeHolder.get().to(device)
            lat = unpack_latents(latents, H // 8, W // 8).to(vae.dtype).unsqueeze(2)
            mean = torch.tensor(vae.config.latents_mean, device=device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
            std  = torch.tensor(vae.config.latents_std,  device=device, dtype=lat.dtype).view(1, -1, 1, 1, 1)
            img = vae.decode(lat * std + mean, return_dict=False)[0][:, :, 0]
            img = ((img.float() / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
            vae.to("cpu")

        from PIL import Image
        out = os.path.join(OUTPUT_DIR, f"preview_step_{step}.png")
        Image.fromarray(img).save(out)
        print(f"  ↳ Preview saved to / Preview guardada: {out}")
    finally:
        for h in preview_hooks:
            h.remove()
        if was_training:
            model.train()
        free_vram()


def ensure_custom_prompt_encoded():
    if PREVIEW_CAPTION_MODE == "custom" and PREVIEW_CUSTOM_PROMPT:
        c_emb_path = os.path.join(CACHE_DIR, "_custom_embed.pt")
        c_msk_path = os.path.join(CACHE_DIR, "_custom_mask.pt")
        
        full_p = PREVIEW_CUSTOM_PROMPT
        if TRIGGER_WORD and TRIGGER_WORD.lower() not in full_p.lower():
            full_p = f"{TRIGGER_WORD}, {full_p}".strip(", ")
            
        print(f"\n[Custom Prompt] Encoding text: '{full_p}'...")
        te_pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            transformer=None,
            vae=None,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        with torch.inference_mode():
            c_emb, c_msk = te_pipe.encode_prompt(prompt=full_p, max_sequence_length=128)
            torch.save(c_emb.cpu(), c_emb_path)
            torch.save(c_msk.cpu(), c_msk_path)

        del te_pipe
        free_vram()
        print("  ✓ Custom Prompt encoded and ready for previews.")


def train_krea2():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(CACHE_DIR) or not any(f.endswith("_latent.pt") for f in os.listdir(CACHE_DIR)):
        print(f"\n[!] ERROR: Cache directory '{CACHE_DIR}' is empty or does not exist.")
        print(f"[!] Please run Pre-Cache first! / ¡Por favor ejecuta el Pre-Caché primero!")
        return

    ensure_model_downloaded(
        local_path=MODEL_ID,
        repo_id=HF_REPO_ID
    )

    ensure_custom_prompt_encoded()

    print("Loading Krea-2 Transformer... / Cargando Transformer de Krea-2...")

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        vae=None,
        text_encoder=None,
        torch_dtype=torch.bfloat16,
    )

    transformer = pipe.transformer
    scheduler   = pipe.scheduler

    del pipe
    free_vram()

    shift_cfg = (
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 6400),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )

    NF4_CACHE_DIR = MODEL_ID

    if os.path.exists(os.path.join(NF4_CACHE_DIR, "index.json")):
        print("\n¡NF4 CACHE DETECTED! / ¡CACHÉ NF4 DETECTADA!")
        print("Skipping NF4 quantization... / No se ejecutará cuantización NF4.")
        t0 = time.time()
        transformer = load_nf4_cache_(transformer, NF4_CACHE_DIR)
        print(f"[NF4] Cache loaded in / Caché cargada en {time.time() - t0:.1f}s", flush=True)
        transformer.to("cuda")
        free_vram()
        print(f"Transformer 12B pinned in VRAM. Usage / Uso: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    else:
        print("\nNo NF4 cache found. Performing quantization... / No se encontró caché NF4. Ejecutando cuantización...")
        quantize_to_nf4_(transformer)
        transformer.to("cuda")
        free_vram()
        print(f"Transformer 12B pinned in VRAM. Usage / Uso: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    transformer.enable_gradient_checkpointing()

    target_modules = [name for name, m in transformer.named_modules()
                      if isinstance(m, (torch.nn.Linear, bnb.nn.Linear4bit))]
    print(f"Target LoRA Layers / Capas LoRA objetivo: {len(target_modules)}")

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
        target_modules=target_modules, use_dora=False, init_lora_weights=True,
    )

    model = get_peft_model(transformer, lora_config)

    for module in model.modules():
        if hasattr(module, "lora_A"):
            for adapter in module.lora_A.values():
                adapter.to(dtype=torch.bfloat16)
        if hasattr(module, "lora_B"):
            for adapter in module.lora_B.values():
                adapter.to(dtype=torch.bfloat16)
        if hasattr(module, "lora_embedding_A"):
            for adapter in module.lora_embedding_A.values():
                adapter.data = adapter.data.to(torch.bfloat16)
        if hasattr(module, "lora_embedding_B"):
            for adapter in module.lora_embedding_B.values():
                adapter.data = adapter.data.to(torch.bfloat16)

    model.print_trainable_parameters()

    def _make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    transformer.img_in.register_forward_hook(_make_inputs_require_grad)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_at(step):
        if step < WARMUP_STEPS:
            return LR * step / max(1, WARMUP_STEPS)
        prog = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
        return LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * prog)))

    start_step = 0
    lora_weights_path = os.path.join(RESUME_DIR, "adapter_model.safetensors")
    if os.path.exists(STEP_FILE) and os.path.exists(OPT_FILE) and os.path.exists(lora_weights_path):
        print("=" * 65)
        print("¡Checkpoint detected! Restoring state... / ¡Checkpoint detectado! Restaurando estado...")
        try:
            with open(STEP_FILE, "r", encoding="utf-8") as f:
                start_step = int(f.read().strip())
            with open(lora_weights_path, "rb") as f:
                set_peft_model_state_dict(model, load(f.read()))
            optimizer.load_state_dict(torch.load(OPT_FILE, weights_only=False))
            print(f"Resuming training from step / Reanudando entrenamiento desde el paso {start_step}...")
        except Exception as e:
            print(f"[!] Warning reading checkpoint / Advertencia al leer checkpoint: {e}")
            start_step = 0
        print("=" * 65)

    last_step_executed = start_step

    def save_checkpoint_now(current_s):
        if current_s <= 0:
            return
        print(f"\nSaving checkpoint state at step / Guardando estado en paso {current_s}...")
        os.makedirs(RESUME_DIR, exist_ok=True)
        model.save_pretrained(RESUME_DIR)
        torch.save(optimizer.state_dict(), OPT_FILE)
        with open(STEP_FILE, "w", encoding="utf-8") as f:
            f.write(str(current_s))
        ckpt = os.path.join(OUTPUT_DIR, f"Krea2_LoRA_step_{current_s}.safetensors")
        _export_lora(model, ckpt)
        print(f"✓ Checkpoint saved successfully at step / Checkpoint guardado en paso {current_s}: {ckpt}")

    def handle_signal(sig, frame):
        nonlocal last_step_executed
        print(f"\n[!] Signal received / Señal de detención recibida ({sig}).")
        save_checkpoint_now(last_step_executed)
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, handle_signal)
    except Exception:
        pass

    model.train()
    optimizer.zero_grad(set_to_none=True)

    pin = torch.cuda.is_available()
    cache_data, buckets = {}, defaultdict(list)

    for f in os.listdir(CACHE_DIR):
        if not f.endswith("_latent.pt"):
            continue
        nombre = f.replace("_latent.pt", "")
        lat  = torch.load(f"{CACHE_DIR}/{nombre}_latent.pt", weights_only=True)
        emb  = torch.load(f"{CACHE_DIR}/{nombre}_embed.pt",  weights_only=True)
        msk  = torch.load(f"{CACHE_DIR}/{nombre}_mask.pt",   weights_only=True).bool()
        lat, emb, msk = lat.to(torch.bfloat16), emb.to(torch.bfloat16), msk
        if pin:
            lat, emb, msk = lat.pin_memory(), emb.pin_memory(), msk.pin_memory()
        cache_data[nombre] = (lat, emb, msk)
        buckets[(lat.shape[2], lat.shape[3])].append(nombre)

    if os.path.exists(f"{CACHE_DIR}/_custom_embed.pt"):
        first_ref = list(cache_data.values())[0][0]
        c_emb = torch.load(f"{CACHE_DIR}/_custom_embed.pt", weights_only=True).to(torch.bfloat16)
        c_msk = torch.load(f"{CACHE_DIR}/_custom_mask.pt",  weights_only=True).bool()
        if pin:
            c_emb, c_msk = c_emb.pin_memory(), c_msk.pin_memory()
        cache_data["_custom"] = (first_ref, c_emb, c_msk)

    neg = None
    if os.path.exists(f"{CACHE_DIR}/_neg_embed.pt"):
        neg = (torch.load(f"{CACHE_DIR}/_neg_embed.pt", weights_only=True),
               torch.load(f"{CACHE_DIR}/_neg_mask.pt",  weights_only=True).bool())

    pos_cache = {}
    def get_pos_ids(text_len, lh, lw):
        key = (text_len, lh, lw)
        if key not in pos_cache:
            pos_cache[key] = prepare_position_ids(text_len, lh // 2, lw // 2, "cuda")
        return pos_cache[key]

    all_preview_names = sorted(k for k in cache_data.keys() if not k.startswith("_"))

    def get_preview_sample(step):
        if PREVIEW_CAPTION_MODE == "custom" and "_custom" in cache_data:
            return "_custom"
        elif PREVIEW_CAPTION_MODE == "random":
            return random.choice(all_preview_names)
        elif PREVIEW_CAPTION_MODE == "rotate4":
            idx = (step // max(1, PREVIEW_EVERY)) % min(4, len(all_preview_names))
            return all_preview_names[idx]
        else:
            return all_preview_names[0]

    running_loss, t_step_avg = 0.0, 0.0
    print(f"\nSTARTING TRAINING / ¡ARRANCANDO ENTRENAMIENTO! {len(all_preview_names)} images in {len(buckets)} buckets.")

    try:
        for step in range(start_step + 1, TOTAL_STEPS + 1):
            last_step_executed = step
            t0 = time.time()

            size = random.choice(list(buckets))
            names = [random.choice(buckets[size]) for _ in range(BATCH_SIZE)]
            latents = torch.cat([cache_data[n][0] for n in names]).to("cuda", non_blocking=True)
            embeds  = torch.cat([cache_data[n][1] for n in names]).to("cuda", non_blocking=True)
            masks   = torch.cat([cache_data[n][2] for n in names]).to("cuda", non_blocking=True)

            latent_patched = pack_latents(latents)
            B, seq_img, _ = latent_patched.shape

            sigma  = sample_sigma(B, seq_img, "cuda", shift_cfg)
            noise  = torch.randn_like(latent_patched)
            t_exp  = sigma.view(-1, 1, 1)

            target_dtype = next(model.parameters()).dtype
            noisy = ((1 - t_exp) * latent_patched + t_exp * noise).to(target_dtype)
            target = noise - latent_patched

            pos_ids = get_pos_ids(embeds.shape[1], size[0], size[1])

            pred = model(
                hidden_states=noisy,
                encoder_hidden_states=embeds,
                timestep=sigma,
                position_ids=pos_ids,
                encoder_attention_mask=masks,
                return_dict=False,
            )[0]

            loss = F.mse_loss(pred.float(), target.float()) / GRAD_ACCUM_STEPS
            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM_STEPS

            grad_norm = 0.0
            if step % GRAD_ACCUM_STEPS == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM).item()
                for gparam in optimizer.param_groups:
                    gparam["lr"] = lr_at(step)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            t_step     = time.time() - t0
            t_step_avg = t_step if t_step_avg == 0 else 0.1 * t_step + 0.9 * t_step_avg
            eta_s      = (TOTAL_STEPS - step) * t_step_avg
            eta        = f"{int(eta_s//3600):02d}:{int((eta_s%3600)//60):02d}:{int(eta_s%60):02d}"
            pct        = step / TOTAL_STEPS
            barra      = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
            
            avg_loss = running_loss / max(1, step - start_step)
            progress_line = (
                f"Step/Paso {step:4d}/{TOTAL_STEPS} [{barra}] {pct*100:5.1f}% | "
                f"Loss {avg_loss:.4f} | gnorm {grad_norm:.3f} | "
                f"lr {lr_at(step):.2e} | {t_step_avg:.2f}s/it | ETA {eta}"
            )
            print(f"\r{progress_line}", end="", flush=True)

            if step % SAVE_EVERY == 0:
                print()
                save_checkpoint_now(step)

            if PREVIEW_EVERY > 0 and step % PREVIEW_EVERY == 0:
                p_name = get_preview_sample(step)
                lat0, emb0, msk0 = cache_data[p_name]
                print(f"\n  [Preview] Mode: {PREVIEW_CAPTION_MODE} | Sample: {p_name}")
                run_preview(model, scheduler, emb0, msk0, neg,
                            (lat0.shape[2] * 8, lat0.shape[3] * 8), step, shift_cfg)

    except (KeyboardInterrupt, SystemExit):
        save_checkpoint_now(last_step_executed)
        return

    print("\n\nTraining completed! / ¡Entrenamiento finalizado!")
    final = os.path.join(OUTPUT_DIR, "Krea2_FINAL_LoRA.safetensors")
    _export_lora(model, final)
    print(f"✓ Final LoRA saved to / Tu LoRA definitivo está en: {final}")


if __name__ == "__main__":
    train_krea2()
