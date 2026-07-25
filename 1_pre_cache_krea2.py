# -*- coding: utf-8 -*-
"""
1_pre_cache_krea2.py — Pre-caché de latentes + embeddings para Krea 2 (RAW)
Pre-cache of latents + embeddings for Krea 2 (RAW)

Lee configuración desde pre_cache_settings.json si existe.
Reads configuration from pre_cache_settings.json if present.
"""
import os
import gc
import math
import json
import torch
import torchvision.transforms.functional as F_vision
from PIL import Image
from diffusers import DiffusionPipeline
import logging
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── DEFAULTS / VALORES POR DEFECTO ──────────────────────────────────────────
DEFAULTS = {
    "model_id": "Krea-2-NF4",
    "dataset_path": "./dataset",
    "cache_dir": "./cached_data_krea2",
    "target_area": 512 * 512,
    "max_side": 1280,
    "multiple": 16,
    "max_seq_len": 128,
    "project_name": "",
    "trigger_word": "",
    "preview_custom_prompt": "",
}

# ── CARGAR CONFIGURACIÓN / LOAD CONFIG ──────────────────────────────────────
CONFIG_PATH = "pre_cache_settings.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"✓ Configuration loaded from {CONFIG_PATH} / Configuración cargada desde {CONFIG_PATH}")
else:
    cfg = {}
    print(f"⚠ {CONFIG_PATH} not found, using defaults / No se encontró {CONFIG_PATH}, usando valores por defecto.")

MODEL_ID     = cfg.get("model_id",     DEFAULTS["model_id"])
DATASET_PATH = cfg.get("dataset_path", DEFAULTS["dataset_path"])
TARGET_AREA  = cfg.get("target_area",  DEFAULTS["target_area"])
MAX_SIDE     = cfg.get("max_side",     DEFAULTS["max_side"])
MULTIPLE     = cfg.get("multiple",     DEFAULTS["multiple"])
MAX_SEQ_LEN  = cfg.get("max_seq_len",  DEFAULTS["max_seq_len"])
TRIGGER_WORD = cfg.get("trigger_word", "")
PROJECT_NAME = cfg.get("project_name", "").strip()
PREVIEW_CUSTOM_PROMPT = cfg.get("preview_custom_prompt", "").strip()

# Formato automático de carpeta de caché según nombre del proyecto
if PROJECT_NAME:
    CACHE_DIR = f"./cached_data_krea2_{PROJECT_NAME}"
else:
    CACHE_DIR = cfg.get("cache_dir", DEFAULTS["cache_dir"])

if MULTIPLE not in (8, 16, 32, 64):
    print(f"⚠ Invalid Multiple {MULTIPLE}. Defaulting to 16 / Múltiplo inválido {MULTIPLE}. Usando 16 por defecto.")
    MULTIPLE = 16

print(f"  Model ID / ID Modelo        : {MODEL_ID}")
print(f"  Project Name / Proyecto     : {PROJECT_NAME if PROJECT_NAME else '(Default)'}")
print(f"  Trigger Word / Palabra      : {TRIGGER_WORD}")
print(f"  Dataset Path / Ruta Dataset : {DATASET_PATH}")
print(f"  Cache Dir / Carpeta Caché   : {CACHE_DIR}")
print(f"  Target Area / Área Objetivo : {TARGET_AREA} px²")
print(f"  Max Side / Lado Máximo      : {MAX_SIDE}")
print(f"  Multiple / Múltiplo         : {MULTIPLE}")
print(f"  Max Seq Len / Long. Sec.    : {MAX_SEQ_LEN}")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def free_vram(*tensors):
    for t in tensors:
        del t
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


def bucket_size(w: int, h: int):
    ar = w / h
    bh = math.sqrt(TARGET_AREA / ar)
    bw = ar * bh
    bw = max(MULTIPLE, round(bw / MULTIPLE) * MULTIPLE)
    bh = max(MULTIPLE, round(bh / MULTIPLE) * MULTIPLE)
    if max(bw, bh) > MAX_SIDE:
        s = MAX_SIDE / max(bw, bh)
        bw = max(MULTIPLE, int(bw * s) // MULTIPLE * MULTIPLE)
        bh = max(MULTIPLE, int(bh * s) // MULTIPLE * MULTIPLE)
    return bw, bh


def ensure_model_downloaded(local_path, repo_id):
    if os.path.exists(local_path) and os.path.isdir(local_path):
        has_content = any(
            os.path.exists(os.path.join(local_path, f))
            for f in ["index.json", "model_index.json", "config.json"]
        ) or len(os.listdir(local_path)) > 0
        if has_content:
            print(f"✓ Local model found at / Modelo local encontrado en: {local_path}")
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
    print(f"✓ Model downloaded to / Modelo descargado en: {downloaded_path}")
    return downloaded_path


def preprocess_krea2():
    if not os.path.exists(DATASET_PATH):
        print(f"[!] Dataset folder does not exist / La carpeta del dataset no existe: {DATASET_PATH}")
        return

    archivos_img = sorted(f for f in os.listdir(DATASET_PATH)
                          if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    if not archivos_img:
        print(f"[!] No images found in '{DATASET_PATH}'. Please add images.")
        return

    ensure_model_downloaded(
        local_path=MODEL_ID,
        repo_id="AcademiaSD/Krea-2-NF4-for-LoRA-Training"
    )

    print("Loading VAE (Qwen-Image) and Text Encoder (Qwen3-VL-4B)... / Cargando VAE y Text Encoder de Krea-2...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        transformer=None,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    if not hasattr(pipe.vae.config, "latents_mean") or not hasattr(pipe.vae.config, "latents_std"):
        raise RuntimeError("This VAE lacks latents_mean/latents_std configuration")

    z_dim = pipe.vae.config.z_dim
    latents_mean = torch.tensor(pipe.vae.config.latents_mean, device="cuda", dtype=torch.float32).view(1, z_dim, 1, 1, 1)
    latents_std  = torch.tensor(pipe.vae.config.latents_std,  device="cuda", dtype=torch.float32).view(1, z_dim, 1, 1, 1)

    with torch.inference_mode():
        neg_embed, neg_mask = pipe.encode_prompt(prompt="", max_sequence_length=MAX_SEQ_LEN)
        torch.save(neg_embed.cpu(), os.path.join(CACHE_DIR, "_neg_embed.pt"))
        torch.save(neg_mask.cpu(),  os.path.join(CACHE_DIR, "_neg_mask.pt"))
        del neg_embed, neg_mask

        # Pre-cache opcional de Prompt Manual si existe
        if PREVIEW_CUSTOM_PROMPT:
            c_prompt = PREVIEW_CUSTOM_PROMPT
            if TRIGGER_WORD and TRIGGER_WORD.lower() not in c_prompt.lower():
                c_prompt = f"{TRIGGER_WORD}, {c_prompt}".strip(", ")
            print(f"[Custom Prompt Cache] Encoding: '{c_prompt}'...")
            c_emb, c_msk = pipe.encode_prompt(prompt=c_prompt, max_sequence_length=MAX_SEQ_LEN)
            torch.save(c_emb.cpu(), os.path.join(CACHE_DIR, "_custom_embed.pt"))
            torch.save(c_msk.cpu(), os.path.join(CACHE_DIR, "_custom_mask.pt"))
            del c_emb, c_msk

        for idx, archivo in enumerate(archivos_img, 1):
            nombre_base = os.path.splitext(archivo)[0]
            ruta_texto  = os.path.join(DATASET_PATH, f"{nombre_base}.txt")
            print(f"[{idx}/{len(archivos_img)}] Processing / Procesando: {archivo}")

            img = Image.open(os.path.join(DATASET_PATH, archivo)).convert("RGB")
            bw, bh = bucket_size(*img.size)

            ratio = max(bw / img.width, bh / img.height)
            img = img.resize((math.ceil(img.width * ratio), math.ceil(img.height * ratio)), Image.LANCZOS)
            left = (img.width - bw) // 2
            top  = (img.height - bh) // 2
            img  = img.crop((left, top, left + bw, top + bh))

            img_tensor = F_vision.pil_to_tensor(img).unsqueeze(0).unsqueeze(2)
            img_tensor = (img_tensor.float() / 127.5) - 1.0
            img_tensor = img_tensor.to("cuda", dtype=torch.bfloat16)

            z = pipe.vae.encode(img_tensor).latent_dist.sample().float()
            latent = (z - latents_mean) / latents_std
            latent = latent[:, :, 0].to(torch.bfloat16)

            torch.save(latent.cpu(), os.path.join(CACHE_DIR, f"{nombre_base}_latent.pt"))
            free_vram(img_tensor, z, latent)

            prompt = ""
            if os.path.exists(ruta_texto):
                with open(ruta_texto, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()

            if TRIGGER_WORD and TRIGGER_WORD.lower() not in prompt.lower():
                prompt = f"{TRIGGER_WORD}, {prompt}".strip(", ")

            embeds, mask = pipe.encode_prompt(prompt=prompt, max_sequence_length=MAX_SEQ_LEN)

            torch.save(embeds.cpu(), os.path.join(CACHE_DIR, f"{nombre_base}_embed.pt"))
            torch.save(mask.cpu(),   os.path.join(CACHE_DIR, f"{nombre_base}_mask.pt"))

            print(f"   [OK] {bw}x{bh} | Latents & Embeddings cached.")

    free_vram(pipe)
    print("\n✓ Pre-caching finished! VRAM freed / ¡Pre-caché finalizado! VRAM liberada.")


if __name__ == "__main__":
    preprocess_krea2()