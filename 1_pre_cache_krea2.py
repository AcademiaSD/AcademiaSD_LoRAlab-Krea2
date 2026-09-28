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
from diffusers import DiffusionPipeline, AutoencoderKLQwenImage
from transformers import Qwen3VLModel
import logging
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HF_REPO_ID = "AcademiaSD/Krea-2-NF4-for-LoRA-Training"

# VRAM que se deja libre al repartir el text encoder entre GPU y RAM (activaciones, contexto CUDA).
OFFLOAD_RESERVE_BYTES = int(1.5 * 1024**3)

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
    if (os.path.exists(os.path.join(local_path, "model_index.json"))
            and os.path.exists(os.path.join(local_path, "text_encoder", "config.json"))
            and os.path.exists(os.path.join(local_path, "vae", "config.json"))):
        print(f"✓ Local model found at / Modelo local encontrado en: {local_path}")
        return local_path

    print(f"⚠ Local model not found at / No se encontró modelo local en: {local_path}")
    print(f"  Downloading from Hugging Face / Descargando desde Hugging Face: {repo_id}")

    enable_hf_file_progress()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with pip install huggingface_hub")

    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        token=get_hf_token(),
        max_workers=2,
    )
    print(f"✓ Model downloaded to / Modelo descargado en: {downloaded_path}")
    return downloaded_path


def with_trigger(prompt):
    if TRIGGER_WORD and TRIGGER_WORD.lower() not in prompt.lower():
        prompt = f"{TRIGGER_WORD}, {prompt}".strip(", ")
    return prompt


def load_text_encoder(device):
    path = os.path.join(MODEL_ID, "text_encoder")
    if device == "cpu":
        return Qwen3VLModel.from_pretrained(path, dtype=torch.bfloat16, device_map="cpu")

    # BF16 exacto en cualquier GPU: las capas que no caben en la VRAM libre se quedan en RAM y se ejecutan desde allí.
    free_bytes, _ = torch.cuda.mem_get_info()
    gpu_budget = max(free_bytes - OFFLOAD_RESERVE_BYTES, 0)
    print(f"  GPU budget / Presupuesto GPU: {gpu_budget / 1024**3:.1f} GB (rest in RAM / resto en RAM)")
    return Qwen3VLModel.from_pretrained(path, dtype=torch.bfloat16, device_map="auto",
                                        max_memory={0: gpu_budget, "cpu": "512GiB"})


def load_text_pipe(device="cuda"):
    return DiffusionPipeline.from_pretrained(MODEL_ID, transformer=None, vae=None,
                                             text_encoder=load_text_encoder(device), torch_dtype=torch.bfloat16)


def encode(pipe, prompt, device="cuda"):
    with torch.inference_mode():
        return pipe.encode_prompt(prompt=prompt, max_sequence_length=MAX_SEQ_LEN, device=device)


def encode_and_save(pipe, prompt, name):
    embeds, mask = encode(pipe, prompt)
    torch.save(embeds.cpu(), os.path.join(CACHE_DIR, f"{name}_embed.pt"))
    torch.save(mask.cpu(), os.path.join(CACHE_DIR, f"{name}_mask.pt"))


def read_caption(name):
    path = os.path.join(DATASET_PATH, f"{name}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def encode_latents(vae, jobs):
    z_dim = vae.config.z_dim
    latents_mean = torch.tensor(vae.config.latents_mean, device="cuda", dtype=torch.float32).view(1, z_dim, 1, 1, 1)
    latents_std  = torch.tensor(vae.config.latents_std,  device="cuda", dtype=torch.float32).view(1, z_dim, 1, 1, 1)

    with torch.inference_mode():
        for idx, (src, out, (bw, bh)) in enumerate(jobs, 1):
            img = Image.open(src).convert("RGB")
            ratio = max(bw / img.width, bh / img.height)
            img = img.resize((math.ceil(img.width * ratio), math.ceil(img.height * ratio)), Image.LANCZOS)
            left, top = (img.width - bw) // 2, (img.height - bh) // 2
            img = img.crop((left, top, left + bw, top + bh))

            img_tensor = F_vision.pil_to_tensor(img).unsqueeze(0).unsqueeze(2)
            img_tensor = (img_tensor.float() / 127.5) - 1.0
            img_tensor = img_tensor.to("cuda", dtype=torch.bfloat16)

            z = vae.encode(img_tensor).latent_dist.sample().float()
            latent = ((z - latents_mean) / latents_std)[:, :, 0].to(torch.bfloat16)

            # Escritura atómica: el trainer puede estar leyendo la caché.
            tmp = os.path.join(CACHE_DIR, out + ".tmp")
            torch.save(latent.cpu(), tmp)
            os.replace(tmp, os.path.join(CACHE_DIR, out))
            del img_tensor, z, latent

            print(f"[{idx}/{len(jobs)}] Image / Imagen: {os.path.basename(src)} -> {out} | {bw}x{bh}", flush=True)


def preprocess_krea2():
    if not os.path.exists(DATASET_PATH):
        print(f"[!] Dataset folder does not exist / La carpeta del dataset no existe: {DATASET_PATH}")
        sys.exit(1)

    archivos_img = sorted(f for f in os.listdir(DATASET_PATH)
                          if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    if not archivos_img:
        print(f"[!] No images found in '{DATASET_PATH}'. Please add images.")
        sys.exit(1)
    samples = [(os.path.splitext(f)[0], os.path.join(DATASET_PATH, f)) for f in archivos_img]

    # El trainer carga todo lo que haya en la caché: se quitan las muestras que ya no están en el dataset.
    expected = {f"{name}_{suffix}.pt" for name, _ in samples for suffix in ("latent", "embed", "mask")}
    stale = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pt") and not f.startswith("_") and f not in expected]
    for f in stale:
        os.remove(os.path.join(CACHE_DIR, f))
    if stale:
        print(f"  Removed {len(stale)} stale cache files / Eliminados {len(stale)} ficheros antiguos de la caché")

    ensure_model_downloaded(local_path=MODEL_ID, repo_id=HF_REPO_ID)

    # ── FASE 1: TEXT ENCODER (Qwen3-VL-4B) ──────────────────────────────────
    # El text encoder y el VAE nunca coinciden en VRAM.
    print("\nLoading Text Encoder (Qwen3-VL-4B)... / Cargando Text Encoder (Qwen3-VL-4B)...")
    pipe = load_text_pipe()

    encode_and_save(pipe, "", "_neg")

    if PREVIEW_CUSTOM_PROMPT:
        c_prompt = with_trigger(PREVIEW_CUSTOM_PROMPT)
        print(f"[Custom Prompt Cache] Encoding: '{c_prompt}'")
        encode_and_save(pipe, c_prompt, "_custom")
        # El trainer y el servidor comparan con este texto para saber si la caché está al día.
        with open(os.path.join(CACHE_DIR, "_custom_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(c_prompt)
    else:
        for f in ("_custom_embed.pt", "_custom_mask.pt", "_custom_prompt.txt"):
            if os.path.exists(os.path.join(CACHE_DIR, f)):
                os.remove(os.path.join(CACHE_DIR, f))

    for idx, (name, _) in enumerate(samples, 1):
        encode_and_save(pipe, with_trigger(read_caption(name)), name)
        print(f"[{idx}/{len(samples)}] Text / Texto: {name}")

    del pipe
    free_vram()

    # ── FASE 2: VAE (Qwen-Image) ────────────────────────────────────────────
    # Un latente ya cacheado con el tamaño de bucket actual y más nuevo que su imagen no se
    # vuelve a codificar: relanzar el pre-caché para cambiar el prompt manual o los captions
    # solo cuesta la fase de texto.
    pending = []
    for name, src in samples:
        with Image.open(src) as im:
            bw, bh = bucket_size(*im.size)
        lat_path = os.path.join(CACHE_DIR, f"{name}_latent.pt")
        if (os.path.exists(lat_path) and os.path.getmtime(lat_path) > os.path.getmtime(src)
                and tuple(torch.load(lat_path, weights_only=True).shape[-2:]) == (bh // 8, bw // 8)):
            continue
        pending.append((src, f"{name}_latent.pt", (bw, bh)))

    if len(pending) < len(samples):
        print(f"\n{len(samples) - len(pending)} latents already cached, skipped / latentes ya cacheados, se saltan.")
    if not pending:
        print("\n✓ Pre-caching finished! / ¡Pre-caché finalizado!")
        return

    print("\nLoading VAE (Qwen-Image)... / Cargando VAE (Qwen-Image)...")
    vae = AutoencoderKLQwenImage.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
    encode_latents(vae, pending)
    del vae
    free_vram()
    print("\n✓ Pre-caching finished! VRAM freed / ¡Pre-caché finalizado! VRAM liberada.")


def encode_preview_prompt(cache_dir, prompt, device):
    """
    Codifica solo el prompt manual de las previews. Lo lanza el servidor al guardar un prompt
    nuevo: sin entrenamiento en marcha va en GPU; con el entrenamiento en marcha va en CPU para
    no tocar su VRAM, y el trainer relee el embedding antes de la siguiente preview.
    """
    where = device.upper().replace("CUDA", "GPU")
    # El servidor lee la línea "[Custom Prompt] Prompt" para mostrar la fase en la GUI.
    print(f"[Custom Prompt] Prompt: encoding on {where} / Codificando en {where}: '{prompt}'", flush=True)
    pipe = load_text_pipe(device)
    embeds, mask = encode(pipe, prompt, device)
    del pipe
    free_vram()

    # Escrituras atómicas; el embedding el último, porque es el fichero que vigila el trainer.
    def replace(name, write):
        tmp = os.path.join(cache_dir, name + ".tmp")
        write(tmp)
        os.replace(tmp, os.path.join(cache_dir, name))

    replace("_custom_mask.pt", lambda f: torch.save(mask.cpu(), f))
    replace("_custom_prompt.txt", lambda f: open(f, "w", encoding="utf-8").write(prompt))
    replace("_custom_embed.pt", lambda f: torch.save(embeds.cpu(), f))
    print("[Custom Prompt] Ready / Listo.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--prompt-only":
        # --prompt-only <carpeta de caché> <prompt con el trigger ya puesto> <cuda|cpu>
        encode_preview_prompt(*sys.argv[2:])
    else:
        preprocess_krea2()
