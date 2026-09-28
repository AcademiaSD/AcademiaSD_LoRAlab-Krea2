# -*- coding: utf-8 -*-
"""
0_caption_krea2.py — Captions automáticos del dataset con Qwen3-VL-4B
Dataset auto-captioning with Qwen3-VL-4B

Un .txt por imagen con una descripción. La palabra trigger va siempre al principio.
One .txt per image with a description. The trigger word always goes first.

No descarga ningún modelo nuevo: el text encoder de Krea-2 es Qwen3-VL-4B-Instruct
con lm_head atado a los embeddings, así que puede escribir texto. Se cuantiza a NF4
al cargar (~3.5 GB de VRAM, mismas descripciones que en BF16).

Lee caption_settings.json; la ruta del dataset y el trigger salen de
pre_cache_settings.json.
"""
import os
import sys
import json
import time
import warnings

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# NF4: algunas capas de visión no son múltiplo de 64 y usan el kernel general; solo cambia la velocidad.
warnings.filterwarnings("ignore", message=".*is not aligned for fast kernel.*")

DEFAULTS = {
    "model_id": "Krea-2-NF4",
    "dataset_path": "./dataset",
    "trigger_word": "",
    "caption_prompt": (
        "Describe this image for training a text-to-image model, in at most 150 words. "
        "One single paragraph of plain prose, no lists and no preamble. Cover the main "
        "subject and what they are doing, their appearance, hair and clothing, the pose, "
        "the framing and camera angle, the setting and background, the lighting and the "
        "colors. Do not invent a name and do not mention that this is an image or a photo."
    ),
    # 150 palabras son ~200 tokens; el margen evita captions cortados a media frase.
    "max_new_tokens": 300,
    "temperature": 0.3,
    # Lado largo de la copia EN MEMORIA que ve el modelo; el fichero del dataset no se toca.
    # Más píxeles = más tokens de visión y un prefill más lento. 0 = sin reescalar.
    "max_image_side": 768,
    # False = no toca las imágenes que ya tienen un .txt con contenido.
    "overwrite": True,
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

PRECACHE_CONFIG = "pre_cache_settings.json"
CONFIG_PATH = "caption_settings.json"
HF_REPO_ID = "AcademiaSD/Krea-2-NF4-for-LoRA-Training"


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_config():
    cfg = dict(DEFAULTS)
    precache = read_json(PRECACHE_CONFIG)
    for key in ("model_id", "dataset_path", "trigger_word"):
        if precache.get(key):
            cfg[key] = precache[key]
    cfg.update({k: v for k, v in read_json(CONFIG_PATH).items() if v != ""})
    return cfg


def load_captioner(model_id):
    path = os.path.join(model_id, "text_encoder")
    processor_path = os.path.join(model_id, "processor")
    if not os.path.exists(os.path.join(path, "config.json")) or not os.path.exists(os.path.join(processor_path, "preprocessor_config.json")):
        # Instalación limpia: solo hace falta el text encoder y el processor (~9 GB), no el modelo entero.
        print(f"Downloading captioner from Hugging Face / Descargando el captioner desde Hugging Face: {HF_REPO_ID}", flush=True)
        token = read_json("HF_token.json").get("token", "").strip() or None
        snapshot_download(repo_id=HF_REPO_ID, local_dir=model_id, token=token, max_workers=2,
                          allow_patterns=["text_encoder/*", "processor/*"])

    print("Loading Qwen3-VL-4B (NF4)... / Cargando Qwen3-VL-4B (NF4)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda", quantization_config=quant)
    model.eval()
    processor = AutoProcessor.from_pretrained(processor_path)
    print(f"Ready / Listo. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    return model, processor


def shrink(image, max_side):
    if max_side <= 0 or max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)


def clean_caption(text):
    text = " ".join(text.split())
    for prefix in ("The image shows ", "This image shows ", "The image depicts ", "This image depicts ",
                   "The photo shows ", "In this image, ", "In the image, ", "Sure, ", "Certainly, ", "Here is a "):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            text = text[:1].upper() + text[1:]
            break
    return text.strip().strip('"')


def caption_image(model, processor, image, prompt, cfg):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

    temperature = float(cfg["temperature"])
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=int(cfg["max_new_tokens"]),
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-4),
        )
    return clean_caption(processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


def with_trigger(trigger, caption):
    trigger = trigger.strip().rstrip(",")
    if not trigger or caption.lower().startswith(trigger.lower()):
        return caption
    return f"{trigger}, {caption}"


def main():
    cfg = load_config()
    dataset = cfg["dataset_path"]

    if not os.path.isdir(dataset):
        print(f"[!] Dataset folder does not exist / La carpeta del dataset no existe: {dataset}")
        return 1

    images = sorted(f for f in os.listdir(dataset) if f.lower().endswith(IMAGE_EXTS))
    pending = []
    for name in images:
        txt = os.path.join(dataset, os.path.splitext(name)[0] + ".txt")
        if not cfg["overwrite"] and os.path.isfile(txt) and open(txt, encoding="utf-8").read().strip():
            continue
        pending.append(name)

    print(f"  Dataset / Ruta Dataset : {os.path.abspath(dataset)}")
    print(f"  Images / Imágenes      : {len(images)}, {len(pending)} to caption / por describir")
    print(f"  Trigger Word / Palabra : {cfg['trigger_word'] or '(none / ninguna)'}")
    print(f"  Overwrite / Rehacer    : {'yes / sí' if cfg['overwrite'] else 'only missing / solo los que faltan'}")

    if not pending:
        print("Nothing to do: every image already has a caption. / Nada que hacer: todas las imágenes ya tienen caption.")
        return 0

    model, processor = load_captioner(cfg["model_id"])

    started = time.time()
    done, failed = 0, []
    for i, name in enumerate(pending, 1):
        t0 = time.time()
        stem = os.path.splitext(name)[0]
        try:
            with Image.open(os.path.join(dataset, name)) as img:
                image = shrink(img.convert("RGB"), int(cfg["max_image_side"]))
            caption = caption_image(model, processor, image, cfg["caption_prompt"], cfg)
            if not caption:
                raise RuntimeError("empty caption / caption vacío")

            final = with_trigger(cfg["trigger_word"], caption)
            with open(os.path.join(dataset, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write(final)
            done += 1

            eta = (time.time() - started) / i * (len(pending) - i)
            print(f"[{i}/{len(pending)}] {name} | {time.time() - t0:.1f}s | ETA {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)
            print(f"    {final[:200]}{'...' if len(final) > 200 else ''}", flush=True)
        except Exception as e:
            failed.append(name)
            print(f"[{i}/{len(pending)}] {name} FAILED / FALLÓ: {e}", flush=True)

    total = time.time() - started
    print(f"\n✓ {done} captions in / en {int(total // 60)}m{int(total % 60):02d}s ({total / max(done, 1):.1f}s per image / por imagen).")
    if failed:
        print(f"[!] {len(failed)} failed / fallaron: {', '.join(failed[:10])}")
    print("Review them in the Dataset Manager before pre-caching. / Revísalos en el Dataset Manager antes del pre-caché.")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
