# -*- coding: utf-8 -*-
"""
4_convert_to_nf4.py

Convierte el Transformer de Krea-2-Raw a NF4 y guarda una caché
reutilizable en:

    ./Krea-2-NF4/

La conversión pesada se ejecuta UNA SOLA VEZ.

El script guarda:
    - configuración del Transformer
    - pesos NF4 de las capas cuantizadas
    - QuantState serializado con packed=True
    - bias de las capas cuantizadas
    - pesos BF16 de las capas excluidas de la cuantización
    - metadata de la conversión

Posteriormente el trainer podrá reconstruir el Transformer
sin volver a ejecutar la cuantización NF4.

IMPORTANTE:
La cuantización se realiza en GPU porque bitsandbytes necesita
materializar el QuantState real (bnb_quantized=True).
"""

import os
import gc
import json
import time

import torch
from diffusers import DiffusionPipeline
from safetensors.torch import save_file


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

MODEL_ID = "./Krea-2-Raw"
OUTPUT_DIR = "./Krea-2-NF4"

DTYPE = torch.bfloat16

# Debe coincidir EXACTAMENTE con el trainer.
SKIP_QUANT = (
    "img_in",
    "time_embed",
    "time_mod_proj",
    "txt_in",
    "final_layer",
)


# ============================================================================
# UTILIDADES
# ============================================================================

def free_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def json_safe(value):
    """
    Convierte objetos de configuración de torch a tipos serializables
    por JSON.
    """

    if isinstance(value, torch.dtype):
        return str(value)

    if isinstance(value, torch.Size):
        return list(value)

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


# ============================================================================
# CUANTIZACIÓN
# ============================================================================

def quantize_to_nf4_(module, prefix=""):
    """
    Recorre recursivamente el Transformer y reemplaza Linear por Linear4bit.

    IMPORTANTE:
    No guardamos aquí.

    Esta función únicamente realiza la conversión.
    """

    from bitsandbytes.nn import Linear4bit, Params4bit

    for name, child in list(module.named_children()):

        full = f"{prefix}.{name}" if prefix else name

        if (
            isinstance(child, torch.nn.Linear)
            and not any(s in full for s in SKIP_QUANT)
        ):

            print(f"  NF4 -> {full}")

            # Copia FP32 temporal.
            # Es el mismo procedimiento utilizado por el trainer.
            w = child.weight.data.float().contiguous()

            new_layer = Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                quant_type="nf4",
                compute_dtype=torch.bfloat16,
            )

            new_layer.weight = Params4bit(
                w,
                requires_grad=False,
                quant_type="nf4",
            )

            if child.bias is not None:
                new_layer.bias = torch.nn.Parameter(
                    child.bias.data.clone(),
                    requires_grad=False,
                )

            setattr(module, name, new_layer)

            del child
            del w

        else:
            quantize_to_nf4_(child, full)


# ============================================================================
# EXTRACCIÓN DE LA CACHÉ NF4
# ============================================================================

def extract_nf4_cache(transformer, output_dir):

    from bitsandbytes.nn import Linear4bit

    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    index = {
        "quantized": {},
        "unquantized": {},
    }

    quantized_count = 0
    unquantized_count = 0

    print()
    print("=" * 70)
    print("EXTRAYENDO CACHÉ NF4")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Recorremos TODAS las capas
    # ------------------------------------------------------------------------

    for name, layer in transformer.named_modules():

        # ------------------------------------------------------------
        # CAPAS NF4
        # ------------------------------------------------------------

        if isinstance(layer, Linear4bit):

            weight = layer.weight

            if not getattr(weight, "bnb_quantized", False):

                print(
                    f"[AVISO] {name}: Linear4bit todavía no está "
                    f"marcada como bnb_quantized=True"
                )

                continue

            qs = weight.quant_state

            if qs is None:

                print(
                    f"[AVISO] {name}: quant_state es None"
                )

                continue

            # --------------------------------------------------------
            # Serialización oficial de QuantState
            # --------------------------------------------------------

            qs_dict = qs.as_dict(packed=True)

            # --------------------------------------------------------
            # Creamos un nombre seguro para el archivo
            # --------------------------------------------------------

            safe_name = name.replace(".", "__")

            filename = safe_name + ".safetensors"

            filepath = os.path.join(
                weights_dir,
                filename,
            )

            tensors = {}

            # Peso NF4 real.
            #
            # En tu prueba:
            #
            # dtype = torch.uint8
            #
            # y shape = [N, 1]
            #
            tensors["weight"] = (
                weight.data.detach()
                .cpu()
                .contiguous()
            )

            # Bias, si existe.
            if layer.bias is not None:

                tensors["bias"] = (
                    layer.bias.detach()
                    .to(torch.bfloat16)
                    .cpu()
                    .contiguous()
                )

            # --------------------------------------------------------
            # QuantState packed
            # --------------------------------------------------------

            metadata = {
                "layer_name": name,
                "type": "Linear4bit",
                "quant_type": "nf4",
                "bnb_quantized": "true",
                "in_features": str(layer.in_features),
                "out_features": str(layer.out_features),
                "bias": str(layer.bias is not None),
            }

            # Los tensores contenidos dentro de QuantState
            # se guardan junto con el peso.
            #
            # as_dict(packed=True) puede devolver:
            #
            # absmax
            # quant_map
            # nested_absmax
            # nested_quant_map
            # quant_state.bitsandbytes__nf4
            #
            # Los elementos tensoriales se agregan directamente
            # al safetensors.

            packed_state = {}

            for key, value in qs_dict.items():

                if torch.is_tensor(value):

                    # Evitamos colisión con "weight" y "bias".
                    tensor_key = "quant_state." + key

                    packed_state[tensor_key] = (
                        value.detach()
                        .cpu()
                        .contiguous()
                    )

                else:

                    # Los valores no tensoriales se almacenan
                    # como metadata JSON.
                    metadata[
                        "qs_" + key
                    ] = json.dumps(
                        json_safe(value)
                    )

            tensors.update(packed_state)

            save_file(
                tensors,
                filepath,
                metadata=metadata,
            )

            index["quantized"][name] = {
                "file": filename,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "bias": layer.bias is not None,
                "quant_type": "nf4",
                "compute_dtype": "bfloat16",
                "quant_state_keys": list(qs_dict.keys()),
            }

            quantized_count += 1

            if quantized_count % 25 == 0:

                print(
                    f"  Guardadas {quantized_count} capas NF4..."
                )

        # ------------------------------------------------------------
        # CAPAS NO CUANTIZADAS
        # ------------------------------------------------------------

        elif isinstance(layer, torch.nn.Linear):

            # Solo guardamos aquí las Linear que están excluidas
            # explícitamente de la cuantización.
            #
            # Estas son:
            #
            # img_in
            # time_embed
            # time_mod_proj
            # txt_in
            # final_layer

            if any(
                s in name
                for s in SKIP_QUANT
            ):

                safe_name = name.replace(".", "__")

                filename = (
                    safe_name
                    + ".safetensors"
                )

                filepath = os.path.join(
                    weights_dir,
                    filename,
                )

                tensors = {
                    "weight": (
                        layer.weight.detach()
                        .to(torch.bfloat16)
                        .cpu()
                        .contiguous()
                    )
                }

                if layer.bias is not None:

                    tensors["bias"] = (
                        layer.bias.detach()
                        .to(torch.bfloat16)
                        .cpu()
                        .contiguous()
                    )

                save_file(
                    tensors,
                    filepath,
                    metadata={
                        "layer_name": name,
                        "type": "Linear",
                        "quantized": "false",
                    },
                )

                index["unquantized"][name] = {
                    "file": filename,
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "bias": layer.bias is not None,
                }

                unquantized_count += 1

    # =========================================================================
    # GUARDAR INDEX
    # =========================================================================

    index_path = os.path.join(
        output_dir,
        "index.json",
    )

    with open(
        index_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            index,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return (
        quantized_count,
        unquantized_count,
        index,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    start_time = time.time()

    print()
    print("=" * 70)
    print(" KREA-2 NF4 CACHE BUILDER")
    print("=" * 70)

    print()
    print("Modelo:", MODEL_ID)
    print("Salida:", OUTPUT_DIR)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA no está disponible. "
            "La cuantización NF4 requiere CUDA."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print()

    # ------------------------------------------------------------------------
    # Crear directorio
    # ------------------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Evitar sobrescribir accidentalmente una caché existente
    # ------------------------------------------------------------------------

    index_path = os.path.join(
        OUTPUT_DIR,
        "index.json",
    )

    if os.path.exists(index_path):

        print()
        print(
            "[AVISO] Ya existe una caché NF4:"
        )

        print(
            " ",
            OUTPUT_DIR,
        )

        respuesta = input(
            "\n¿Quieres sobrescribirla? [s/N]: "
        ).strip().lower()

        if respuesta not in ("s", "si", "sí", "y", "yes"):

            print(
                "\nConversión cancelada."
            )

            return

    # ------------------------------------------------------------------------
    # CARGAR MODELO
    # ------------------------------------------------------------------------

    print()
    print(
        "Cargando Transformer de Krea-2-Raw..."
    )

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        vae=None,
        text_encoder=None,
        torch_dtype=DTYPE,
    )

    transformer = pipe.transformer

    # El scheduler no se modifica, pero guardamos su config
    # por comodidad para el futuro loader.

    scheduler = pipe.scheduler

    del pipe

    free_memory()

    print(
        "Transformer cargado."
    )

    print(
        "Parámetros:",
        sum(
            p.numel()
            for p in transformer.parameters()
        ),
    )

    # ------------------------------------------------------------------------
    # CUANTIZAR
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "CUANTIZANDO TRANSFORMER A NF4"
    )
    print(
        "(excepto img_in/time/final/txt_in)"
    )
    print("=" * 70)

    quantize_to_nf4_(
        transformer
    )

    print()
    print(
        "Conversión estructural completada."
    )

    # ------------------------------------------------------------------------
    # MOVER A CUDA
    # ------------------------------------------------------------------------

    print()
    print(
        "Moviendo Transformer a CUDA..."
    )

    transformer.to("cuda")

    free_memory()

    print(
        "Transformer en GPU."
    )

    print(
        "VRAM:",
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB",
    )

    # ------------------------------------------------------------------------
    # FORZAR / VERIFICAR CUANTIZACIÓN REAL
    # ------------------------------------------------------------------------

    print()
    print(
        "Verificando QuantState NF4..."
    )

    from bitsandbytes.nn import Linear4bit

    verified = 0

    for name, layer in transformer.named_modules():

        if isinstance(layer, Linear4bit):

            if getattr(
                layer.weight,
                "bnb_quantized",
                False,
            ):

                if layer.weight.quant_state is not None:

                    verified += 1

    print(
        "Capas Linear4bit verificadas:",
        verified,
    )

    if verified == 0:

        raise RuntimeError(
            "No se encontró ninguna Linear4bit "
            "con bnb_quantized=True y QuantState válido."
        )

    # ------------------------------------------------------------------------
    # GUARDAR CONFIG DEL TRANSFORMER
    # ------------------------------------------------------------------------

    print()
    print(
        "Guardando configuración..."
    )

    config_path = os.path.join(
        OUTPUT_DIR,
        "config.json",
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_safe(
                dict(transformer.config)
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------------
    # GUARDAR CONFIG DEL SCHEDULER
    # ------------------------------------------------------------------------

    scheduler_path = os.path.join(
        OUTPUT_DIR,
        "scheduler_config.json",
    )

    with open(
        scheduler_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_safe(
                dict(scheduler.config)
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------------
    # EXTRAER CACHÉ
    # ------------------------------------------------------------------------

    quantized_count, unquantized_count, index = (
        extract_nf4_cache(
            transformer,
            OUTPUT_DIR,
        )
    )

    # ------------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------------

    metadata = {

        "format": "Krea2-NF4",

        "version": 1,

        "model_id": MODEL_ID,

        "dtype": "bfloat16",

        "quant_type": "nf4",

        "compute_dtype": "bfloat16",

        "quantized_layers": quantized_count,

        "unquantized_layers": unquantized_count,

        "skip_quant": list(
            SKIP_QUANT
        ),

        "bitsandbytes_prequantized": True,

        "quant_state_serialization": "as_dict(packed=True)",

        "loader": "Params4bit.from_prequantized",

    }

    metadata_path = os.path.join(
        OUTPUT_DIR,
        "metadata.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------------
    # RESUMEN
    # ------------------------------------------------------------------------

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("CONVERSIÓN COMPLETADA")
    print("=" * 70)

    print(
        "Capas NF4:",
        quantized_count,
    )

    print(
        "Capas BF16:",
        unquantized_count,
    )

    print(
        "VRAM final:",
        f"{torch.cuda.memory_allocated()/1e9:.2f} GB",
    )

    print(
        "Tiempo:",
        f"{elapsed / 60:.1f} minutos",
    )

    print()
    print(
        "Caché creada en:"
    )

    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )

    print()
    print(
        "Archivos principales:"
    )

    print(
        "  config.json"
    )

    print(
        "  scheduler_config.json"
    )

    print(
        "  metadata.json"
    )

    print(
        "  index.json"
    )

    print(
        "  weights/"
    )

    print()
    print(
        "IMPORTANTE:"
    )

    print(
        "No borres Krea-2-Raw todavía."
    )

    print(
        "Primero debemos probar la reconstrucción"
    )

    print(
        "de la caché NF4 y comparar el modelo."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()