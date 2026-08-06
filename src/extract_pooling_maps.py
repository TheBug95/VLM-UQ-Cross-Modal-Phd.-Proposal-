"""src/extract_pooling_maps.py — Extrae los mapas de pesos (16×16) de las 8 técnicas de pooling.

El CSV principal (`results_full.csv`) solo guarda los valores escalares ya
agregados de cada variante; los pesos por token se calculan en memoria durante
la inferencia y se descartan. Este script re-corre la pasada single-pass sobre
las 129 imágenes y guarda, para cada imagen y cada técnica de pooling, el vector
de 256 pesos (uno por token visual), que el dashboard renderiza como heatmap
16×16 superpuesto al fundus.

Salida: ``results/pooling_maps.csv`` en formato largo:
    image_filename, prompt_id, pooling, token_idx (0–255), weight

Mapas extraídos (capa 34, misma pasada):
    mean     uniforme (1/256) — referencia
    max      frecuencia con la que cada token es el argmax por dimensión (ganadora)
    topk     1/k en los k tokens de mayor norma L2, 0 en el resto
    normw    norma L2 normalizada
    attn     atención cruzada del último token (capa 34, media de cabezas)
    rollout  attention rollout (Abnar & Zuidema 2020)
    headspec media de las 4 cabezas más "visuales" (capa 34)
    roi      máscara de disco promediada por celda (oracle; solo 69 patológicas)

Requiere GPU + HF_TOKEN + licencia HAI-DEF. En Colab T4 (~4.3 s/imagen):
~10 min por prompt. Requisito: el dataset completo descargado (las máscaras
`*_disc.png` son necesarias para el mapa `roi`) — `python -m src.data` lo
garantiza. Desde el notebook de Colab habitual:

    !python -m src.extract_pooling_maps --prompt P1 --n 3   # prueba rápida
    !python -m src.extract_pooling_maps --prompt P1         # corrida completa
    !python -m src.extract_pooling_maps --prompt both       # P1 + P4
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import get_config
from src.inference import MedGemmaInference, load_image
from src.uncertainty import (
    attention_rollout,
    compute_attention_weights,
    compute_roi_weights,
    head_specific_attention,
)

POOLINGS = ["mean", "max", "topk", "normw", "attn", "rollout", "headspec", "roi"]
LAYER = 34  # capa ganadora; las mapas no dependen de τ


def pooling_maps_for_image(
    pipe: MedGemmaInference,
    image,
    prompt_id: str,
    image_filename: str,
    split: str,
) -> dict[str, np.ndarray]:
    """Devuelve {pooling: array(256,)} con los pesos por token visual."""
    prompt = pipe.build_prompt(prompt_id.lower())  # config.yaml usa claves p1/p4
    inputs = pipe.processor(images=image, text=prompt, return_tensors="pt").to(pipe.device)

    with torch.inference_mode():
        out = pipe.model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            output_hidden_states=True,
            output_attentions=True,   # requerido para attn/rollout/headspec
            return_dict_in_generate=True,
        )

    hs = out.hidden_states[0]  # prefill
    input_ids = inputs["input_ids"][0]
    img_positions = (input_ids == pipe.cfg.tokens.image_token_index).nonzero().flatten()
    n_tokens = len(img_positions)
    if n_tokens != pipe.cfg.inference.num_image_tokens:
        raise RuntimeError(
            f"{image_filename}: se esperaban {pipe.cfg.inference.num_image_tokens} "
            f"tokens de imagen, se encontraron {n_tokens}"
        )

    h_img = hs[LAYER][0][img_positions, :].float()  # (256, 2560)
    maps: dict[str, np.ndarray] = {}

    # mean: uniforme (referencia)
    maps["mean"] = np.full(n_tokens, 1.0 / n_tokens, dtype=np.float32)

    # max: frecuencia con la que cada token gana el argmax por dimensión
    argmax_tokens = h_img.argmax(dim=0)  # (2560,)
    maps["max"] = (
        torch.bincount(argmax_tokens, minlength=n_tokens).cpu().numpy() / h_img.shape[1]
    ).astype(np.float32)

    # topk: 1/k en los k tokens de mayor norma (misma k que src.inference)
    k = max(1, n_tokens // 10)
    topk_idx = h_img.norm(dim=1).topk(k).indices.cpu().numpy()
    topk_map = np.zeros(n_tokens, dtype=np.float32)
    topk_map[topk_idx] = 1.0 / k
    maps["topk"] = topk_map

    # normw: norma L2 normalizada
    norms = h_img.norm(dim=1).cpu().numpy()
    maps["normw"] = (norms / norms.sum()).astype(np.float32)

    # attn / rollout / headspec: requieren atenciones (eager)
    attentions = out.attentions
    seq_len = hs[LAYER].shape[1]
    maps["attn"] = compute_attention_weights(attentions, LAYER, img_positions, seq_len)
    maps["rollout"] = attention_rollout(attentions, img_positions, end_layer=LAYER)
    maps["headspec"] = head_specific_attention(attentions, LAYER, img_positions)

    # roi: oracle (None si no hay máscara → no se guardan filas)
    roi = compute_roi_weights(
        pipe.cfg, image_filename, split,
        grid_size=int(np.sqrt(pipe.cfg.inference.num_image_tokens)),
    )
    if roi is not None:
        maps["roi"] = roi.astype(np.float32)

    return maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae mapas de pesos de los 8 poolings")
    parser.add_argument("--prompt", default="P1", choices=["P1", "P4", "both"])
    parser.add_argument("--n", type=int, default=None, help="Limitar a N imágenes (prueba)")
    parser.add_argument("--out", default="pooling_maps.csv", help="Nombre del CSV en results/")
    args = parser.parse_args()

    cfg = get_config(resolve_ids=True)
    master = pd.read_csv(cfg.paths.master_table)
    if args.n is not None:
        master = master.head(args.n)

    prompts = ["P1", "P4"] if args.prompt == "both" else [args.prompt]
    out_path = Path(cfg.paths.results) / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pipe = MedGemmaInference(cfg)

    n_rows = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "prompt_id", "pooling", "token_idx", "weight"])
        for i, row in master.iterrows():
            t0 = time.time()
            image = load_image(cfg, row["image_filename"], row["split"])
            for prompt_id in prompts:
                maps = pooling_maps_for_image(
                    pipe, image, prompt_id, row["image_filename"], row["split"]
                )
                for pooling in POOLINGS:
                    if pooling not in maps:
                        continue  # roi sin máscara
                    w = maps[pooling]
                    for token_idx in range(len(w)):
                        writer.writerow(
                            [row["image_filename"], prompt_id, pooling, token_idx, f"{w[token_idx]:.6e}"]
                        )
                        n_rows += 1
            print(f"[{i + 1}/{len(master)}] {row['image_filename']} "
                  f"({time.time() - t0:.1f} s)")

    print(f"\nListo: {out_path} ({n_rows} filas)")


if __name__ == "__main__":
    main()
