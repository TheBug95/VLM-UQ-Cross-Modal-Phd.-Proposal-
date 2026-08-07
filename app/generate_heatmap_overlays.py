"""Genera los heatmaps SUPERPUSTOS (overlay) de pooling sobre el fundus.

A partir de ``results/pooling_maps.csv`` (los 256 pesos por imagen × pooling,
extraídos en GPU con ``python -m src.extract_pooling_maps``) y las miniaturas
cuadradas de ``app/assets/thumbnails_square/``, genera una imagen JPEG por
imagen × técnica de pooling con el heatmap 16×16 mezclado sobre el fundus —
misma visualización que ``src.uncertainty.generate_attention_heatmap`` pero
sin GPU ni torch, replicando su receta (upsample por celdas, normalización
mín–máx, colormap jet, alpha blend).

Salida: ``app/assets/heatmaps/{stem}_{pooling}.jpg``
(129 imágenes × 8 poolings ≈ 1.032 archivos; roi solo en patológicas).

Uso:
    python app/generate_heatmap_overlays.py            # solo los que falten
    python app/generate_heatmap_overlays.py --force    # regenerar todo
    python app/generate_heatmap_overlays.py --alpha 0.5 --colormap viridis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ASSETS = Path(__file__).resolve().parent / "assets"
SQUARE = ASSETS / "thumbnails_square"
OUT = ASSETS / "heatmaps"
GRID = 16  # tokens por lado (256 tokens visuales)

POOLINGS = ["mean", "max", "topk", "normw", "attn", "rollout", "headspec", "roi"]


def overlay(square_img: Image.Image, weights: np.ndarray,
            alpha: float = 0.55, colormap: str = "jet") -> Image.Image:
    """Mezcla el heatmap 16×16 sobre la imagen cuadrada (receta de src/uncertainty)."""
    import matplotlib

    size = square_img.size[0]
    cell = size // GRID
    if size % GRID != 0:
        square_img = square_img.resize((GRID * cell, GRID * cell), Image.BILINEAR)
        size = square_img.size[0]
        cell = size // GRID

    img = np.asarray(square_img, dtype=np.float32) / 255.0
    heat = weights.reshape(GRID, GRID)
    up = np.kron(heat, np.ones((cell, cell)))
    if up.max() > up.min():
        up = (up - up.min()) / (up.max() - up.min())

    rgb = matplotlib.colormaps[colormap](up)[:, :, :3]
    out = np.clip((1 - alpha) * img + alpha * rgb, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera overlays de pooling sobre el fundus")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--colormap", default="jet")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prompt", default="P1")
    args = parser.parse_args()

    maps_path = ASSETS / "pooling_maps.csv"
    if not maps_path.exists():
        raise SystemExit(
            f"No existe {maps_path}. Copia results/pooling_maps.csv y corre "
            "python app/prepare_assets.py primero."
        )
    df = pd.read_csv(maps_path)
    df = df[df["prompt_id"] == args.prompt]
    OUT.mkdir(parents=True, exist_ok=True)

    n_ok, n_skip, n_fail = 0, 0, 0
    for (img_name, pooling), g in df.groupby(["image_filename", "pooling"]):
        dst = OUT / f"{Path(img_name).stem}_{pooling}.jpg"
        if dst.exists() and not args.force:
            n_skip += 1
            continue
        thumb = SQUARE / img_name
        if not thumb.exists():
            print(f"  [sin thumbnail] {img_name}")
            n_fail += 1
            continue
        g = g.sort_values("token_idx")
        if len(g) != GRID * GRID:
            print(f"  [mapa incompleto] {img_name} {pooling}: {len(g)} tokens")
            n_fail += 1
            continue
        weights = g["weight"].to_numpy()
        img = Image.open(thumb).convert("RGB")
        overlay(img, weights, alpha=args.alpha, colormap=args.colormap).save(
            dst, "JPEG", quality=85
        )
        n_ok += 1

    print(f"Overlays: {n_ok} generados, {n_skip} ya existían, {n_fail} fallidos -> {OUT}")


if __name__ == "__main__":
    main()
