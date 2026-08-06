"""Prepara los assets estáticos del dashboard (app/assets/).

Copia los CSVs de resultados y genera miniaturas de las 129 imágenes del
dataset MM-ODIR-129. Si el dataset no está descargado en ``data/mm_odir_129``,
lo descarga con ``snapshot_download`` (solo los .jpg y .json, sin máscaras).

Uso:
    python app/prepare_assets.py            # todo
    python app/prepare_assets.py --solo-csv # solo copiar CSVs

⚠️ Doble ciego: no copiar nada que identifique al autor (AGENTS.md §3).
   patient_id se omite de los assets de imágenes (solo se usa image_filename).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(__file__).resolve().parent / "assets"
THUMBS = ASSETS / "thumbnails"
SQUARE = ASSETS / "thumbnails_square"
MAX_THUMB_PX = 384

CSV_FILES = [
    ("results/results_full.csv", "results_full.csv"),
    ("results/evaluation_summary.csv", "evaluation_summary.csv"),
    ("results/results_verbalized.csv", "results_verbalized.csv"),
    ("results/results_self_consistency.csv", "results_self_consistency.csv"),
    ("results/acc_cov_P1_winner.csv", "acc_cov_P1_winner.csv"),
    ("results/acc_cov_P1_rankcombo.csv", "acc_cov_P1_rankcombo.csv"),
    ("results/acc_cov_P4_winner.csv", "acc_cov_P4_winner.csv"),
    ("results/acc_cov_P4_rankcombo.csv", "acc_cov_P4_rankcombo.csv"),
    ("data/master_table.csv", "master_table.csv"),
    ("figures/tabla_t1_resultados.csv", "tabla_t1_resultados.csv"),
    ("figures/tabla_t2_ablaciones.csv", "tabla_t2_ablaciones.csv"),
    ("figures/tabla_t3_comparativa.csv", "tabla_t3_comparativa.csv"),
    ("figures/tabla_t4_costo_beneficio.csv", "tabla_t4_costo_beneficio.csv"),
    ("figures/tabla_t5_calibracion.csv", "tabla_t5_calibracion.csv"),
    # Mapas de pooling (generado por src/extract_pooling_maps.py en Colab/GPU;
    # si no existe, el tab «Mapas de pooling» muestra las instrucciones)
    ("results/pooling_maps.csv", "pooling_maps.csv"),
]

DATASET_REPO = "TheBug95/MM-ODIR-129"  # ⚠️ no propagar esta URL a la submission (doble ciego)
SPLITS = ["train", "validation", "test"]


def copiar_csvs() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for src_rel, dst_name in CSV_FILES:
        src = ROOT / src_rel
        if not src.exists():
            print(f"  [skip] {src_rel} no existe")
            continue
        shutil.copy2(src, ASSETS / dst_name)
        print(f"  [ok] {src_rel} -> app/assets/{dst_name}")


def asegurar_dataset() -> Path:
    """Devuelve la ruta local del dataset; lo descarga si hace falta (solo jpg/json)."""
    data_dir = ROOT / "data" / "mm_odir_129"
    if data_dir.exists() and any(data_dir.rglob("*.jpg")):
        return data_dir
    print("  Dataset no encontrado localmente; descargando (solo .jpg/.json)...")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=data_dir,
        allow_patterns=["*.jpg", "*.json"],
    )
    return data_dir


def generar_thumbnails() -> None:
    master = pd.read_csv(ASSETS / "master_table.csv")
    data_dir = asegurar_dataset()
    THUMBS.mkdir(parents=True, exist_ok=True)
    SQUARE.mkdir(parents=True, exist_ok=True)
    n_ok, n_fail = 0, 0
    for _, row in master.iterrows():
        dst = THUMBS / row["image_filename"]
        dst_sq = SQUARE / row["image_filename"]
        if dst.exists() and dst_sq.exists():
            n_ok += 1
            continue
        src = data_dir / row["split"] / row["image_filename"]
        if not src.exists():
            # fallback: buscar en cualquier split
            candidatos = list(data_dir.rglob(row["image_filename"]))
            src = candidatos[0] if candidatos else src
        if not src.exists():
            print(f"  [falta] {row['image_filename']}")
            n_fail += 1
            continue
        img = Image.open(src).convert("RGB")
        # Versión cuadrada: el AutoProcessor estira la imagen a 896×896, así que
        # los mapas de pooling (grid 16×16) se alinean con esta versión.
        if not dst_sq.exists():
            img.resize((MAX_THUMB_PX, MAX_THUMB_PX), Image.LANCZOS).save(
                dst_sq, "JPEG", quality=85
            )
        if not dst.exists():
            thumb = img.copy()
            thumb.thumbnail((MAX_THUMB_PX, MAX_THUMB_PX), Image.LANCZOS)
            thumb.save(dst, "JPEG", quality=85)
        n_ok += 1
    print(f"  Thumbnails: {n_ok} ok, {n_fail} faltantes -> {THUMBS}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara app/assets/ para el dashboard")
    parser.add_argument("--solo-csv", action="store_true", help="No descargar imágenes")
    args = parser.parse_args()

    print("== Copiando CSVs ==")
    copiar_csvs()
    if not args.solo_csv:
        print("== Generando thumbnails ==")
        generar_thumbnails()
    print("Listo.")


if __name__ == "__main__":
    main()
