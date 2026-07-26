"""src/uncertainty.py — Funciones de incertidumbre y pooling ponderado por ROI.

Implementa la ablación ROI-weighted pooling propuesta para mitigar la dilución
espacial de la señal del disco óptico (5–10% de la imagen) en el mean pooling
uniforme de los 256 tokens visuales.

Convenciones:
    - Imagen procesada: 896×896 (config.inference.resolution).
    - Grid de tokens visuales: 16×16 = 256 tokens (config.inference.num_image_tokens).
    - Cada token corresponde a una región de 56×56 píxeles (896 / 16 = 56).
    - La máscara de disco se redimensiona a 896×896 y se promedia por celda.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import Config


# ------------------------------------------------------------------------------
# Carga y preproceso de máscaras
# ------------------------------------------------------------------------------
def load_disc_mask(cfg: Config, image_filename: str, split: str) -> np.ndarray | None:
    """Carga la máscara de disco (o copa como fallback) para una imagen.

    Devuelve un array binario (H, W) con 1 = dentro del disco, 0 = fuera.
    Si no existe máscara, devuelve None.
    """
    stem = Path(image_filename).stem
    base_dir = Path(cfg.paths.data) / "mm_odir_129" / split

    # Preferir máscara de disco; fallback a copa
    for suffix in ("_disc.png", "_cup.png"):
        path = base_dir / f"{stem}{suffix}"
        if path.exists():
            mask = Image.open(path).convert("L")
            mask_arr = np.asarray(mask, dtype=np.uint8)
            return (mask_arr > 127).astype(np.uint8)
    return None


def resize_mask_to_input(mask: np.ndarray, target_size: int = 896) -> np.ndarray:
    """Redimensiona la máscara al tamaño de entrada del modelo (896×896).

    Usa NEAREST para preservar la binaridad. El AutoProcessor aplica el mismo
    resize a la imagen RGB; replicamos ese preproceso sobre la máscara.
    """
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img = img.resize((target_size, target_size), resample=Image.Resampling.NEAREST)
    return (np.asarray(img, dtype=np.uint8) > 127).astype(np.uint8)


# ------------------------------------------------------------------------------
# Pesos ROI por token visual
# ------------------------------------------------------------------------------
def compute_roi_weights(
    cfg: Config,
    image_filename: str,
    split: str,
    grid_size: int = 16,
) -> np.ndarray | None:
    """Calcula los pesos alpha_m para cada token visual según la máscara de disco.

    Args:
        cfg: configuración del experimento.
        image_filename: nombre del archivo de imagen.
        split: train / validation / test.
        grid_size: lado del grid de tokens (16 para 256 tokens).

    Returns:
        Array (grid_size * grid_size,) con pesos normalizados que suman 1,
        o None si no hay máscara disponible.
    """
    mask = load_disc_mask(cfg, image_filename, split)
    if mask is None:
        return None

    target_size = cfg.inference.resolution
    mask_resized = resize_mask_to_input(mask, target_size=target_size)

    cell_size = target_size // grid_size
    weights = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            cell = mask_resized[
                i * cell_size : (i + 1) * cell_size,
                j * cell_size : (j + 1) * cell_size,
            ]
            weights[i, j] = cell.mean()

    total = weights.sum()
    if total <= 0:
        warnings.warn(f"Máscara vacía para {image_filename}; usando uniforme.")
        weights = np.ones((grid_size, grid_size), dtype=np.float32) / (grid_size * grid_size)
    else:
        weights /= total

    return weights.flatten()


# ------------------------------------------------------------------------------
# Pooling ponderado
# ------------------------------------------------------------------------------
def roi_weighted_pooling(h_img: torch.Tensor, weights: np.ndarray) -> torch.Tensor:
    """Aplica pooling ponderado por ROI sobre los tokens de imagen.

    Args:
        h_img: tensor (num_tokens, hidden_dim) con los hidden states de los
            256 tokens visuales.
        weights: array (num_tokens,) con pesos que suman 1.

    Returns:
        Vector (hidden_dim,) con el pooling ponderado.
    """
    w = torch.tensor(weights, dtype=h_img.dtype, device=h_img.device)
    return (h_img * w.unsqueeze(-1)).sum(dim=0)


# ------------------------------------------------------------------------------
# Utilidades para análisis
# ------------------------------------------------------------------------------
def get_roi_coverage(cfg: Config, image_filename: str, split: str) -> float | None:
    """Devuelve la fracción de la imagen cubierta por la máscara de disco.

    Sirve para reportar en el paper qué tan pequeña es la ROI en promedio.
    """
    mask = load_disc_mask(cfg, image_filename, split)
    if mask is None:
        return None
    return float(mask.mean())
