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
    # Convertir a float32 para evitar pérdida de precisión con bfloat16
    w = torch.tensor(weights, dtype=torch.float32, device=h_img.device)
    return (h_img.float() * w.unsqueeze(-1)).sum(dim=0)


# ------------------------------------------------------------------------------
# Atención cruzada (deployable alternative a máscaras)
# ------------------------------------------------------------------------------
def compute_attention_weights(
    attentions: tuple,
    layer: int,
    img_positions: torch.Tensor,
    seq_len: int,
) -> np.ndarray:
    """Extrae pesos de atención del último token hacia los tokens de imagen.

    En un transformer decoder-only, el último token (que genera la respuesta)
    atiende a todos los tokens anteriores. Los pesos sobre los tokens visuales
    nos dicen "dónde mira el modelo" al decidir.

    Args:
        attentions: tupla de atenciones por paso de generación (de
            model.generate(output_attentions=True)). Con max_new_tokens=1,
            solo existe attentions[0] (prefill).
        layer: índice de capa (17, 26, 34).
        img_positions: tensor con las posiciones de los tokens de imagen.
        seq_len: longitud total de la secuencia.

    Returns:
        Array (num_image_tokens,) con pesos normalizados que suman 1.

    Raises:
        ValueError: si attentions está vacío o no tiene la estructura esperada.
    """
    if not attentions or len(attentions) == 0:
        raise ValueError(
            "attentions está vacío. ¿Se cargó el modelo con "
            "attn_implementation='eager'? sdpa no soporta output_attentions."
        )

    # attentions[0] es la tupla por capa del prefill (sin embeddings)
    # hidden_states tiene 35 entradas (0=embeddings, 1-34=capas)
    # attentions tiene 34 entradas (0-33=capas) → layer-1
    attn_idx = layer - 1
    if attn_idx < 0 or attn_idx >= len(attentions[0]):
        raise ValueError(
            f"Capa {layer} fuera de rango: attentions[0] tiene {len(attentions[0])} entradas "
            f"(índice válido: 1–{len(attentions[0])})"
        )
    attn_layer = attentions[0][attn_idx]  # (1, num_heads, seq_len, seq_len)

    # Atención del último token (índice -1) hacia todos los tokens
    # Promediar sobre cabezas de atención y convertir a float32 (NumPy no soporta bfloat16)
    attn_last = attn_layer[0, :, -1, :].mean(dim=0).float()  # (seq_len,)

    # Extraer solo los pesos sobre tokens de imagen
    img_attn = attn_last[img_positions].cpu().numpy()

    # Normalizar
    total = img_attn.sum()
    if total > 0:
        img_attn = img_attn / total
    else:
        img_attn = np.ones_like(img_attn) / len(img_attn)

    return img_attn


def attention_weighted_pooling(h_img: torch.Tensor, weights: np.ndarray) -> torch.Tensor:
    """Aplica pooling ponderado por atención sobre los tokens de imagen.

    Args:
        h_img: tensor (num_tokens, hidden_dim).
        weights: array (num_tokens,) con pesos que suman 1.

    Returns:
        Vector (hidden_dim,).
    """
    # Convertir a float32 para evitar pérdida de precisión con bfloat16
    w = torch.tensor(weights, dtype=torch.float32, device=h_img.device)
    return (h_img.float() * w.unsqueeze(-1)).sum(dim=0)


# ------------------------------------------------------------------------------
# Top-K Spatial Pooling (sin parámetros)
# ------------------------------------------------------------------------------
def topk_pooling(h_img: torch.Tensor, k: int = 26) -> torch.Tensor:
    """Selecciona los K tokens con mayor norma L2 y promedia solo esos.

    Descarta el resto (dilución espacial reducida). Si k >= num_tokens, usa
    mean pooling.

    Args:
        h_img: tensor (num_tokens, hidden_dim).
        k: número de tokens a seleccionar (default 26 ≈ 10% de 256).

    Returns:
        Vector (hidden_dim,) con el promedio de los K tokens más activos.
    """
    if k >= h_img.shape[0]:
        return h_img.float().mean(dim=0)

    # Norma L2 por token
    norms = h_img.float().norm(dim=1)  # (num_tokens,)
    # Índices de los K mayores
    topk_idx = norms.topk(k).indices
    # Promedio solo esos K
    return h_img[topk_idx].float().mean(dim=0)


# ------------------------------------------------------------------------------
# Norm-Weighted Pooling (sin parámetros)
# ------------------------------------------------------------------------------
def norm_weighted_pooling(h_img: torch.Tensor) -> torch.Tensor:
    """Pondera cada token por su norma L2 (self-weighting).

    Las regiones de mayor activación dominan el vector resultante.

    Args:
        h_img: tensor (num_tokens, hidden_dim).

    Returns:
        Vector (hidden_dim,) con el pooling ponderado por norma.
    """
    h_float = h_img.float()
    norms = h_float.norm(dim=1)  # (num_tokens,)
    weights = norms / norms.sum()
    return (h_float * weights.unsqueeze(-1)).sum(dim=0)


# ------------------------------------------------------------------------------
# Attention Rollout (Abnar & Zuidema 2020)
# ------------------------------------------------------------------------------
def attention_rollout(
    attentions: tuple,
    img_positions: torch.Tensor,
    start_layer: int = 0,
    end_layer: int = 34,
) -> np.ndarray:
    """Calcula la atención efectiva del último token a los tokens de imagen
    propagando la atención a través de todas las capas.

    Captura caminos indirectos: respuesta -> token de texto -> tokens de imagen.
    Ref: Abnar & Zuidema (2020), "Quantifying Attention Flow".

    Args:
        attentions: tupla de atenciones por paso de generación (de
            model.generate(output_attentions=True)).
        img_positions: tensor con las posiciones de los tokens de imagen.
        start_layer: capa inicial (default 0).
        end_layer: capa final (default 34).

    Returns:
        Array (num_image_tokens,) con pesos normalizados que suman 1.
    """
    if not attentions or len(attentions) == 0:
        raise ValueError("attentions está vacío")

    # Empezar con la identidad
    rollout = None

    for layer_idx in range(start_layer, min(end_layer, len(attentions[0]))):
        # Promediar sobre cabezas y convertir a float32
        attn = attentions[0][layer_idx][0].float().mean(dim=0)  # (seq, seq)

        # Añadir conexión residual: 0.5 * identidad + 0.5 * atención
        eye = torch.eye(attn.shape[0], device=attn.device)
        attn_with_residual = 0.5 * eye + 0.5 * attn

        # Re-normalizar filas para que sumen 1
        attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)

        if rollout is None:
            rollout = attn_with_residual
        else:
            rollout = attn_with_residual @ rollout

    # Extraer: atención del último token -> tokens de imagen
    last_to_img = rollout[-1, img_positions].cpu().numpy()

    # Normalizar
    total = last_to_img.sum()
    if total > 0:
        last_to_img = last_to_img / total
    else:
        last_to_img = np.ones_like(last_to_img) / len(last_to_img)

    return last_to_img


# ------------------------------------------------------------------------------
# Head-Specific Attention (selección de cabezas visuales)
# ------------------------------------------------------------------------------
def head_specific_attention(
    attentions: tuple,
    layer: int,
    img_positions: torch.Tensor,
    top_k_heads: int = 4,
) -> np.ndarray:
    """Selecciona las K cabezas que más atienden a tokens de imagen
    y usa solo esas para el pooling.

    Intuición: las cabezas "visuales" concentran su atención en los tokens
    de imagen cuando deben responder sobre la imagen. Las cabezas "posicionales"
    atienden uniformemente y añaden ruido.

    Args:
        attentions: tupla de atenciones por paso de generación.
        layer: índice de capa (17, 26, 34).
        img_positions: tensor con las posiciones de los tokens de imagen.
        top_k_heads: número de cabezas a seleccionar (default 4).

    Returns:
        Array (num_image_tokens,) con pesos normalizados que suman 1.
    """
    if not attentions or len(attentions) == 0:
        raise ValueError("attentions está vacío")

    attn_idx = layer - 1
    if attn_idx < 0 or attn_idx >= len(attentions[0]):
        raise ValueError(f"Capa {layer} fuera de rango")

    attn_layer = attentions[0][attn_idx][0].float()  # (num_heads, seq, seq)
    num_heads = attn_layer.shape[0]

    # Para cada cabeza: ¿cuánta atención total pone el último token
    # sobre los tokens de imagen?
    last_token_attn = attn_layer[:, -1, :]  # (num_heads, seq)
    img_focus = last_token_attn[:, img_positions].sum(dim=-1)  # (num_heads,)

    # Seleccionar las K cabezas más "visuales"
    _, top_heads = img_focus.topk(min(top_k_heads, num_heads))

    # Promediar solo sobre esas cabezas
    selected_attn = last_token_attn[top_heads, :][:, img_positions]  # (K, 256)
    weights = selected_attn.mean(dim=0).cpu().numpy()

    # Normalizar
    total = weights.sum()
    if total > 0:
        weights = weights / total
    else:
        weights = np.ones_like(weights) / len(weights)

    return weights


# ------------------------------------------------------------------------------
# Heatmap de atención
# ------------------------------------------------------------------------------
def generate_attention_heatmap(
    image: Image.Image,
    weights: np.ndarray,
    grid_size: int = 16,
    alpha: float = 0.6,
    colormap: str = "jet",
) -> Image.Image:
    """Genera un heatmap de atención superpuesto sobre la imagen original.

    Args:
        image: imagen PIL original (cualquier tamaño).
        weights: array (grid_size * grid_size,) con pesos por token.
        grid_size: lado del grid (16 para 256 tokens).
        alpha: transparencia del heatmap (0 = invisible, 1 = opaco).
        colormap: mapa de colores de matplotlib.

    Returns:
        Imagen PIL con el heatmap superpuesto.
    """
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    # Convertir imagen a array y redimensionar a 896×896 (tamaño de entrada del modelo)
    target_size = 896
    img_resized = image.resize((target_size, target_size), resample=Image.Resampling.BILINEAR)
    img_arr = np.asarray(img_resized, dtype=np.float32) / 255.0

    # Crear mapa de calor desde los pesos
    heat = weights.reshape(grid_size, grid_size)
    cell_size = target_size // grid_size

    # Upsampling del heatmap a 896×896 (cada celda se expande)
    heat_up = np.kron(heat, np.ones((cell_size, cell_size)))

    # Normalizar para visualización
    if heat_up.max() > heat_up.min():
        heat_up = (heat_up - heat_up.min()) / (heat_up.max() - heat_up.min())

    # Aplicar colormap
    cmap = cm.get_cmap(colormap)
    heat_rgb = cmap(heat_up)[:, :, :3]  # (896, 896, 3)

    # Superponer
    overlay = (1 - alpha) * img_arr + alpha * heat_rgb
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    return Image.fromarray(overlay)


def save_attention_heatmap(
    image: Image.Image,
    weights: np.ndarray,
    out_path: str | Path,
    **kwargs,
) -> None:
    """Guarda el heatmap de atención en disco."""
    heatmap = generate_attention_heatmap(image, weights, **kwargs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap.save(out_path)


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
