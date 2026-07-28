"""src/inference.py — Pipeline de inferencia sobre MedGemma-4B.

Extrae logits yes/no, baselines de incertidumbre (entropy, MSP, energy) y las
54 variantes de KL/JSD (capas {17,26,34} × τ {1,2,4} × pooling {mean,max}).
Guarda resultados incrementalmente en results/results_full.csv.

Convenciones congeladas (ver AGENTS.md §6.2 y val_04_generate_api.py):
    - p_text = hidden_states[0][capa][:, -1, :] (última posición del prefill).
    - Máscara de imagen por input_ids == config.image_token_index, nunca slicing.
    - scores[0] = logits del primer token generado (greedy).
    - yes=4443, no=1904 (sin espacio inicial).

Uso:
    python -m src.inference --pilot --n 20        # piloto con sanity checks
    python -m src.inference --run-full            # corrida completa
    python -m src.inference --self-consistency    # baseline multi-pass
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.distance import jensenshannon

from src.config import Config
from src.data import download_dataset
from src.uncertainty import (
    attention_rollout,
    attention_weighted_pooling,
    compute_attention_weights,
    compute_roi_weights,
    cosine_distance,
    generate_attention_heatmap,
    head_specific_attention,
    norm_weighted_pooling,
    roi_weighted_pooling,
    save_attention_heatmap,
    topk_pooling,
)

# Columnas de salida (orden congelado por el diseño)
BASE_COLUMNS = [
    "image_filename", "patient_id", "prompt_id", "split",
    "logit_yes", "logit_no", "p_yes", "pred", "label", "correct",
    "entropy_answer", "msp_answer", "energy_answer",
]


# ------------------------------------------------------------------------------
# Utilidades de incertidumbre
# ------------------------------------------------------------------------------
def to_distribution(vec: torch.Tensor, tau: float = 4.0) -> torch.Tensor:
    """Convierte un vector crudo a distribución con F.log_softmax en float64.

    Usa softmax crudo con τ alta (default 4.0) para aplanar la distribución
    sin normalizar. Esto preserva las diferencias relativas entre hidden states
    de imagen y texto, permitiendo que la KL varíe entre casos.

    Nota: no normalizamos (ni z-score, ni norma L2) porque ambos aplanan
    demasiado, haciendo que la KL sea casi cero para todos los casos.
    """
    # Convertir a float32 para evitar overflow/underflow con bfloat16
    vec = vec.float()

    log_p = torch.nn.functional.log_softmax(vec / tau, dim=0, dtype=torch.float64)
    return log_p.exp()


def kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    """KL(p ‖ q) con estabilidad numérica.

    Args:
        p, q: distribuciones de probabilidad (1-D) que suman 1.
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return float((p * (p.log() - q.log())).sum().item())


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (scipy devuelve distancia, se eleva al cuadrado).

    Base natural (e): la cota superior es ln 2 ≈ 0.693, como exige el diseño.
    """
    return float(jensenshannon(p, q) ** 2)


# ------------------------------------------------------------------------------
# Pipeline de inferencia
# ------------------------------------------------------------------------------
class MedGemmaInference:
    """Carga MedGemma-4B y ejecuta la pasada single-pass de extracción."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Carga processor y modelo según config.yaml."""
        from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration, Gemma3Processor

        model_name = self.cfg.model.name
        kwargs: dict[str, Any] = {}

        if self.device == "cuda":
            kwargs["device_map"] = self.cfg.model.device_map
        if self.cfg.model.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kwargs["torch_dtype"] = getattr(torch, self.cfg.model.torch_dtype)

        # Para output_attentions=True se requiere eager attention (sdpa no lo soporta)
        attn_impl = getattr(self.cfg.model, "attn_implementation", None)
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        self._processor = Gemma3Processor.from_pretrained(model_name)
        self._model = Gemma3ForConditionalGeneration.from_pretrained(model_name, **kwargs)
        self._model.eval()

        # Resolver IDs desde el tokenizer real si difieren de config.yaml
        self.cfg.resolve_model_ids(quiet=True)

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self.load()
        return self._processor

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------
    def build_prompt(self, prompt_id: str) -> str:
        """Construye el prompt de chat para P1 o P4."""
        p = self.cfg.prompts[prompt_id]
        msgs = []
        if p.system:
            msgs.append({"role": "system", "content": [{"type": "text", "text": p.system}]})
        msgs.append({"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": p.user},
        ]})
        return self.processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    # ------------------------------------------------------------------
    # Extracción
    # ------------------------------------------------------------------
    def _extract_signals(
        self,
        out,
        inputs: dict[str, torch.Tensor],
        roi_weights: np.ndarray | None = None,
        attentions: tuple | None = None,
    ) -> dict[str, Any]:
        """Extrae logits, baselines y las 54 variantes de KL/JSD."""
        results: dict[str, Any] = {}

        # --- Logits yes/no -------------------------------------------------
        scores0 = out.scores[0]  # (1, vocab_size)
        yes_id = self.cfg.tokens.yes
        no_id = self.cfg.tokens.no
        logit_yes = float(scores0[0, yes_id].item())
        logit_no = float(scores0[0, no_id].item())

        logits_yesno = torch.tensor([logit_yes, logit_no], device=self.device)
        probs = torch.softmax(logits_yesno, dim=0)
        p_yes = float(probs[0].item())
        pred = 1 if p_yes > 0.5 else 0

        results.update({
            "logit_yes": logit_yes,
            "logit_no": logit_no,
            "p_yes": p_yes,
            "pred": pred,
        })

        # --- Baselines de salida -------------------------------------------
        eps = self.cfg.uncertainty.epsilon
        entropy = float(-(probs * torch.log(probs + eps)).sum().item())
        msp = float(probs.max().item())
        energy = float(-torch.logsumexp(logits_yesno, dim=0).item())
        results.update({
            "entropy_answer": entropy,
            "msp_answer": msp,
            "energy_answer": energy,
        })

        # --- Hidden states --------------------------------------------------
        hs = out.hidden_states[0]  # prefill
        input_ids = inputs["input_ids"][0]
        img_token_id = self.cfg.tokens.image_token_index
        img_mask = input_ids == img_token_id
        img_positions = img_mask.nonzero().flatten()

        if len(img_positions) != self.cfg.inference.num_image_tokens:
            warnings.warn(
                f"Se esperaban {self.cfg.inference.num_image_tokens} tokens de imagen, "
                f"se encontraron {len(img_positions)}"
            )

        # Tokens de texto del prompt (sin imagen, sin especiales finales)
        # Para la ablación imagen↔prompt usamos las posiciones de texto.
        text_mask = ~img_mask
        text_positions = text_mask.nonzero().flatten()
        # Excluir el último token (que es p_text) para no duplicar
        if len(text_positions) > 0:
            text_positions = text_positions[:-1]

        # Solo capa 34 para KL (capas 17 y 26 colapsan por baja diferenciación)
        layer = 34
        h_layer = hs[layer][0]  # (seq_len, 2560)

        # p_text: última posición del prefill
        p_text_vec = h_layer[-1, :]

        # p_vis: pooling de tokens de imagen
        h_img = h_layer[img_positions, :]

        for pooling in self.cfg.uncertainty.pooling:
            if pooling == "mean":
                p_vis_vec = h_img.mean(dim=0)
            else:  # max
                p_vis_vec = h_img.max(dim=0).values

            # Ablación imagen↔prompt: mean pooling de tokens de texto
            if len(text_positions) > 0 and pooling == "mean":
                p_prompt_vec = h_layer[text_positions, :].mean(dim=0)
            else:
                p_prompt_vec = None

            for tau in self.cfg.uncertainty.temperatures:
                p_vis = to_distribution(p_vis_vec, tau)
                p_text = to_distribution(p_text_vec, tau)

                suffix = f"L{layer}_tau{tau}_{pooling}"
                results[f"kl_v_t_{suffix}"] = kl_div(p_vis, p_text, eps)
                results[f"kl_t_v_{suffix}"] = kl_div(p_text, p_vis, eps)
                results[f"jsd_{suffix}"] = jsd(
                    p_vis.cpu().numpy(), p_text.cpu().numpy()
                )

                # Distancia coseno (no requiere softmax, no colapsa)
                results[f"cosine_{suffix}"] = cosine_distance(p_vis_vec, p_text_vec)

                # Ablación prompt (solo L34, tau=1, mean según diseño)
                if (
                    tau == 1.0
                    and pooling == "mean"
                    and p_prompt_vec is not None
                ):
                    p_prompt = to_distribution(p_prompt_vec, tau)
                    results["kl_prompt_L34_tau1_mean"] = kl_div(
                        p_vis, p_prompt, eps
                    )

            # Ablación ROI-weighted pooling (responde a la crítica de dilución espacial)
            if roi_weights is not None:
                p_vis_roi_vec = roi_weighted_pooling(h_img, roi_weights)
                for tau in self.cfg.uncertainty.temperatures:
                    p_vis_roi = to_distribution(p_vis_roi_vec, tau)
                    p_text = to_distribution(p_text_vec, tau)
                    suffix = f"L{layer}_tau{tau}_roi"
                    results[f"kl_v_t_{suffix}"] = kl_div(p_vis_roi, p_text, eps)
                    results[f"kl_t_v_{suffix}"] = kl_div(p_text, p_vis_roi, eps)
                    results[f"jsd_{suffix}"] = jsd(
                        p_vis_roi.cpu().numpy(), p_text.cpu().numpy()
                    )
                    results[f"cosine_{suffix}"] = cosine_distance(p_vis_roi_vec, p_text_vec)

            # Ablación attention-weighted pooling (deployable, sin máscaras externas)
            if attentions is not None:
                attn_weights = compute_attention_weights(
                    attentions, layer, img_positions, seq_len=hs[layer].shape[1]
                )
                p_vis_attn_vec = attention_weighted_pooling(h_img, attn_weights)
                for tau in self.cfg.uncertainty.temperatures:
                    p_vis_attn = to_distribution(p_vis_attn_vec, tau)
                    p_text = to_distribution(p_text_vec, tau)
                    suffix = f"L{layer}_tau{tau}_attn"
                    results[f"kl_v_t_{suffix}"] = kl_div(p_vis_attn, p_text, eps)
                    results[f"kl_t_v_{suffix}"] = kl_div(p_text, p_vis_attn, eps)
                    results[f"jsd_{suffix}"] = jsd(
                        p_vis_attn.cpu().numpy(), p_text.cpu().numpy()
                    )
                    results[f"cosine_{suffix}"] = cosine_distance(p_vis_attn_vec, p_text_vec)

            # Ablación Top-K pooling (sin parámetros, deployable)
            p_vis_topk_vec = topk_pooling(h_img, k=max(1, len(img_positions) // 10))
            for tau in self.cfg.uncertainty.temperatures:
                p_vis_topk = to_distribution(p_vis_topk_vec, tau)
                p_text = to_distribution(p_text_vec, tau)
                suffix = f"L{layer}_tau{tau}_topk"
                results[f"kl_v_t_{suffix}"] = kl_div(p_vis_topk, p_text, eps)
                results[f"kl_t_v_{suffix}"] = kl_div(p_text, p_vis_topk, eps)
                results[f"jsd_{suffix}"] = jsd(
                    p_vis_topk.cpu().numpy(), p_text.cpu().numpy()
                )
                results[f"cosine_{suffix}"] = cosine_distance(p_vis_topk_vec, p_text_vec)

            # Ablación Norm-Weighted pooling (sin parámetros, deployable)
            p_vis_normw_vec = norm_weighted_pooling(h_img)
            for tau in self.cfg.uncertainty.temperatures:
                p_vis_normw = to_distribution(p_vis_normw_vec, tau)
                p_text = to_distribution(p_text_vec, tau)
                suffix = f"L{layer}_tau{tau}_normw"
                results[f"kl_v_t_{suffix}"] = kl_div(p_vis_normw, p_text, eps)
                results[f"kl_t_v_{suffix}"] = kl_div(p_text, p_vis_normw, eps)
                results[f"jsd_{suffix}"] = jsd(
                    p_vis_normw.cpu().numpy(), p_text.cpu().numpy()
                )
                results[f"cosine_{suffix}"] = cosine_distance(p_vis_normw_vec, p_text_vec)

            # Ablación Attention Rollout (caminos indirectos de información)
            # NOTA: el rollout termina en la misma capa que los hidden states
            # para evitar data leak temporal (usar información futura).
            if attentions is not None:
                rollout_weights = attention_rollout(
                    attentions, img_positions, end_layer=layer
                )
                p_vis_rollout_vec = attention_weighted_pooling(h_img, rollout_weights)
                for tau in self.cfg.uncertainty.temperatures:
                    p_vis_rollout = to_distribution(p_vis_rollout_vec, tau)
                    p_text = to_distribution(p_text_vec, tau)
                    suffix = f"L{layer}_tau{tau}_rollout"
                    results[f"kl_v_t_{suffix}"] = kl_div(p_vis_rollout, p_text, eps)
                    results[f"kl_t_v_{suffix}"] = kl_div(p_text, p_vis_rollout, eps)
                    results[f"jsd_{suffix}"] = jsd(
                        p_vis_rollout.cpu().numpy(), p_text.cpu().numpy()
                    )
                    results[f"cosine_{suffix}"] = cosine_distance(p_vis_rollout_vec, p_text_vec)

            # Ablación Head-Specific Attention (cabezas visuales seleccionadas)
            if attentions is not None:
                headspec_weights = head_specific_attention(attentions, layer, img_positions)
                p_vis_headspec_vec = attention_weighted_pooling(h_img, headspec_weights)
                for tau in self.cfg.uncertainty.temperatures:
                    p_vis_headspec = to_distribution(p_vis_headspec_vec, tau)
                    p_text = to_distribution(p_text_vec, tau)
                    suffix = f"L{layer}_tau{tau}_headspec"
                    results[f"kl_v_t_{suffix}"] = kl_div(p_vis_headspec, p_text, eps)
                    results[f"kl_t_v_{suffix}"] = kl_div(p_text, p_vis_headspec, eps)
                    results[f"jsd_{suffix}"] = jsd(
                        p_vis_headspec.cpu().numpy(), p_text.cpu().numpy()
                    )
                    results[f"cosine_{suffix}"] = cosine_distance(p_vis_headspec_vec, p_text_vec)

        return results

    # ------------------------------------------------------------------
    # Inferencia de una imagen
    # ------------------------------------------------------------------
    def infer_one(
        self,
        image: Image.Image,
        prompt_id: str,
        image_filename: str | None = None,
        split: str | None = None,
        output_attentions: bool = False,
    ) -> dict[str, Any]:
        """Ejecuta la pasada single-pass sobre una imagen con un prompt.

        Si se pasan image_filename y split, se calculan pesos ROI desde la
        máscara de disco y se añaden las columnas de ablación roi_weighted.
        Si output_attentions=True, se añaden las columnas de ablación
        attention_weighted (deployable, sin máscaras externas).
        """
        prompt = self.build_prompt(prompt_id)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=output_attentions,
                return_dict_in_generate=True,
            )

        roi_weights = None
        if image_filename is not None and split is not None:
            roi_weights = compute_roi_weights(
                self.cfg, image_filename, split,
                grid_size=int(np.sqrt(self.cfg.inference.num_image_tokens)),
            )

        attentions = out.attentions if output_attentions else None
        signals = self._extract_signals(out, inputs, roi_weights=roi_weights, attentions=attentions)
        return signals

    # ------------------------------------------------------------------
    # Heatmap de atención
    # ------------------------------------------------------------------
    def generate_attention_heatmap(
        self,
        image: Image.Image,
        prompt_id: str,
        layer: int = 34,
        out_path: str | Path | None = None,
    ) -> Image.Image:
        """Ejecuta inferencia y genera un heatmap de atención superpuesto.

        Devuelve la imagen PIL con el heatmap. Si out_path se pasa, también
        la guarda en disco.
        """
        prompt = self.build_prompt(prompt_id)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        # Extraer pesos de atención
        input_ids = inputs["input_ids"][0]
        img_token_id = self.cfg.tokens.image_token_index
        img_positions = (input_ids == img_token_id).nonzero().flatten()

        attn_weights = compute_attention_weights(
            out.attentions, layer, img_positions, seq_len=input_ids.shape[0]
        )

        # Generar heatmap
        heatmap = generate_attention_heatmap(
            image, attn_weights,
            grid_size=int(np.sqrt(self.cfg.inference.num_image_tokens)),
        )

        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            heatmap.save(out_path)

        return heatmap

    # ------------------------------------------------------------------
    # Self-consistency (baseline multi-pass)
    # ------------------------------------------------------------------
    def infer_self_consistency(
        self,
        image: Image.Image,
        prompt_id: str,
        n_samples: int = 10,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Muestrea n respuestas a temperatura T y devuelve la fracción de 'yes'."""
        prompt = self.build_prompt(prompt_id)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        yes_count = 0
        with torch.inference_mode():
            for _ in range(n_samples):
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=temperature,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                scores0 = out.scores[0]
                logit_yes = float(scores0[0, self.cfg.tokens.yes].item())
                logit_no = float(scores0[0, self.cfg.tokens.no].item())
                if logit_yes > logit_no:
                    yes_count += 1

        frac_yes = yes_count / n_samples
        # Entropía de la frecuencia como u(x)
        eps = self.cfg.uncertainty.epsilon
        entropy_sc = -(
            frac_yes * np.log(frac_yes + eps) + (1 - frac_yes) * np.log(1 - frac_yes + eps)
        )
        return {
            "self_consistency_frac_yes": frac_yes,
            "self_consistency_entropy": float(entropy_sc),
        }


# ------------------------------------------------------------------------------
# Carga de imágenes desde master_table
# ------------------------------------------------------------------------------
def load_image(cfg: Config, image_filename: str, split: str) -> Image.Image:
    """Carga una imagen desde data/mm_odir_129/{split}/{image_filename}."""
    path = Path(cfg.paths.data) / "mm_odir_129" / split / image_filename
    if not path.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {path}")
    return Image.open(path).convert("RGB")


# ------------------------------------------------------------------------------
# Guardado incremental
# ------------------------------------------------------------------------------
def append_result(cfg: Config, row: dict[str, Any], filename: str = "results_full.csv") -> None:
    """Añade una fila al CSV de resultados (crea con header si no existe)."""
    path = Path(cfg.paths.results) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def to_long_format(
    row: dict[str, Any],
    signal_prefixes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convierte un registro de formato ancho a formato largo.

    Formato ancho: una fila con muchas columnas (kl_v_t_L34_tau1_mean, ...).
    Formato largo: una fila por señal (image_filename, prompt_id, signal_type, layer, tau, pooling, value).

    Args:
        row: registro en formato ancho.
        signal_prefixes: prefijos de columnas de señal a extraer (default: kl_, jsd_, cosine_).

    Returns:
        Lista de registros en formato largo.
    """
    signal_prefixes = signal_prefixes or ["kl_", "jsd_", "cosine_"]

    # Metadatos (columnas que no son señales)
    meta_cols = [
        "image_filename", "patient_id", "prompt_id", "split",
        "logit_yes", "logit_no", "p_yes", "pred", "label", "correct",
        "entropy_answer", "msp_answer", "energy_answer",
        "inference_ms", "seed",
    ]

    base_record = {k: v for k, v in row.items() if k in meta_cols}

    long_records = []

    # Extraer señales
    for key, value in row.items():
        if not any(key.startswith(p) for p in signal_prefixes):
            continue
        if key in meta_cols:
            continue

        # Parsear nombre de señal: {signal_type}_L{layer}_tau{tau}_{pooling}
        # Ej: kl_v_t_L34_tau1.0_mean -> signal_type=kl_v_t, layer=34, tau=1.0, pooling=mean
        parts = key.split("_")
        if key.startswith("kl_"):
            signal_type = "_".join(parts[:3])  # kl_v_t o kl_t_v
            layer = parts[3].replace("L", "")
            tau = parts[4].replace("tau", "")
            pooling = parts[5] if len(parts) > 5 else "mean"
        elif key.startswith("jsd_"):
            signal_type = "jsd"
            layer = parts[1].replace("L", "")
            tau = parts[2].replace("tau", "")
            pooling = parts[3] if len(parts) > 3 else "mean"
        elif key.startswith("cosine_"):
            signal_type = "cosine"
            layer = parts[1].replace("L", "")
            tau = parts[2].replace("tau", "")
            pooling = parts[3] if len(parts) > 3 else "mean"
        else:
            continue

        long_record = {
            **base_record,
            "signal_type": signal_type,
            "layer": layer,
            "tau": tau,
            "pooling": pooling,
            "value": value,
        }
        long_records.append(long_record)

    return long_records


# ------------------------------------------------------------------------------
# Piloto con sanity checks
# ------------------------------------------------------------------------------
def run_pilot(cfg: Config, n: int = 20, with_attentions: bool = False, seed: int | None = None, save: bool = True) -> pd.DataFrame:
    """Ejecuta el piloto de n imágenes con sanity checks obligatorios.

    Si with_attentions=True, extrae atenciones cruzadas y genera heatmaps
    de ejemplo para las primeras 3 imágenes. Si seed se pasa, se añade como
    columna a los resultados. Si save=False, no escribe el CSV y devuelve el DataFrame.
    """
    print("=" * 70)
    print("PILOTO — sanity checks obligatorios")
    print("=" * 70)

    # Asegurar que el dataset está descargado
    download_dataset(cfg)

    # Cargar master_table para seleccionar n imágenes balanceadas
    master = pd.read_csv(cfg.paths.master_table)
    per_class = n // 2
    pilot_df = pd.concat([
        master[master["label"] == 0].sample(n=min(per_class, (master["label"] == 0).sum()), random_state=cfg.experiment.seed),
        master[master["label"] == 1].sample(n=min(per_class, (master["label"] == 1).sum()), random_state=cfg.experiment.seed),
    ]).reset_index(drop=True)
    print(f"[pilot] {len(pilot_df)} imágenes seleccionadas")

    pipeline = MedGemmaInference(cfg)
    pipeline.load()

    # Sanity check #1: KL(p‖p) == 0
    p_test = torch.softmax(torch.randn(2560), dim=0)
    kl_same = kl_div(p_test, p_test)
    print(f"[{'PASS' if abs(kl_same) < 1e-6 else 'FAIL'}] KL(p‖p) == 0 — obtenido: {kl_same}")

    # Sanity check #3: máscara de imagen (verificación con una imagen real)
    first = pilot_df.iloc[0]
    img = load_image(cfg, first["image_filename"], first["split"])
    prompt = pipeline.build_prompt("p1")
    inputs = pipeline.processor(images=img, text=prompt, return_tensors="pt")
    img_mask = inputs["input_ids"][0] == cfg.tokens.image_token_index
    n_img = int(img_mask.sum())
    pos = img_mask.nonzero().flatten()
    contiguous = bool((pos.diff() == 1).all()) if len(pos) > 1 else True
    print(f"[{'PASS' if n_img == 256 and contiguous else 'FAIL'}] Máscara: 256 tokens contiguos — {n_img}, contiguos={contiguous}")

    # Ejecutar inferencia en el piloto
    results = []
    p_yes_values = []
    kl_values = []
    black_kl = None

    for _, row in pilot_df.iterrows():
        img = load_image(cfg, row["image_filename"], row["split"])
        for prompt_id in ["p1", "p4"]:
            t0 = time.time()
            signals = pipeline.infer_one(
                img, prompt_id,
                image_filename=row["image_filename"],
                split=row["split"],
                output_attentions=with_attentions,
            )
            elapsed_ms = (time.time() - t0) * 1000

            record = {
                "image_filename": row["image_filename"],
                "patient_id": row["patient_id"],
                "prompt_id": prompt_id.upper(),
                "split": row["split"],
                **signals,
                "label": int(row["label"]),
                "correct": int(signals["pred"] == row["label"]),
                "inference_ms": elapsed_ms,
            }
            if seed is not None:
                record["seed"] = seed
            results.append(record)
            p_yes_values.append(signals["p_yes"])

            # Guardar KL primaria para checks
            kl_key = "kl_v_t_L34_tau1.0_mean"
            if kl_key in signals:
                kl_values.append(signals[kl_key])

            # Sanity #4: P(yes) + P(no) ≈ 1
            p_no = 1 - signals["p_yes"]
            assert abs(signals["p_yes"] + p_no - 1.0) < 1e-6, "P(yes)+P(no) != 1"

    # Sanity #2: misma imagen dos veces → u(x) idéntico
    img = load_image(cfg, pilot_df.iloc[0]["image_filename"], pilot_df.iloc[0]["split"])
    s1 = pipeline.infer_one(img, "p1")
    s2 = pipeline.infer_one(img, "p1")
    kl1 = s1.get("kl_v_t_L34_tau1.0_mean", np.nan)
    kl2 = s2.get("kl_v_t_L34_tau1.0_mean", np.nan)
    print(f"[{'PASS' if np.isclose(kl1, kl2, rtol=1e-5) else 'FAIL'}] Reproducibilidad — KL1={kl1:.6f}, KL2={kl2:.6f}")

    # Sanity #5: imagen negra → KL visiblemente alta
    black = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
    s_black = pipeline.infer_one(black, "p1")
    black_kl = s_black.get("kl_v_t_L34_tau1.0_mean", np.nan)
    if kl_values:
        mean_kl = float(np.nanmean(kl_values))
        status = "PASS" if black_kl > mean_kl else "WARN"
        print(f"[{status}] Imagen negra KL={black_kl:.4f} vs media={mean_kl:.4f}")
    else:
        print(f"[WARN] No hay valores KL para comparar — imagen negra KL={black_kl:.4f}")

    # Sanity #6: P(yes) no colapsada
    p_yes_arr = np.array(p_yes_values)
    collapsed = (p_yes_arr < 0.01).all() or (p_yes_arr > 0.99).all()
    print(f"[{'FAIL' if collapsed else 'PASS'}] P(yes) no colapsada — rango [{p_yes_arr.min():.3f}, {p_yes_arr.max():.3f}]")

    # Generar heatmaps de atención de ejemplo (si está activado)
    if with_attentions:
        print("\n[pilot] Generando heatmaps de atención de ejemplo...")
        for _, row in pilot_df.head(3).iterrows():
            img = load_image(cfg, row["image_filename"], row["split"])
            heatmap_path = Path(cfg.paths.figures) / f"heatmap_{row['image_filename'].replace('.jpg', '.png')}"
            pipeline.generate_attention_heatmap(img, "p1", layer=34, out_path=heatmap_path)
            print(f"  → {heatmap_path}")

    # Guardar piloto en formato largo
    long_records = []
    for record in results:
        long_records.extend(to_long_format(record))

    df = pd.DataFrame(long_records)
    if save:
        out_path = Path(cfg.paths.results) / "results_pilot.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n[pilot] Guardado: {out_path} ({len(df)} filas, formato largo)")
    return df

    # Accuracy base (primer número a mirar)
    acc = df["correct"].mean()
    print(f"[pilot] Accuracy base de MedGemma: {acc:.3f}")
    if acc > 0.85:
        print("[WARN] Accuracy > 85%: activar vigilancia de 'pocos errores' (§8.3)")


# ------------------------------------------------------------------------------
# Corrida completa
# ------------------------------------------------------------------------------
def run_full(cfg: Config, with_attentions: bool = False, seed: int | None = None, save: bool = True) -> pd.DataFrame:
    """Ejecuta la corrida completa: 129 imágenes × 2 prompts = 258 inferencias.

    Si with_attentions=True, extrae atenciones cruzadas y añade columnas
    *_attn (deployable, sin máscaras externas). Usa eager attention, más
    lento y con más memoria que sdpa. Si seed se pasa, se añade como columna
    a los resultados. Si save=False, no escribe el CSV y devuelve el DataFrame.
    """
    # Asegurar que el dataset está descargado
    download_dataset(cfg)

    master = pd.read_csv(cfg.paths.master_table)
    pipeline = MedGemmaInference(cfg)
    pipeline.load()

    print(f"[full] {len(master)} imágenes × 2 prompts = {len(master) * 2} inferencias")
    if with_attentions:
        print("[full] Atenciones activadas (eager attention)")

    results = []

    for _, row in master.iterrows():
        try:
            img = load_image(cfg, row["image_filename"], row["split"])
        except FileNotFoundError as exc:
            warnings.warn(str(exc))
            continue
        for prompt_id in ["p1", "p4"]:
            t0 = time.time()
            try:
                signals = pipeline.infer_one(
                    img, prompt_id,
                    image_filename=row["image_filename"],
                    split=row["split"],
                    output_attentions=with_attentions,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Error en {row['image_filename']} {prompt_id}: {exc}")
                continue
            elapsed_ms = (time.time() - t0) * 1000

            # Verificación robusta de label y correct
            label_val = row["label"]
            if pd.isna(label_val):
                warnings.warn(f"label NaN para {row['image_filename']} en master_table")
                continue
            label_val = int(label_val)

            correct_val = int(signals["pred"] == label_val)

            record = {
                "image_filename": row["image_filename"],
                "patient_id": row["patient_id"],
                "prompt_id": prompt_id.upper(),
                "split": row["split"],
                **signals,
                "label": label_val,
                "correct": correct_val,
                "inference_ms": elapsed_ms,
            }
            if seed is not None:
                record["seed"] = seed
            results.append(record)
            print(f"[full] {row['image_filename']} {prompt_id}: p_yes={signals['p_yes']:.3f} correct={record['correct']}")

    # Escribir todo al final en formato largo (cada fila = observación × variante de señal)
    long_records = []
    for record in results:
        long_records.extend(to_long_format(record))

    df = pd.DataFrame(long_records)
    if save:
        out_path = Path(cfg.paths.results) / "results_full.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n[full] Guardado: {out_path} ({len(df)} filas, formato largo)")
    return df


# ------------------------------------------------------------------------------
# Self-consistency (50 imágenes × 10 muestras)
# ------------------------------------------------------------------------------
def run_self_consistency(cfg: Config, n_images: int = 50, n_samples: int = 10, seed: int | None = None) -> None:
    """Ejecuta el baseline multi-pass sobre un subconjunto estratificado.

    Si seed se pasa, se usa para la selección del subconjunto y se añade como
    columna a los resultados.
    """
    master = pd.read_csv(cfg.paths.master_table)
    random_state = seed if seed is not None else cfg.experiment.seed
    subset = (
        master.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min(n_images // 2, len(x)), random_state=random_state))
        .reset_index(drop=True)
    )

    pipeline = MedGemmaInference(cfg)
    pipeline.load()

    print(f"[sc] {len(subset)} imágenes × {n_samples} muestras × 2 prompts")

    for _, row in subset.iterrows():
        img = load_image(cfg, row["image_filename"], row["split"])
        for prompt_id in ["p1", "p4"]:
            signals = pipeline.infer_self_consistency(
                img, prompt_id, n_samples=n_samples, temperature=0.7
            )
            record = {
                "image_filename": row["image_filename"],
                "patient_id": row["patient_id"],
                "prompt_id": prompt_id.upper(),
                "split": row["split"],
                **signals,
                "label": int(row["label"]),
            }
            if seed is not None:
                record["seed"] = seed
            append_result(cfg, record, filename="results_self_consistency.csv")
            print(f"[sc] {row['image_filename']} {prompt_id}: frac_yes={signals['self_consistency_frac_yes']:.2f}")


# ------------------------------------------------------------------------------
# Repeticiones con múltiples semillas
# ------------------------------------------------------------------------------
def run_with_seeds(cfg: Config, seeds: list[int], mode: str, **kwargs) -> None:
    """Ejecuta la inferencia para cada semilla en la lista.

    Para cada semilla:
        1. Fija la semilla (random, numpy, torch).
        2. Ejecuta el modo especificado (pilot, full, self-consistency).
        3. Acumula los resultados con la semilla como columna.
        4. Al final, guarda un solo CSV con todas las semillas.

    Args:
        cfg: configuración del experimento.
        seeds: lista de semillas (p. ej. [42, 123, 456]).
        mode: "pilot", "full" o "self-consistency".
        **kwargs: argumentos adicionales para la función de corrida.
    """
    all_results = []

    for seed in seeds:
        print("\n" + "=" * 70)
        print(f"CORRIDA CON SEMILLA {seed}")
        print("=" * 70)
        cfg.set_seed(seed)

        if mode == "pilot":
            df = run_pilot(cfg, seed=seed, save=False, **kwargs)
            all_results.append(df)
        elif mode == "full":
            df = run_full(cfg, seed=seed, save=False, **kwargs)
            all_results.append(df)
        elif mode == "self-consistency":
            run_self_consistency(cfg, seed=seed, **kwargs)
        else:
            raise ValueError(f"Modo desconocido: {mode}")

    # Guardar un solo CSV con todas las semillas (para pilot y full)
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        filename = "results_pilot.csv" if mode == "pilot" else "results_full.csv"
        out_path = Path(cfg.paths.results) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(out_path, index=False)
        print(f"\n[seeds] Guardado: {out_path} ({len(df_all)} filas, {len(seeds)} semillas)")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de inferencia MedGemma-4B")
    parser.add_argument("--pilot", action="store_true", help="Ejecutar piloto con sanity checks")
    parser.add_argument("--run-full", action="store_true", help="Ejecutar corrida completa")
    parser.add_argument("--self-consistency", action="store_true", help="Ejecutar baseline multi-pass")
    parser.add_argument("--n", type=int, default=20, help="Imágenes para el piloto")
    parser.add_argument("--attentions", action="store_true",
                        help="Extraer atenciones cruzadas y generar heatmaps (piloto)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Lista de semillas para repeticiones (p. ej. --seeds 42 123 456)")
    args = parser.parse_args()

    cfg = Config()
    cfg.ensure_paths()
    cfg.set_determinism()

    # Si se pasan semillas, ejecutar repeticiones
    if args.seeds:
        if args.pilot:
            run_with_seeds(cfg, args.seeds, mode="pilot", n=args.n, with_attentions=args.attentions)
        elif args.run_full:
            run_with_seeds(cfg, args.seeds, mode="full", with_attentions=args.attentions)
        elif args.self_consistency:
            run_with_seeds(cfg, args.seeds, mode="self-consistency")
        else:
            print("Uso: python -m src.inference [--pilot | --run-full | --self-consistency] --seeds 42 123 456")
        return

    # Corrida simple (sin semillas)
    cfg.set_seed()
    if args.pilot:
        run_pilot(cfg, n=args.n, with_attentions=args.attentions)
    elif args.run_full:
        run_full(cfg, with_attentions=args.attentions)
    elif args.self_consistency:
        run_self_consistency(cfg)
    else:
        print("Uso: python -m src.inference [--pilot | --run-full | --self-consistency]")
        print("Ejemplo: python -m src.inference --pilot --n 20")
        print("Con semillas: python -m src.inference --run-full --seeds 42 123 456")


if __name__ == "__main__":
    main()
