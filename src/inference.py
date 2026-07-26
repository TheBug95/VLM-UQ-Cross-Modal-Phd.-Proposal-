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
from src.uncertainty import compute_roi_weights, roi_weighted_pooling

# Columnas de salida (orden congelado por el diseño)
BASE_COLUMNS = [
    "image_filename", "patient_id", "prompt_id", "split",
    "logit_yes", "logit_no", "p_yes", "pred", "label", "correct",
    "entropy_answer", "msp_answer", "energy_answer",
]


# ------------------------------------------------------------------------------
# Utilidades de incertidumbre
# ------------------------------------------------------------------------------
def to_distribution(vec: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Convierte un vector crudo a distribución con F.log_softmax en float64.

    Regla numérica dura (ver AGENTS.md §5): las massive activations de Gemma
    colapsan softmax float32 a ceros exactos. float64 lo evita.
    """
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
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        model_name = self.cfg.model.name
        kwargs: dict[str, Any] = {}

        if self.device == "cuda":
            kwargs["device_map"] = self.cfg.model.device_map
        if self.cfg.model.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kwargs["torch_dtype"] = getattr(torch, self.cfg.model.torch_dtype)

        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
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

        for layer in self.cfg.inference.layers:
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

                    # Ablación prompt (solo L34, tau=1, mean según diseño)
                    if (
                        layer == 34
                        and tau == 1.0
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
    ) -> dict[str, Any]:
        """Ejecuta la pasada single-pass sobre una imagen con un prompt.

        Si se pasan image_filename y split, se calculan pesos ROI desde la
        máscara de disco y se añaden las columnas de ablación roi_weighted.
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
                return_dict_in_generate=True,
            )

        roi_weights = None
        if image_filename is not None and split is not None:
            roi_weights = compute_roi_weights(
                self.cfg, image_filename, split,
                grid_size=int(np.sqrt(self.cfg.inference.num_image_tokens)),
            )

        signals = self._extract_signals(out, inputs, roi_weights=roi_weights)
        return signals

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


# ------------------------------------------------------------------------------
# Piloto con sanity checks
# ------------------------------------------------------------------------------
def run_pilot(cfg: Config, n: int = 20) -> None:
    """Ejecuta el piloto de n imágenes con sanity checks obligatorios."""
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
            results.append(record)
            p_yes_values.append(signals["p_yes"])

            # Guardar KL primaria para checks
            kl_key = f"kl_v_t_L34_tau1_mean"
            if kl_key in signals:
                kl_values.append(signals[kl_key])

            # Sanity #4: P(yes) + P(no) ≈ 1
            p_no = 1 - signals["p_yes"]
            assert abs(signals["p_yes"] + p_no - 1.0) < 1e-6, "P(yes)+P(no) != 1"

    # Sanity #2: misma imagen dos veces → u(x) idéntico
    img = load_image(cfg, pilot_df.iloc[0]["image_filename"], pilot_df.iloc[0]["split"])
    s1 = pipeline.infer_one(img, "p1")
    s2 = pipeline.infer_one(img, "p1")
    kl1 = s1.get("kl_v_t_L34_tau1_mean", np.nan)
    kl2 = s2.get("kl_v_t_L34_tau1_mean", np.nan)
    print(f"[{'PASS' if np.isclose(kl1, kl2, rtol=1e-5) else 'FAIL'}] Reproducibilidad — KL1={kl1:.6f}, KL2={kl2:.6f}")

    # Sanity #5: imagen negra → KL visiblemente alta
    black = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
    s_black = pipeline.infer_one(black, "p1")
    black_kl = s_black.get("kl_v_t_L34_tau1_mean", np.nan)
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

    # Guardar piloto
    df = pd.DataFrame(results)
    out_path = Path(cfg.paths.results) / "results_pilot.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[pilot] Guardado: {out_path} ({len(df)} filas)")

    # Accuracy base (primer número a mirar)
    acc = df["correct"].mean()
    print(f"[pilot] Accuracy base de MedGemma: {acc:.3f}")
    if acc > 0.85:
        print("[WARN] Accuracy > 85%: activar vigilancia de 'pocos errores' (§8.3)")


# ------------------------------------------------------------------------------
# Corrida completa
# ------------------------------------------------------------------------------
def run_full(cfg: Config) -> None:
    """Ejecuta la corrida completa: 129 imágenes × 2 prompts = 258 inferencias."""
    # Asegurar que el dataset está descargado
    download_dataset(cfg)

    master = pd.read_csv(cfg.paths.master_table)
    pipeline = MedGemmaInference(cfg)
    pipeline.load()

    print(f"[full] {len(master)} imágenes × 2 prompts = {len(master) * 2} inferencias")

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
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Error en {row['image_filename']} {prompt_id}: {exc}")
                continue
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
            append_result(cfg, record)
            print(f"[full] {row['image_filename']} {prompt_id}: p_yes={signals['p_yes']:.3f} correct={record['correct']}")


# ------------------------------------------------------------------------------
# Self-consistency (50 imágenes × 10 muestras)
# ------------------------------------------------------------------------------
def run_self_consistency(cfg: Config, n_images: int = 50, n_samples: int = 10) -> None:
    """Ejecuta el baseline multi-pass sobre un subconjunto estratificado."""
    master = pd.read_csv(cfg.paths.master_table)
    subset = (
        master.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min(n_images // 2, len(x)), random_state=cfg.experiment.seed))
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
            append_result(cfg, record, filename="results_self_consistency.csv")
            print(f"[sc] {row['image_filename']} {prompt_id}: frac_yes={signals['self_consistency_frac_yes']:.2f}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de inferencia MedGemma-4B")
    parser.add_argument("--pilot", action="store_true", help="Ejecutar piloto con sanity checks")
    parser.add_argument("--run-full", action="store_true", help="Ejecutar corrida completa")
    parser.add_argument("--self-consistency", action="store_true", help="Ejecutar baseline multi-pass")
    parser.add_argument("--n", type=int, default=20, help="Imágenes para el piloto")
    args = parser.parse_args()

    cfg = Config()
    cfg.ensure_paths()
    cfg.set_seed()
    cfg.set_determinism()

    if args.pilot:
        run_pilot(cfg, n=args.n)
    elif args.run_full:
        run_full(cfg)
    elif args.self_consistency:
        run_self_consistency(cfg)
    else:
        print("Uso: python -m src.inference [--pilot | --run-full | --self-consistency]")
        print("Ejemplo: python -m src.inference --pilot --n 20")


if __name__ == "__main__":
    main()
