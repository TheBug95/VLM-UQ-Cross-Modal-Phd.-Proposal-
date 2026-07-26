"""src/config.py — Carga centralizada de hiperparámetros.

Uso típico:
    from src.config import Config
    cfg = Config()                       # carga config.yaml
    cfg.resolve_model_ids()              # intenta sobreescribir IDs desde HF
    print(cfg.model.name)
    print(cfg.tokens.image_token_index)

La configuración de diseño vive en config.yaml (hiperparámetros, prompts,
capas, temperaturas, etc.). Los IDs de tokens especiales se pueden resolver
automáticamente desde el tokenizer de HuggingFace; si falla, se conservan los
valores hardcodeados de config.yaml como fallback.
"""
from __future__ import annotations

import os
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _project_root() -> Path:
    """Raíz del proyecto: dos niveles arriba de src/config.py."""
    return Path(__file__).resolve().parent.parent


class DotDict(dict):
    """Diccionario que permite acceso con punto: cfg.paths.data

    Los dicts anidados se convierten a DotDict en __init__ para que las
    asignaciones anidadas (cfg.paths.results = "x") persistan en el original.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, DotDict):
                self[key] = DotDict(value)

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(f"Config no tiene clave '{item}'") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        self[key] = value


class Config:
    """Configuración del experimento con resolución automática de IDs."""

    def __init__(self, path: str | Path | None = None):
        self.root = _project_root()
        self.path = Path(path) if path else self.root / "config.yaml"
        with open(self.path, "r", encoding="utf-8") as f:
            self._data = DotDict(yaml.safe_load(f))
        self._resolve_paths()

    # ------------------------------------------------------------------
    # Acceso
    # ------------------------------------------------------------------
    def __getattr__(self, item: str) -> Any:
        return getattr(self._data, item)

    def __getitem__(self, item: str) -> Any:
        return self._data[item]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict:
        """Devuelve una copia plana del diccionario subyacente."""
        return dict(self._data)

    # ------------------------------------------------------------------
    # Resolución de paths
    # ------------------------------------------------------------------
    def _resolve_paths(self) -> None:
        """Convierte paths relativos a absolutos desde la raíz del proyecto."""
        root = Path(self.paths.root)
        if not root.is_absolute():
            root = self.root / root
        root = root.resolve()

        self._data.paths.root = str(root)
        for key in ["data", "results", "figures", "artifacts"]:
            if key in self._data.paths:
                self._data.paths[key] = str(root / self._data.paths[key])

        for key in ["master_table", "results_full", "results_pilot"]:
            if key in self._data.paths:
                self._data.paths[key] = str(root / self._data.paths[key])

    def ensure_paths(self) -> None:
        """Crea los directorios de salida si no existen."""
        for key in ["data", "results", "figures", "artifacts"]:
            path = Path(self.paths[key])
            path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolución automática de IDs desde el modelo/tokenizer
    # ------------------------------------------------------------------
    def resolve_model_ids(
        self,
        model_name: str | None = None,
        force: bool = False,
        quiet: bool = False,
    ) -> None:
        """Intenta cargar el tokenizer y config de HF para extraer IDs reales.

        Args:
            model_name: modelo a cargar; por defecto usa cfg.model.name.
            force: si es True, sobrescribe incluso IDs ya distintos de los
                valores por defecto.
            quiet: si es True, no imprime advertencias.
        """
        try:
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as exc:
            if not quiet:
                warnings.warn(
                    "transformers no instalado; se usan IDs hardcodeados de config.yaml. "
                    f"Error: {exc}"
                )
            return

        model_name = model_name or self.model.name
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            config = AutoConfig.from_pretrained(model_name)
        except Exception as exc:  # noqa: BLE001
            if not quiet:
                warnings.warn(
                    f"No se pudo cargar '{model_name}' desde HuggingFace "
                    f"({type(exc).__name__}: {exc}). Se usan IDs hardcodeados."
                )
            return

        # Diccionario de valores candidatos: (nombre_en_yaml, valor_extraído)
        candidates = {
            "image_soft_token": tokenizer.convert_tokens_to_ids("<image_soft_token>"),
            "start_of_image": tokenizer.convert_tokens_to_ids("<start_of_image>"),
            "end_of_image": tokenizer.convert_tokens_to_ids("<end_of_image>"),
            "yes": tokenizer.encode("yes", add_special_tokens=False),
            "no": tokenizer.encode("no", add_special_tokens=False),
        }

        # yes/no pueden venir como lista; nos quedamos con el primer elemento
        for key in ("yes", "no"):
            value = candidates[key]
            if isinstance(value, list) and len(value) == 1:
                candidates[key] = value[0]
            elif isinstance(value, list) and len(value) > 1:
                if not quiet:
                    warnings.warn(
                        f"'{key}' se tokeniza en múltiples IDs {value}; "
                        "se usa el primero."
                    )
                candidates[key] = value[0]
            elif not isinstance(value, int):
                if not quiet:
                    warnings.warn(f"ID inesperado para '{key}': {value}; se ignora.")
                del candidates[key]

        # image_token_index suele estar en config
        image_token_index = getattr(config, "image_token_index", None)
        if image_token_index is not None:
            candidates["image_token_index"] = image_token_index

        # Sobreescribir solo si es seguro o si force=True
        for key, value in candidates.items():
            if not isinstance(value, int):
                continue
            current = self._data.tokens.get(key)
            if current != value and (force or current == self._default_token(key)):
                self._data.tokens[key] = value
                if not quiet:
                    print(f"[config] {key}: {current} -> {value} (desde {model_name})")

        # Actualizar conteo de tokens de imagen si se puede inferir del modelo
        vision_tokens = getattr(config, "vision_soft_tokens_per_image", None)
        if vision_tokens is not None and isinstance(vision_tokens, int):
            self._data.inference.num_image_tokens = vision_tokens
            if not quiet:
                print(f"[config] num_image_tokens -> {vision_tokens} (desde config)")

    @staticmethod
    def _default_token(key: str) -> Any:
        """Valores por defecto de config.yaml para detectar si fueron editados."""
        defaults = {
            "yes": 4443,
            "no": 1904,
            "image_soft_token": 262144,
            "start_of_image": 255999,
            "end_of_image": 256000,
            "image_token_index": 262144,
        }
        return defaults.get(key)

    # ------------------------------------------------------------------
    # Reproducibilidad
    # ------------------------------------------------------------------
    def set_seed(self, seed: int | None = None) -> None:
        """Fija semillas de random, numpy y torch (si está disponible)."""
        seed = seed if seed is not None else self.experiment.seed
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def set_determinism(self) -> None:
        """Aplica flags de determinismo para CUDA si están disponibles."""
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import torch
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except ImportError:
            pass


# ------------------------------------------------------------------------------
# Helper para importaciones rápidas
# ------------------------------------------------------------------------------
def get_config(resolve_ids: bool = False, quiet: bool = False) -> Config:
    """Devuelve una instancia de Config lista para usar.

    Args:
        resolve_ids: si True, intenta cargar el tokenizer real para resolver IDs.
        quiet: suprime advertencias de resolución.
    """
    cfg = Config()
    cfg.ensure_paths()
    if resolve_ids:
        cfg.resolve_model_ids(quiet=quiet)
    return cfg


if __name__ == "__main__":
    cfg = get_config(resolve_ids=False)
    print("Configuración cargada:")
    print(f"  modelo : {cfg.model.name}")
    print(f"  dataset: {cfg.dataset.name}")
    print(f"  capas  : {cfg.inference.layers}")
    print(f"  prompts: {list(cfg.prompts.keys())}")
    print(f"  tokens (fallback): {dict(cfg.tokens)}")
