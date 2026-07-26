"""src/data.py — Descarga y construcción de la tabla maestra (master_table.csv).

Qué hace:
    1. Descarga MM-ODIR-129 vía huggingface_hub.snapshot_download.
    2. Lee annotations.json + split.json.
    3. Construye un DataFrame con una fila por imagen y los metadatos exigidos
       por el diseño experimental.
    4. Ejecuta una auditoría heurística de artefactos de anotación (flechas
       negras quemadas en píxeles) y marca has_annotation_artifact.
    5. Guarda data/master_table.csv y reporta sanity checks.

Nota de privacidad: split.json expone doctor_name (PII del anotador). Este
script nunca lo copia ni lo imprime.

Uso:
    python -m src.data [--skip-audit] [--audit-threshold 25]
"""
from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from src.config import Config

# Columnas finales de master_table.csv (orden congelado por el diseño)
COLUMNS = [
    "image_filename",
    "patient_id",
    "eye",
    "label",
    "split",
    "transcription",
    "cdr_grade",
    "neuroretinal_rim",
    "disc_hemorrhage",
    "peripapillary_atrophy",
    "rnfl_defect",
    "disc_pallor",
    "vessel_changes",
    "has_masks",
    "has_annotation_artifact",
]

# Mapeo verificado: el campo del diseño cdr_grade es cup_to_disc_ratio en el dataset
GRADING_KEYS = {
    "cdr_grade": "cup_to_disc_ratio",
    "neuroretinal_rim": "neuroretinal_rim",
    "disc_hemorrhage": "disc_hemorrhage",
    "peripapillary_atrophy": "peripapillary_atrophy",
    "rnfl_defect": "rnfl_defect",
    "disc_pallor": "disc_pallor",
    "vessel_changes": "vessel_changes",
}

# Imágenes con artefacto de anotación confirmado visualmente (análisis 21-jul-2026)
KNOWN_ARTIFACTS = {"1281_right.jpg"}


# ------------------------------------------------------------------------------
# Descarga
# ------------------------------------------------------------------------------
def download_dataset(cfg: Config, local_dir: str | Path | None = None) -> Path:
    """Descarga el dataset completo en data/mm_odir_129.

    En Windows + Google Drive/Dropbox, `snapshot_download(local_dir=...)` falla
    al crear sus archivos temporales. Por eso se descarga primero al caché
    estándar de HuggingFace (fuera de Drive) y luego se copia el contenido
    necesario a `data/mm_odir_129`.

    Si el directorio ya existe y contiene annotations.json + split.json, lo
    reutiliza sin volver a copiar.
    """
    local_dir = Path(local_dir or Path(cfg.paths.data) / "mm_odir_129")
    local_dir.mkdir(parents=True, exist_ok=True)

    required = {"annotations.json", "split.json"}
    if required.issubset({p.name for p in local_dir.iterdir()}):
        print(f"[data] Dataset ya presente en {local_dir}")
        return local_dir

    print(f"[data] Descargando {cfg.dataset.name} al caché de HuggingFace")
    cache_dir = snapshot_download(
        repo_id=cfg.dataset.name,
        repo_type="dataset",
        ignore_patterns=["**/desktop.ini", "**/.DS_Store", "**/Thumbs.db"],
    )
    cache_path = Path(cache_dir)
    print(f"[data] Copiando desde caché -> {local_dir}")

    # Copiar estructura completa excepto metadatos de Windows y caché interno
    for item in cache_path.iterdir():
        if item.name in {"desktop.ini", ".DS_Store", "Thumbs.db"}:
            continue
        if item.name == ".cache":
            continue
        dest = local_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    return local_dir


# ------------------------------------------------------------------------------
# Lectura de anotaciones
# ------------------------------------------------------------------------------
def load_annotations(local_dir: Path) -> list[dict[str, Any]]:
    """Lee annotations.json (lista de 129 dicts)."""
    path = local_dir / "annotations.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split_map(local_dir: Path) -> dict[str, str]:
    """Lee split.json y devuelve {image_filename: split}.

    split.json es un dict {train, validation, test}; cada entrada contiene
    anotaciones completas o al menos image_filename. Se usa solo para mapear
    imagen -> split. Nunca se copia doctor_name.
    """
    path = local_dir / "split.json"
    with open(path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    mapping: dict[str, str] = {}
    for split_name, entries in splits.items():
        for entry in entries:
            fname = entry.get("image_filename") or entry.get("filename")
            if fname:
                mapping[fname] = split_name
    return mapping


# ------------------------------------------------------------------------------
# Parseo y extracción
# ------------------------------------------------------------------------------
def parse_filename(image_filename: str) -> tuple[str, str]:
    """Extrae patient_id y eye de '{patient_id}_{eye}.jpg'."""
    stem = Path(image_filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Nombre de archivo inesperado: {image_filename}")
    patient_id, eye = parts
    return patient_id, eye


def extract_gradings(entry: dict[str, Any]) -> dict[str, Any]:
    """Extrae los 7 gradings de locs_data.glaucoma con manejo defensivo."""
    locs = entry.get("locs_data") or {}
    if isinstance(locs, dict):
        gradings = locs.get("glaucoma") or {}
    else:
        gradings = {}

    out: dict[str, Any] = {}
    for target_key, source_key in GRADING_KEYS.items():
        value = gradings.get(source_key)
        # Normalizar: ordinal entero 0-4 o NaN
        if value is None:
            out[target_key] = np.nan
            continue
        try:
            ivalue = int(float(value))
        except (TypeError, ValueError):
            out[target_key] = np.nan
            continue
        out[target_key] = ivalue if 0 <= ivalue <= 4 else np.nan
    return out


def check_masks(local_dir: Path, image_filename: str) -> bool:
    """True si existen las máscaras _cup y _disc para la imagen."""
    stem = Path(image_filename).stem
    split = None
    # Buscar en qué carpeta está la imagen
    for candidate in ("train", "validation", "test"):
        if (local_dir / candidate / image_filename).exists():
            split = candidate
            break
    if split is None:
        return False
    cup = local_dir / split / f"{stem}_cup.png"
    disc = local_dir / split / f"{stem}_disc.png"
    return cup.exists() and disc.exists()


# ------------------------------------------------------------------------------
# Auditoría de artefactos de anotación (heurística)
# ------------------------------------------------------------------------------
def _load_image_rgb(path: Path) -> np.ndarray:
    """Carga una imagen JPEG como array RGB uint8."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _detect_black_marker(
    img: np.ndarray,
    threshold: int = 25,
    min_area: int = 30,
    max_area: int = 800,
    density_threshold: float = 0.70,
) -> bool:
    """Detecta regiones negras pequeñas y densas dentro del fundus.

    Heurística simple:
        1. Convertir a escala de grises.
        2. Estimar el bounding box del fundus (píxeles con intensidad > 20).
        3. Buscar componentes conectados de píxeles muy oscuros (< threshold).
        4. Filtrar por área (min_area, max_area) y densidad (area / bbox_area).
        5. Marcar como sospechosa si al menos un componente cumple todo.

    Advertencia: los vasos sanguíneos normales también son oscuros y ramificados,
    por lo que esta función puede generar falsos positivos. Sirve como triage
    inicial; la confirmación debe ser visual.
    """
    gray = np.mean(img, axis=2)
    fundus_mask = gray > 20
    if not fundus_mask.any():
        return False

    # Bounding box del fundus
    rows = np.any(fundus_mask, axis=1)
    cols = np.any(fundus_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    fundus_region = gray[rmin : rmax + 1, cmin : cmax + 1]

    # Componentes conectados de píxeles muy oscuros
    dark = fundus_region < threshold
    if not dark.any():
        return False

    try:
        from scipy import ndimage
    except ImportError:
        warnings.warn("scipy no disponible; auditoría de artefactos limitada.")
        return False

    labeled, n_features = ndimage.label(dark)
    if n_features == 0:
        return False

    for feature_id in range(1, n_features + 1):
        ys, xs = np.where(labeled == feature_id)
        area = len(ys)
        if area < min_area or area > max_area:
            continue
        bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        density = area / bbox_area if bbox_area > 0 else 0.0
        if density >= density_threshold:
            return True
    return False


def audit_annotation_artifacts(
    local_dir: Path,
    image_filename: str,
    threshold: int = 25,
) -> bool:
    """True si la imagen parece tener un artefacto de anotación.

    Combina: (a) lista de confirmadas visualmente, (b) detector heurístico.
    """
    if image_filename in KNOWN_ARTIFACTS:
        return True

    # Buscar el archivo en las carpetas de split
    for split in ("train", "validation", "test"):
        path = local_dir / split / image_filename
        if path.exists():
            try:
                img = _load_image_rgb(path)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"No se pudo auditar {image_filename}: {exc}")
                return False
            return _detect_black_marker(img, threshold=threshold)
    return False


# ------------------------------------------------------------------------------
# Construcción de la tabla maestra
# ------------------------------------------------------------------------------
def build_master_table(cfg: Config, skip_audit: bool = False, audit_threshold: int = 25) -> pd.DataFrame:
    """Construye el DataFrame maestro a partir de annotations.json + split.json."""
    local_dir = download_dataset(cfg)
    annotations = load_annotations(local_dir)
    split_map = load_split_map(local_dir)

    rows: list[dict[str, Any]] = []
    for entry in annotations:
        image_filename = entry.get("image_filename")
        if not image_filename:
            warnings.warn(f"Entrada sin image_filename: {entry.get('id', '?')}")
            continue

        patient_id, eye = parse_filename(image_filename)
        label = entry.get("label")
        label_int = 1 if label == cfg.dataset.label_pathological else 0

        split = split_map.get(image_filename)
        if split is None:
            # Fallback: buscar en carpetas
            for candidate in ("train", "validation", "test"):
                if (local_dir / candidate / image_filename).exists():
                    split = candidate
                    break
        if split is None:
            warnings.warn(f"No se pudo determinar split de {image_filename}")
            split = "unknown"

        gradings = extract_gradings(entry)
        has_masks = check_masks(local_dir, image_filename)

        if skip_audit:
            has_artifact = image_filename in KNOWN_ARTIFACTS
        else:
            has_artifact = audit_annotation_artifacts(local_dir, image_filename, audit_threshold)

        row = {
            "image_filename": image_filename,
            "patient_id": patient_id,
            "eye": eye,
            "label": label_int,
            "split": split,
            "transcription": entry.get("transcription", ""),
            **gradings,
            "has_masks": has_masks,
            "has_annotation_artifact": has_artifact,
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=COLUMNS)
    return df


# ------------------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------------------
def sanity_checks(df: pd.DataFrame) -> None:
    """Ejecuta los checks exigidos por la definición experimental."""
    print("\n" + "=" * 70)
    print("SANITY CHECKS — master_table.csv")
    print("=" * 70)

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

    # Sanity #7: conteos
    n_normal = int((df["label"] == 0).sum())
    n_path = int((df["label"] == 1).sum())
    check("60 Normal / 69 Pathological = 129", n_normal == 60 and n_path == 69,
          f"Normal={n_normal}, Pathological={n_path}, total={len(df)}")

    split_counts = df["split"].value_counts().to_dict()
    check("Splits 77/26/26",
          split_counts.get("train") == 77 and split_counts.get("validation") == 26 and split_counts.get("test") == 26,
          f"obtenido: {split_counts}")

    # cdr_grade no nulo en los 69 Pathological
    pat_cdr = df[df["label"] == 1]["cdr_grade"].notna().sum()
    check("cdr_grade no nulo en los 69 Pathological", pat_cdr == 69,
          f"obtenido: {pat_cdr}")

    # cdr_grade ordinal 0-4
    cdr_vals = df["cdr_grade"].dropna().astype(int)
    check("cdr_grade ordinal 0-4", cdr_vals.between(0, 4).all(),
          f"rango: {cdr_vals.min()}–{cdr_vals.max()}")

    # has_annotation_artifact presente
    n_art = int(df["has_annotation_artifact"].sum())
    check("has_annotation_artifact presente", "has_annotation_artifact" in df.columns,
          f"imágenes marcadas: {n_art}")
    print(f"  artefactos por clase: {df.groupby('label')['has_annotation_artifact'].sum().to_dict()}")

    # patient_id sin solape entre splits (advertencia, no fail duro)
    sets = {s: set(df[df["split"] == s]["patient_id"]) for s in ("train", "validation", "test")}
    overlaps = {
        "train∩val": len(sets["train"] & sets["validation"]),
        "train∩test": len(sets["train"] & sets["test"]),
        "val∩test": len(sets["validation"] & sets["test"]),
    }
    if any(overlaps.values()):
        print(f"[WARN] Solape de pacientes entre splits: {overlaps}")
    else:
        print("[PASS] Sin solape de pacientes entre splits")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Construye master_table.csv desde MM-ODIR-129")
    parser.add_argument("--skip-audit", action="store_true",
                        help="Saltar el detector heurístico y marcar solo las conocidas")
    parser.add_argument("--audit-threshold", type=int, default=25,
                        help="Umbral de intensidad para considerar píxel 'negro' (0-255)")
    args = parser.parse_args()

    cfg = Config()
    cfg.ensure_paths()

    print(f"[data] Construyendo master_table.csv (audit={'off' if args.skip_audit else 'on'})")
    df = build_master_table(cfg, skip_audit=args.skip_audit, audit_threshold=args.audit_threshold)

    out_path = Path(cfg.paths.master_table)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[data] Guardado: {out_path} ({len(df)} filas)")

    sanity_checks(df)


if __name__ == "__main__":
    main()
