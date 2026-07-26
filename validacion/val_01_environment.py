"""val_01_environment.py — Validación del entorno de ejecución.

Qué valida:
    V-ENV-1  transformers >= 4.51.3 (la estructura de outputs.hidden_states
             documentada en el plan solo es válida desde esa versión).
    V-ENV-2  torch + CUDA disponibles; nombre y VRAM de la GPU.
    V-ENV-3  Versiones de scipy / sklearn / numpy / pandas / datasets / PIL.
    V-ENV-4  Flags de determinismo (CUBLAS_WORKSPACE_CONFIG, TF32, seeds).

Criterio PASS:
    - transformers >= 4.51.3 (FAIL duro si no: los resultados de val_04 no
      serían interpretables).
    - scipy >= 1.11 (para bootstrap BCa y permutation_test; el Z de
      Mann-Whitney se calcula manualmente — ver val_06).
    - El resto se reporta como WARN (no bloquea, pero queda registrado).

Uso:
    python val_01_environment.py
"""
import os
import sys

FALLOS = []
WARNINGS = []


def check(nombre, condicion, detalle="", duro=False):
    estado = "PASS" if condicion else ("FAIL" if duro else "WARN")
    print(f"[{estado}] {nombre}" + (f" — {detalle}" if detalle else ""))
    if not condicion:
        (FALLOS if duro else WARNINGS).append(nombre)


def version_tupla(v):
    return tuple(int(p) for p in v.split(".")[:3] if p.isdigit())


print("=" * 70)
print("VAL-01 · ENTORNO DE EJECUCIÓN")
print("=" * 70)

# --- V-ENV-3: versiones base -------------------------------------------------
import numpy as np
import pandas as pd
print(f"python    : {sys.version.split()[0]}")
print(f"numpy     : {np.__version__}")
print(f"pandas    : {pd.__version__}")

try:
    import scipy
    print(f"scipy     : {scipy.__version__}")
    check("scipy >= 1.11 (bootstrap BCa, permutation_test)",
          version_tupla(scipy.__version__) >= (1, 11), scipy.__version__, duro=True)
except ImportError:
    check("scipy instalado", False, "no encontrado", duro=True)

try:
    import sklearn
    print(f"sklearn   : {sklearn.__version__}")
    if version_tupla(sklearn.__version__) < (1, 7):
        WARNINGS.append("sklearn<1.7: brier_score_loss usa y_prob (no y_proba)")
        print("[WARN] sklearn < 1.7 — en val_05 usar y_prob= en brier_score_loss")
except ImportError:
    check("sklearn instalado", False, "no encontrado", duro=True)

try:
    import datasets
    print(f"datasets  : {datasets.__version__}")
except ImportError:
    check("datasets instalado", False, "no encontrado", duro=True)

try:
    import PIL
    print(f"PIL       : {PIL.__version__}")
except ImportError:
    check("PIL instalado", False, "necesario para decodificar imágenes", duro=True)

# --- V-ENV-1: transformers ----------------------------------------------------
try:
    import transformers
    print(f"transformers: {transformers.__version__}")
    check("transformers >= 4.51.3", version_tupla(transformers.__version__) >= (4, 51, 3),
          transformers.__version__, duro=True)
except ImportError:
    check("transformers instalado", False, "no encontrado", duro=True)

# --- V-ENV-2: torch / GPU -----------------------------------------------------
try:
    import torch
    print(f"torch     : {torch.__version__}")
    check("CUDA disponible", torch.cuda.is_available(),
          "sin GPU el piloto es inviable en la práctica", duro=False)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        print(f"GPU       : {props.name} · {vram_gb:.1f} GB")
        # Pesos MedGemma-4B ≈ 8.6 GB → bf16 necesita ~10-12 GB con activaciones;
        # NF4 (4-bit) ~4-5 GB. Con < 14 GB libres, usar cuantización 4-bit.
        check("VRAM >= 14 GB (bf16 cómodo) o plan 4-bit", vram_gb >= 14,
              f"{vram_gb:.1f} GB → usar load_in_4bit=True (bitsandbytes NF4)", duro=False)
except ImportError:
    check("torch instalado", False, "no encontrado", duro=True)

# --- V-ENV-4: determinismo ----------------------------------------------------
cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
check("CUBLAS_WORKSPACE_CONFIG=:4096:8", cublas == ":4096:8",
      f"actual: '{cublas or '(no definida)'}' → exportar antes de correr el piloto "
      "para matmul determinista en GPU", duro=False)

try:
    import torch
    print(f"TF32 matmul : {torch.backends.cuda.matmul.allow_tf32} "
          "(congelar explícitamente en el piloto; afecta la 6ª-7ª cifra decimal)")
    print(f"TF32 cudnn  : {torch.backends.cudnn.allow_tf32}")
except Exception:
    pass

print("=" * 70)
if FALLOS:
    print(f"RESULTADO: FAIL ({len(FALLOS)} fallos duros: {FALLOS})")
    sys.exit(1)
print(f"RESULTADO: PASS ({len(WARNINGS)} warnings)")
sys.exit(0)
