# 10 — Guía de Reproducibilidad

> **Propósito:** permitir que otra persona (o el autor en 6 meses) reproduzca el experimento desde cero: requisitos, instalación, acceso a HuggingFace, comandos en orden y verificación de resultados.

[⬅️ 09 — Dataset MM-ODIR-129](09_Dataset_MM_ODIR_129.md) | [➡️ 11 — Conclusiones y Próximos Pasos](11_Conclusiones_y_Proximos_Pasos.md)

---

## 10.1 Requisitos de hardware y software

| Requisito | Detalle |
|---|---|
| GPU | ~16 GB VRAM en bfloat16 (~6–8 GB en 4-bit con `bitsandbytes`, opcional — p. ej. Colab T4 con `load_in_4bit: true`) |
| Tiempo de cómputo | Corrida principal: 258 inferencias ≈ **10–20 min de GPU** (~4.5 s/imagen). P5 (verbalized): ~5 min. SC: la parte más larga (1.000 generaciones muestreadas) |
| Python | 3.10+ |
| Paquetes clave | `torch` (build CUDA según GPU), `transformers >= 4.51.3` (**obligatorio**), `scipy >= 1.11`, `scikit-learn >= 1.7`, `pandas`, `matplotlib`, `seaborn`, `huggingface_hub`, `pyyaml`, `pillow` |
| Cuenta HuggingFace | Con **licencia HAI-DEF aceptada** para `google/medgemma-4b-it` y token de acceso |

## 10.2 Instalación paso a paso (desde cero)

```bash
# 1. Clonar/copiar el proyecto y entrar al directorio
cd VLM-UQ-Cross-Modal-Phd.-Proposal-

# 2. Crear y activar el entorno virtual
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
# .venv\Scripts\Activate.ps1       # Windows PowerShell
# source .venv/bin/activate        # Linux/Mac

# 3. (GPU NVIDIA) Instalar torch con el índice CUDA correspondiente
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Instalar dependencias
pip install -r requirements.txt
#    (o instalación editable: pip install -e .)
```

## 10.3 Configuración de HuggingFace

1. Aceptar la licencia **HAI-DEF** en la página de `google/medgemma-4b-it`.
2. Exportar el token (nunca commitearlo):

```bash
export HF_TOKEN=hf_...        # Linux/Mac/Git Bash
# $env:HF_TOKEN="hf_..."      # PowerShell
```

3. El dataset `TheBug95/MM-ODIR-129` es público (licencia MIT): no requiere token, pero el mismo `HF_TOKEN` sirve para evitar límites de rate en la descarga.

## 10.4 Comandos del pipeline en orden

```bash
# 0. Validaciones pre-implementación (§10.5) — recomendado en primera corrida
python validacion/val_01_environment.py
python validacion/val_02_tokenizer.py
python validacion/val_03_dataset.py
python validacion/val_05_metrics.py
python validacion/val_06_stats.py
python validacion/val_04_generate_api.py   # requiere GPU + licencia
python validacion/val_07_pilot.py          # requiere GPU + licencia

# 1. Descarga de datos + tabla maestra + auditoría de artefactos
python -m src.data

# 2. Piloto de 20 imágenes con los 8 sanity checks (OBLIGATORIO antes de la corrida)
python -m src.inference --pilot --n 20

# 3. Corrida completa (129 × 2 prompts = 258 inferencias, ~10–20 min GPU)
python -m src.inference --run-full
#    con atenciones (heatmaps; más lento, eager attention):
python -m src.inference --run-full --attentions

# 4. Baselines multi-pass
python -m src.inference --verbalized                     # P5, 129 inferencias (~5 min)
python -m src.inference --self-consistency               # SC: 50 imgs × 10 muestras × 2 prompts
#    temperatura del muestreo SC (default 1.5):
python -m src.inference --self-consistency --sc-temp 1.5

# 5. (Opcional) Repeticiones multi-semilla
python -m src.inference --run-full --seeds 42 123 456
python -m src.inference --self-consistency --seeds 42 123

# 6. Estadística y figuras
python -m src.evaluation          # guarda results/evaluation_summary.csv
python -m src.evaluation --all-signals   # tabla rápida de AUROC por variante
python -m src.figures             # Figuras 2–9 + Tablas T1–T4

# 7. Verificación independiente (recomendado siempre)
python validacion/val_08_resultados.py   # 19 checks; debe imprimir 19/19 PASS
```

## 10.5 Validaciones pre-implementación (7 scripts)

| Script | Qué valida |
|---|---|
| `val_01_environment.py` | Versiones de Python/torch/transformers, GPU, VRAM |
| `val_02_tokenizer.py` | Token IDs (`yes`=4443, `no`=1904, `image_soft_token`=262.144, vecinos 255.999/256.000); variantes con espacio; máscara de 256 tokens de imagen |
| `val_03_dataset.py` | Descarga, conteos (60N/69P, splits 77/26/26), estructura de `annotations.json`/`split.json` |
| `val_04_generate_api.py` | API de `generate`: `hidden_states[0]` con 35 entradas, `scores[0]`, máscara por `image_token_index` (requiere GPU) |
| `val_05_metrics.py` | Implementaciones de AUROC/AUPRC contra sklearn |
| `val_06_stats.py` | Bootstrap BCa, Mann-Whitney, permutación de Spearman |
| `val_07_pilot.py` | Piloto end-to-end con reglas numéricas duras (float64, softmax cruda) — origen de las decisiones de [§4.3](04_Arquitectura_Tecnica.md) |

## 10.6 Verificación de resultados: conteos y checksums esperados

Tras la corrida completa, verificar:

| Artefacto | Valor esperado |
|---|---|
| `data/master_table.csv` | 129 filas × 15 columnas; 60 Normal / 69 Pathological; train 77 / val 26 / test 26; flag `has_annotation_artifact` presente |
| `results/results_full.csv` | ~23.600 filas (formato largo); 129 `image_filename` × 2 `prompt_id` únicos |
| Accuracy base P1 | 79.8% (103/129 correctas; 26 errores) |
| Señal ganadora | `kl_t_v_L34_tau1.0_max` (elegida solo en train) |
| AUROC ganadora (129) | 0.661 [0.522, 0.772] — exacto a mano: 0.660941 |
| AUROC combinación | 0.698 — exacto a mano: 0.697535 |
| `results_verbalized.csv` | 129 filas; solo 2 valores de confianza: 95 (n=118), 90 (n=11) |
| `results_self_consistency.csv` | 100 filas (50 imgs × 2 prompts) |
| `validacion/val_08_resultados.py` | **19/19 checks PASS** |
| Regla de portabilidad | Mantener `epsilon = 1.0e-10` fijo (`config.yaml`); NO comparar nats absolutos entre corridas con eps distinto; derivación clínica por percentil de cohorte |

**Nota sobre reproducibilidad cross-hardware:** los logits son bitwise reproducibles entre GPUs (0/129 predicciones cambian); la KL puede variar ~0.4 nats por backend de atención (eager vs. sdpa) sin afectar el ranking (Spearman 0.964, ΔAUROC 0.016) — ver [§12.4](12_Verificacion_y_Validacion.md).

---

[⬅️ 09 — Dataset MM-ODIR-129](09_Dataset_MM_ODIR_129.md) | [➡️ 11 — Conclusiones y Próximos Pasos](11_Conclusiones_y_Proximos_Pasos.md)
