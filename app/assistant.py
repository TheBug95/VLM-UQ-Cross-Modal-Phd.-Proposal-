"""Asistente IA del dashboard BIP 2026.

Responde preguntas en español sobre los datos del experimento y explica los
conceptos implementados. Dos modos:

1. **Modo IA completo**: usa un LLM serverless de Hugging Face
   (``huggingface_hub.InferenceClient``) con el resumen real de los resultados
   inyectado en el system prompt → responde con las cifras reales del estudio.
   Requiere la variable de entorno ``HF_TOKEN``.
2. **Modo fallback** (sin token): buscador local por palabras clave sobre el
   glosario de conceptos + resumen de datos. Siempre disponible.

El modelo se puede cambiar con la env var ``UQ_ASSISTANT_MODEL``.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

try:
    from . import dashboard_data as dd  # ejecución como paquete
    from .concepts import CONCEPTOS, RESUMEN_EXPERIMENTO
except ImportError:  # ejecución directa (python app/app.py)
    import dashboard_data as dd
    from concepts import CONCEPTOS, RESUMEN_EXPERIMENTO

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM_TEMPLATE = """\
Eres el asistente del dashboard del experimento BIP 2026: detección de errores
de un modelo de lenguaje con visión (MedGemma-4B) en detección de glaucoma,
usando desacuerdo entre representaciones visuales y textuales (divergencia KL)
como señal de incertidumbre, training-free y single-pass.

REGLAS:
- Responde SIEMPRE en español, claro y conciso (máximo ~150 palabras salvo que
  pidan detalle).
- Usa SOLO las cifras del RESUMEN DE DATOS de abajo. Si te preguntan un número
  que no está, dilo explícitamente en vez de inventarlo.
- Puedes explicar conceptos con el GLOSARIO.
- Recuerda que es una herramienta de investigación, no un dispositivo médico.

=== RESUMEN DE DATOS (cifras reales del experimento) ===
{datos}

=== GLOSARIO DE CONCEPTOS ===
{glosario}
"""


# ---------------------------------------------------------------------------
# Contexto de datos (se precomputa una vez al arrancar)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def build_data_context() -> str:
    """Resumen compacto en texto de todas las métricas relevantes."""
    partes: list[str] = []

    master = dd.load_master()
    n = len(master)
    n_norm = int((master["label"] == 0).sum())
    n_pat = int((master["label"] == 1).sum())
    splits = master["split"].value_counts().to_dict()
    n_art = int(master["has_annotation_artifact"].sum())
    partes.append(
        f"DATASET: {n} imágenes de fondo de ojo ({n_norm} normales, {n_pat} con "
        f"glaucoma). Splits: {splits}. {n_art} imágenes con artefacto de "
        f"anotación detectado en auditoría."
    )

    summary = dd.load_summary()
    partes.append("\nMÉTRICAS PRINCIPALES (detección de errores del modelo):")
    for _, r in summary.iterrows():
        partes.append(
            f"- {r['signal']} [{r['prompt']}, split={r['split']}]: "
            f"AUROC={r['auroc']:.3f} (IC95% {r['auroc_ci_low']:.3f}–{r['auroc_ci_high']:.3f}), "
            f"AUPRC={r['auprc']:.3f}, sens@80%spec={r['sens_80spec']:.3f}, "
            f"TPR@FPR10%={r['tpr_fpr10']:.3f}, ECE={r['ece']:.3f}, "
            f"n={r['n']}, errores={r['n_errors']}, "
            f"p(Mann-Whitney)={r['mannwhitney_p']:.4f}"
        )

    base_acc = None
    frame = dd.get_u("P1", dd.WINNER["family"], dd.WINNER["pooling"], dd.WINNER["tau"])
    base_acc = frame["correct"].mean()
    partes.append(f"\nACCURACY BASE de MedGemma (prompt P1): {base_acc:.3f} "
                  f"({int(frame['correct'].sum())}/{len(frame)} aciertos).")

    for tabla in ["tabla_t1_resultados.csv", "tabla_t2_ablaciones.csv",
                  "tabla_t4_costo_beneficio.csv", "tabla_t5_calibracion.csv"]:
        try:
            t = dd.load_tabla(tabla)
            partes.append(f"\n{tabla}:\n{t.to_csv(index=False)}")
        except FileNotFoundError:
            pass

    # Curva accuracy-coverage de la señal ganadora (triage)
    acc_cov = dd.accuracy_coverage(
        frame["u"].to_numpy(), frame["correct"].to_numpy(),
        steps=[0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
    )
    partes.append("\nTRIAGE con la señal ganadora (P1): derivar el X% más incierto "
                  "al oftalmólogo y aceptar el resto:")
    for _, r in acc_cov.iterrows():
        partes.append(
            f"- Derivar {r['n_referred']} casos ({(1 - r['coverage']) * 100:.0f}%): "
            f"accuracy de los retenidos={r['accuracy']:.3f}, "
            f"errores capturados en la derivación={r['errors_captured']}"
        )

    # Baselines multi-costo
    try:
        verb = pd.read_csv(dd.ASSETS / "results_verbalized.csv")
        conf = verb[verb["parse_ok"] == 1]["verbalized_conf"]
        partes.append(
            f"\nVERBALIZED CONFIDENCE (baseline 2×, n={len(verb)}): "
            f"confianza media declarada={conf.mean():.1f}/100."
        )
    except FileNotFoundError:
        pass
    try:
        sc = pd.read_csv(dd.ASSETS / "results_self_consistency.csv")
        partes.append(
            f"SELF-CONSISTENCY (baseline 10×, n={len(sc)} imágenes × 10 muestras a "
            f"T=1.5): entropía binaria media={sc['sc_entropy_binary'].mean():.3f}."
        )
    except FileNotFoundError:
        pass

    h4 = dd.h4_frame("P1")
    if len(h4) > 0:
        rho = dd.spearman(h4["u"].to_numpy(), h4["cdr_grade"].to_numpy())
        partes.append(f"\nH4 (u(x) vs. severidad CDR en {len(h4)} patológicos): "
                      f"Spearman ρ={rho:.3f}, p=0.99 → H4 RECHAZADA: la señal "
                      f"detecta errores del modelo, no severidad de la enfermedad.")

    return "\n".join(partes)


@lru_cache(maxsize=1)
def build_system_prompt() -> str:
    glosario = "\n\n".join(CONCEPTOS.values())
    return SYSTEM_TEMPLATE.format(datos=build_data_context(), glosario=glosario)


# ---------------------------------------------------------------------------
# Modo IA completo (LLM serverless de Hugging Face)
# ---------------------------------------------------------------------------

def llm_disponible() -> bool:
    return bool(os.environ.get("HF_TOKEN"))


def responder_llm(mensaje: str, historial: list[dict]) -> str:
    from huggingface_hub import InferenceClient

    client = InferenceClient(
        model=os.environ.get("UQ_ASSISTANT_MODEL", DEFAULT_MODEL),
        token=os.environ["HF_TOKEN"],
    )
    messages = [{"role": "system", "content": build_system_prompt()}]
    for turno in historial[-6:]:  # ventana corta de contexto
        messages.append({"role": turno["role"], "content": turno["content"]})
    messages.append({"role": "user", "content": mensaje})

    resp = client.chat_completion(messages=messages, max_tokens=512, temperature=0.3)
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Modo fallback (sin LLM): reglas por palabras clave
# ---------------------------------------------------------------------------

_REGLAS: list[tuple[tuple[str, ...], str]] = [
    (("auroc", "roc"), "auroc"),
    (("auprc", "precisión", "precision", "pr"), "auprc"),
    (("kl", "kullback", "divergencia"), "kl"),
    (("jsd", "jensen"), "jsd"),
    (("u(x)", "incertidumbre", "señal"), "u(x)"),
    (("triage", "derivar", "derivación", "cobertura", "coverage", "oftalm"), "triage"),
    (("calibr", "ece", "platt", "confiabilidad"), "calibracion"),
    (("pooling", "max", "mean", "roi", "atención", "atencion"), "pooling"),
    (("tau", "temperatura"), "tau"),
    (("baseline", "entropía", "entropia", "msp", "energía", "energia", "rank"), "baselines"),
    (("prompt", "p1", "p4"), "p1_p4"),
    (("h4", "severidad", "cdr", "cup", "copa"), "h4"),
    (("límite", "limite", "limitación", "limitacion", "n=129", "cuidado"), "limites"),
]


def responder_fallback(mensaje: str) -> str:
    texto = mensaje.lower()

    # Preguntas de cifras genéricas (prioridad sobre el glosario)
    if any(p in texto for p in ("resultado", "mejor", "ganador", "cuánto", "cuanto", "obtuvo")):
        summary = dd.load_summary()
        w = summary[(summary["signal"] == dd.WINNER_NAME) & (summary["split"] == "all")].iloc[0]
        return (
            f"La señal ganadora es **{dd.WINNER_NAME}** (KL texto→visión, capa 34, "
            f"τ=1, pooling max), seleccionada solo con el split train. "
            f"En las 129 imágenes: **AUROC = {w['auroc']:.3f}** "
            f"(IC95% {w['auroc_ci_low']:.3f}–{w['auroc_ci_high']:.3f}), "
            f"AUPRC = {w['auprc']:.3f}, sensibilidad@80%spec = {w['sens_80spec']:.3f}. "
            f"La combinada rank(KL)+rank(1−MSP) llega a 0.698. Con N=129 los "
            f"intervalos son anchos: evidencia sugestiva, no concluyente. "
            f"Pregúntame por \"triage\", \"KL\" o \"calibración\"."
        )

    # Pregunta numérica de triage
    if any(p in texto for p in ("deriv", "cobertura", "coverage")) and any(
        c.isdigit() for c in texto
    ):
        import re
        nums = re.findall(r"(\d+(?:\.\d+)?)", texto)
        if nums:
            pct = float(nums[0]) / (100 if float(nums[0]) > 1 else 1)
            coverage = 1 - pct
            frame = dd.get_u("P1", dd.WINNER["family"], dd.WINNER["pooling"], dd.WINNER["tau"])
            t = dd.triage_at_coverage(frame["u"].to_numpy(), frame["correct"].to_numpy(), coverage)
            return (
                f"Con la señal ganadora (P1), derivando el {pct:.0%} más incierto "
                f"({t['n_referred']} de {t['n_total']} casos):\n"
                f"- Accuracy de los aceptados: **{t['accuracy_kept']:.3f}** "
                f"(vs. {t['accuracy_overall']:.3f} sin triage)\n"
                f"- Errores capturados: **{t['errors_captured']}** de "
                f"{t['errors_total']} (una derivación aleatoria capturaría "
                f"~{t['errors_expected_random']:.1f})\n\n"
                f"Pruébalo en vivo en la pestaña «Simulador de triage»."
            )

    for claves, concepto in _REGLAS:
        if any(c in texto for c in claves):
            return CONCEPTOS[concepto]

    return (
        "Puedo explicarte los conceptos del experimento y sus cifras. Prueba con:\n"
        "- ¿Qué es u(x) y cómo se calcula?\n"
        "- ¿Qué AUROC obtuvo la señal ganadora?\n"
        "- ¿Qué pasa si derivo el 20% de los casos al oftalmólogo?\n"
        "- ¿Qué es la calibración / ECE?\n"
        "- ¿Cuáles son los límites del estudio?\n\n"
        "*(Modo local sin LLM: configura el secret `HF_TOKEN` para respuestas "
        "completas con IA.)*"
    )


# ---------------------------------------------------------------------------
# Punto de entrada para Gradio
# ---------------------------------------------------------------------------

def responder(mensaje: str, historial: list[dict]) -> str:
    """Responde un mensaje del chat. `historial` = lista de dicts role/content."""
    if llm_disponible():
        try:
            return responder_llm(mensaje, historial)
        except Exception as exc:  # API caída, cuota, modelo no disponible...
            return (
                f"⚠️ El LLM no respondió ({type(exc).__name__}). "
                "Respuesta en modo local:\n\n" + responder_fallback(mensaje)
            )
    return responder_fallback(mensaje)


EJEMPLOS_PREGUNTAS = [
    "¿Qué es u(x) y cómo se calcula?",
    "¿Qué AUROC obtuvo la señal ganadora y qué significa?",
    "¿Qué pasa si derivo el 20% de los casos al oftalmólogo?",
    "¿Qué es la divergencia KL y por qué la usan?",
    "¿En qué se diferencia la señal de los baselines entropy y 1-MSP?",
    "¿Cuáles son los límites del estudio con N=129?",
]
