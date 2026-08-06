"""Dashboard interactivo del experimento BIP 2026.

Detección de errores de MedGemma-4B en glaucoma mediante desacuerdo
cross-modal (divergencia KL) como señal de incertidumbre.

Ejecutar local:
    python app/app.py          # abre http://127.0.0.1:7860

Deploy: HF Space (sdk gradio) — ver app/README.md.
⚠️ Herramienta de investigación, no un dispositivo médico.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from . import assistant, dashboard_data as dd
    from .concepts import CONCEPTOS, MAPAS_POOLING, PANELES, RESUMEN_EXPERIMENTO
except ImportError:
    import assistant
    import dashboard_data as dd
    from concepts import CONCEPTOS, MAPAS_POOLING, PANELES, RESUMEN_EXPERIMENTO

THUMBS = Path(__file__).resolve().parent / "assets" / "thumbnails"
THUMBS_SQ = Path(__file__).resolve().parent / "assets" / "thumbnails_square"

VERDE = "#2ca02c"
ROJO = "#d62728"
AZUL = "#1f77b4"
GRIS = "#7f7f7f"

DISCLAIMER = (
    "⚠️ **Herramienta de investigación.** No es un dispositivo médico ni está "
    "validada para uso clínico."
)

TEMPLATE = "plotly_white"


# ---------------------------------------------------------------------------
# Figuras Plotly
# ---------------------------------------------------------------------------

def fig_roc_pr(frame: pd.DataFrame) -> go.Figure:
    y_err = 1 - frame["correct"].astype(int).to_numpy()
    u = frame["u"].to_numpy()
    fpr, tpr, thr = dd.roc_curve(u, y_err)
    recall, precision, ap = dd.pr_curve(u, y_err)
    auc = dd.auroc(u, y_err)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color=GRIS), name="Azar"))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUROC={auc:.3f})",
                             line=dict(color=AZUL, width=3),
                             customdata=np.round(thr, 3),
                             hovertemplate="FPR=%{x:.3f} TPR=%{y:.3f}<br>umbral u(x)=%{customdata}"))
    fig.update_xaxes(title_text="Tasa de falsos positivos (FPR)", range=[0, 1])
    fig.update_yaxes(title_text="Tasa de verdaderos positivos (TPR)", range=[0, 1.02])
    fig.update_layout(template=TEMPLATE, height=420,
                      title=f"Curva ROC — detección de errores (AUROC = {auc:.3f})")
    return fig


def fig_pr(frame: pd.DataFrame) -> go.Figure:
    y_err = 1 - frame["correct"].astype(int).to_numpy()
    u = frame["u"].to_numpy()
    recall, precision, ap = dd.pr_curve(u, y_err)
    prev = y_err.mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[prev, prev], mode="lines",
                             line=dict(dash="dash", color=GRIS),
                             name=f"Azar (prevalencia={prev:.2f})"))
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                             name=f"PR (AUPRC={ap:.3f})",
                             line=dict(color=ROJO, width=3),
                             hovertemplate="Recall=%{x:.3f} Precisión=%{y:.3f}"))
    fig.update_xaxes(title_text="Recall (errores capturados)", range=[0, 1])
    fig.update_yaxes(title_text="Precisión", range=[0, 1.02])
    fig.update_layout(template=TEMPLATE, height=420,
                      title=f"Curva Precisión-Recall (AUPRC = {ap:.3f})")
    return fig


def fig_boxplot(frame: pd.DataFrame) -> go.Figure:
    ok = frame[frame["correct"] == 1]
    bad = frame[frame["correct"] == 0]
    fig = go.Figure()
    for nombre, sub, color in [("Aciertos", ok, VERDE), ("Errores", bad, ROJO)]:
        fig.add_trace(go.Box(
            y=sub["u"], name=f"{nombre} (n={len(sub)})",
            marker_color=color, boxpoints="all", jitter=0.4, pointpos=0,
            customdata=sub["image_filename"],
            hovertemplate="%{customdata}<br>u(x)=%{y:.3f}",
        ))
    fig.update_yaxes(title_text="u(x) — incertidumbre")
    fig.update_layout(template=TEMPLATE, height=420,
                      title="u(x): ¿los errores tienen mayor incertidumbre?")
    return fig


def fig_histograma(frame: pd.DataFrame) -> go.Figure:
    norm = frame[frame["label"] == 0]
    pat = frame[frame["label"] == 1]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=norm["u"], name=f"Normal (n={len(norm)})",
                               marker_color=AZUL, opacity=0.65, nbinsx=25))
    fig.add_trace(go.Histogram(x=pat["u"], name=f"Glaucoma (n={len(pat)})",
                               marker_color=ROJO, opacity=0.65, nbinsx=25))
    fig.update_layout(barmode="overlay", template=TEMPLATE, height=420,
                      title="Distribución de u(x) por etiqueta clínica real",
                      xaxis_title="u(x)", yaxis_title="n imágenes")
    return fig


def fig_triage(frame: pd.DataFrame, coverage: float) -> go.Figure:
    curva = dd.accuracy_coverage(frame["u"].to_numpy(), frame["correct"].to_numpy())
    t = dd.triage_at_coverage(frame["u"].to_numpy(), frame["correct"].to_numpy(), coverage)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curva["coverage"] * 100, y=curva["accuracy"], mode="lines",
        name="Accuracy de retenidos", line=dict(color=AZUL, width=3),
        customdata=curva[["n_referred", "errors_captured"]],
        hovertemplate=("Cobertura=%{x:.0f}%<br>Accuracy=%{y:.3f}"
                       "<br>Derivados=%{customdata[0]}<br>Errores capturados=%{customdata[1]}"),
    ))
    fig.add_hline(y=t["accuracy_overall"], line_dash="dash", line_color=GRIS,
                  annotation_text=f"Sin triage: {t['accuracy_overall']:.3f}")
    fig.add_trace(go.Scatter(
        x=[coverage * 100], y=[t["accuracy_kept"]], mode="markers",
        marker=dict(size=16, color=ROJO, symbol="diamond"),
        name="Tu selección",
        hovertemplate=f"Cobertura={coverage * 100:.0f}%<br>Accuracy={t['accuracy_kept']:.3f}",
    ))
    fig.update_xaxes(title_text="Cobertura (% casos que acepta el sistema)", range=[45, 102])
    fig.update_yaxes(title_text="Accuracy", range=[0.5, 1.02])
    fig.update_layout(template=TEMPLATE, height=430,
                      title="Accuracy–cobertura: derivar los casos más inciertos mejora el sistema")
    return fig


def fig_cuadrantes(frame: pd.DataFrame, coverage: float) -> go.Figure:
    umbral = frame["u"].quantile(1 - coverage)
    fig = go.Figure()
    for nombre, sub, color, simbolo in [
        ("Acierto", frame[frame["correct"] == 1], VERDE, "circle"),
        ("Error", frame[frame["correct"] == 0], ROJO, "x"),
    ]:
        fig.add_trace(go.Scatter(
            x=sub["p_yes"], y=sub["u"], mode="markers", name=f"{nombre} (n={len(sub)})",
            marker=dict(color=color, symbol=simbolo, size=9, opacity=0.75),
            customdata=sub["image_filename"],
            hovertemplate="%{customdata}<br>p_yes=%{x:.3f}<br>u(x)=%{y:.3f}",
        ))
    fig.add_hline(y=umbral, line_dash="dot", line_color=ROJO,
                  annotation_text=f"Umbral de derivación (cobertura {coverage * 100:.0f}%)")
    fig.update_xaxes(title_text="Confianza del modelo: p_yes")
    fig.update_yaxes(title_text="u(x) — incertidumbre cross-modal")
    fig.update_layout(template=TEMPLATE, height=430,
                      title="Cuadrantes: confianza del modelo vs. nuestra señal")
    return fig


def fig_reliability(frame: pd.DataFrame, n_bins: int = 10) -> go.Figure:
    """Diagrama de confiabilidad estilo FUSE §5.2: Platt scaling de u(x) → P(error),
    ajustado SOLO en train; bins equiprobables sobre toda la cohorte."""
    train = frame[frame["split"] == "train"]
    a, b = dd.platt_fit(train["u"].to_numpy(), (1 - train["correct"]).to_numpy())
    obs = frame.copy()
    obs["p_err"] = dd.platt_predict(obs["u"].to_numpy(), a, b)
    obs["err"] = 1 - obs["correct"]
    obs["bin"] = pd.qcut(obs["p_err"], q=min(n_bins, len(obs)), duplicates="drop")
    g = obs.groupby("bin", observed=True).agg(
        p_media=("p_err", "mean"), err_emp=("err", "mean"), n=("err", "size"),
    ).reset_index()
    ece = float((g["n"] / g["n"].sum() * (g["p_media"] - g["err_emp"]).abs()).sum())
    ancho = float(g["p_media"].diff().median() * 0.8) if len(g) > 1 else 0.02
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color=GRIS), name="Calibración perfecta"))
    fig.add_trace(go.Bar(x=g["p_media"], y=g["err_emp"], width=[ancho] * len(g),
                         name="Tasa de error empírica",
                         marker_color=AZUL, opacity=0.8,
                         customdata=g["n"],
                         hovertemplate=("P(error) predicha=%{x:.3f}<br>Error empírico=%{y:.3f}"
                                        "<br>n=%{customdata}")))
    fig.update_xaxes(title_text="P(error) predicha (Platt, ajustado en train)", range=[0, 1])
    fig.update_yaxes(title_text="Tasa de error empírica", range=[0, 1.02])
    fig.update_layout(template=TEMPLATE, height=420,
                      title=(f"Diagrama de confiabilidad de u(x) tras Platt scaling "
                             f"(ECE = {ece:.3f}, bins equiprobables)"))
    return fig


def fig_h4(prompt: str) -> go.Figure:
    h4 = dd.h4_frame(prompt)
    rho = dd.spearman(h4["u"].to_numpy(), h4["cdr_grade"].to_numpy())
    jitter = np.random.default_rng(42).normal(0, 0.05, len(h4))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=h4["cdr_grade"] + jitter, y=h4["u"], mode="markers",
        marker=dict(color=AZUL, size=10, opacity=0.7),
        customdata=h4["image_filename"],
        hovertemplate="%{customdata}<br>CDR grado=%{x:.1f}<br>u(x)=%{y:.3f}",
    ))
    # mediana por grado
    med = h4.groupby("cdr_grade")["u"].median()
    fig.add_trace(go.Scatter(x=med.index, y=med.values, mode="lines+markers",
                             name="Mediana por grado", line=dict(color=ROJO, width=3)))
    fig.update_xaxes(title_text="Grado cup-to-disc ratio (0–4, ordinal)")
    fig.update_yaxes(title_text="u(x) — señal ganadora")
    fig.update_layout(template=TEMPLATE, height=420,
                      title=f"H4 (rechazada): incertidumbre vs. severidad (Spearman ρ = {rho:.3f}, n={len(h4)})")
    return fig


# ---------------------------------------------------------------------------
# Callbacks de los tabs
# ---------------------------------------------------------------------------

INSTRUCCIONES_MAPAS = """\
### 🗺️ Mapas de pooling — aún no hay datos

Los CSVs del experimento guardan solo los valores escalares de cada variante;
los **pesos por token** de cada pooling se calculan en memoria durante la
inferencia y se descartan. Para visualizarlos hace falta una pasada extra de GPU
(~15–25 min en Colab T4, sin reentrenar nada).

En tu notebook de Colab con GPU (donde ya corre el proyecto), ejecuta:

```bash
# prueba rápida (~1 min)
!python -m src.extract_pooling_maps --prompt P1 --n 3

# corrida completa: las 129 imágenes × 8 poolings
!python -m src.extract_pooling_maps --prompt P1
```

> Requisito: que el dataset esté descargado completo (con máscaras `*_disc.png`,
> necesarias para el pooling `roi`) — `python -m src.data` lo garantiza.

Después, en local:

1. Copia el `results/pooling_maps.csv` generado a `results/pooling_maps.csv`.
2. Ejecuta `python app/prepare_assets.py` y reinicia el dashboard.

Se extraen los mapas 16×16 de las 8 técnicas (mean, max, topk, normw, attn,
rollout, headspec y roi) para las 129 imágenes, capa 34.
"""


def fig_mapas_pooling(image_filename: str, prompt_id: str) -> go.Figure:
    """Grid 2×4 con los heatmaps 16×16 de las 8 técnicas de pooling."""
    grids = dd.pooling_maps_grid(image_filename, prompt_id)
    orden = dd.POOLING_MAPS_ORDEN
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[f"{p}{' ⭐' if p == 'max' else ''}" for p in orden],
        horizontal_spacing=0.06, vertical_spacing=0.12,
    )
    for i, pooling in enumerate(orden):
        r, c = divmod(i, 4)
        if pooling in grids:
            z = grids[pooling]
            zmin, zmax = float(z.min()), float(z.max())
            z_norm = (z - zmin) / (zmax - zmin) if zmax > zmin else z
            fig.add_trace(
                go.Heatmap(
                    z=z_norm, colorscale="Viridis", showscale=False,
                    customdata=np.round(grids[pooling], 5),
                    hovertemplate=(f"{pooling}<br>fila=%{{y}} col=%{{x}}"
                                   "<br>peso=%{customdata}"),
                ),
                row=r + 1, col=c + 1,
            )
        else:
            # p.ej. roi en imágenes normales: sin máscara disponible
            fig.add_annotation(
                text="sin datos<br>(oracle: solo<br>69 patológicas)" if pooling == "roi" else "sin datos",
                xref=f"x{i + 1}", yref=f"y{i + 1}", x=7.5, y=7.5,
                showarrow=False, font=dict(size=11, color=GRIS),
            )
            fig.add_trace(
                go.Heatmap(z=np.full((16, 16), np.nan), colorscale="Greys",
                           showscale=False, hoverinfo="skip"),
                row=r + 1, col=c + 1,
            )
        fig.update_xaxes(showticklabels=False, row=r + 1, col=c + 1)
        fig.update_yaxes(showticklabels=False, autorange="reversed", row=r + 1, col=c + 1)
    fig.update_layout(
        template=TEMPLATE, height=560,
        title=f"Mapas de peso por token visual (16×16) — {image_filename} · {prompt_id}",
        margin=dict(t=80, b=20, l=20, r=20),
    )
    return fig


def actualizar_mapas(image_filename, prompt_id):
    thumb = THUMBS_SQ / image_filename
    return fig_mapas_pooling(image_filename, prompt_id), str(thumb)


def actualizar_explorador(prompt, familia, pooling, tau, split):
    frame = dd.get_u(prompt, familia, pooling, tau, split)
    m = dd.metricas_basicas(frame)
    nombre = dd.NOMBRES_FAMILIA.get(familia, familia)
    if familia in dd.FAMILIAS_KL:
        nombre += f" — pooling {pooling}, τ={tau}"
    metricas_md = (
        f"### {nombre}\n"
        f"| Métrica | Valor |\n|---|---|\n"
        f"| n imágenes | {m['n']} |\n"
        f"| Errores del modelo | {m['n_errors']} ({m['n_errors'] / max(m['n'], 1):.0%}) |\n"
        f"| Accuracy del modelo | {m['accuracy_modelo']:.3f} |\n"
        f"| **AUROC** (detección de errores) | **{m['auroc']:.3f}** |\n"
        f"| AUPRC | {m['auprc']:.3f} |\n"
        f"| Sensibilidad @ 80% especificidad | {m['sens_80spec']:.3f} |\n\n"
        f"*Valores puntuales recalculados en vivo. Los intervalos de confianza "
        f"bootstrap de las señales principales están en la pestaña «Resultados».*"
    )
    if pooling == "roi (oracle)":
        metricas_md += (
            "\n\n> ⚠️ **ROI oracle: solo hay {n} imágenes** (las patológicas con "
            "máscara de disco) — el AUROC **no es comparable** con el resto de "
            "poolings. Y ojo: 0.349 < 0.5 no es un error del dashboard. El 0.889 "
            "del piloto era un artefacto de muestra pequeña; en la corrida "
            "completa el oracle *empeora* la señal (hallazgo documentado en "
            "`Documentacion/07_Ablaciones_y_Analisis_Profundo.md`)."
        ).format(n=m["n"])
    explicacion = _explicacion_senal(familia)
    return (fig_roc_pr(frame), fig_pr(frame), fig_boxplot(frame),
            fig_histograma(frame), metricas_md, explicacion)


def _explicacion_senal(familia: str) -> str:
    mapa = {
        "kl_t_v": "kl", "kl_v_t": "kl", "jsd": "jsd", "cosine": "kl",
        "kl_prompt": "kl", "entropy": "baselines", "1-msp": "baselines",
        "energy": "baselines", "rankcombo": "baselines",
    }
    return CONCEPTOS[mapa.get(familia, "u(x)")]


def actualizar_triage(prompt, coverage):
    frame = dd.get_u(prompt, dd.WINNER["family"], dd.WINNER["pooling"], dd.WINNER["tau"])
    t = dd.triage_at_coverage(frame["u"].to_numpy(), frame["correct"].to_numpy(), coverage)
    resumen = (
        f"### Con una cobertura del {coverage:.0%} (derivar el {1 - coverage:.0%} más incierto)\n"
        f"| Indicador | Valor |\n|---|---|\n"
        f"| Casos derivados al oftalmólogo | **{t['n_referred']}** de {t['n_total']} |\n"
        f"| Accuracy de los casos aceptados | **{t['accuracy_kept']:.3f}** "
        f"(vs. {t['accuracy_overall']:.3f} sin triage) |\n"
        f"| Errores del modelo capturados en la derivación | **{t['errors_captured']}** "
        f"de {t['errors_total']} |\n"
        f"| Errores que capturaría una derivación **aleatoria** del mismo tamaño | "
        f"{t['errors_expected_random']:.1f} |"
    )
    return fig_triage(frame, coverage), fig_cuadrantes(frame, coverage), resumen


def _gallery_items(split, clase, acierto):
    frame = dd.get_u("P1", dd.WINNER["family"], dd.WINNER["pooling"], dd.WINNER["tau"])
    master = dd.load_master()[["image_filename", "has_annotation_artifact"]]
    frame = frame.merge(master, on="image_filename", how="left")
    if split != "all":
        frame = frame[frame["split"] == split]
    if clase != "all":
        frame = frame[frame["label"] == (1 if clase == "Glaucoma" else 0)]
    if acierto != "all":
        frame = frame[frame["correct"] == (1 if acierto == "Aciertos" else 0)]
    items, meta = [], []
    for _, r in frame.iterrows():
        thumb = THUMBS / r["image_filename"]
        if not thumb.exists():
            continue
        marca = "✅" if r["correct"] == 1 else "❌"
        artefacto = " ⚠️" if bool(r["has_annotation_artifact"]) else ""
        items.append((str(thumb), f"{marca}{artefacto} {r['image_filename']}"))
        meta.append(r.to_dict())
    return items, meta


def detalle_imagen(meta, evt: gr.SelectData):
    if not meta:
        return "Sin datos."
    r = meta[evt.index]
    frame_all = dd.get_u("P1", dd.WINNER["family"], dd.WINNER["pooling"], dd.WINNER["tau"])
    pct = dd.percentile_of(frame_all["u"].to_numpy(), r["u"])
    pred_txt = "Glaucoma (yes)" if r["p_yes"] >= 0.5 else "Normal (no)"
    real_txt = "Glaucoma" if r["label"] == 1 else "Normal"
    nivel = "🔴 ALTA — derivar" if pct >= 80 else ("🟡 media" if pct >= 50 else "🟢 baja")
    return (
        f"### {r['image_filename']}\n"
        f"| Campo | Valor |\n|---|---|\n"
        f"| Etiqueta real | {real_txt} |\n"
        f"| Predicción MedGemma (P1) | {pred_txt} (p_yes = {r['p_yes']:.4f}) |\n"
        f"| Resultado | {'✅ Acierto' if r['correct'] == 1 else '❌ Error'} |\n"
        f"| u(x) — KL ganadora | {r['u']:.3f} nats |\n"
        f"| Percentil en la cohorte | **{pct:.0f}** → prioridad de revisión: {nivel} |\n"
        f"| Split | {r['split']} |\n\n"
        f"*El percentil se calcula contra las 129 imágenes: regla de derivación por "
        f"percentil de cohorte, nunca por umbral absoluto de nats.*"
    )


def responder_chat(mensaje, historial):
    return assistant.responder(mensaje, historial)


# ---------------------------------------------------------------------------
# Construcción de la UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    ganador = dd.get_u("P1", dd.WINNER["family"], dd.WINNER["pooling"], dd.WINNER["tau"])
    m0 = dd.metricas_basicas(ganador)
    master = dd.load_master()

    with gr.Blocks(title="Incertidumbre Cross-Modal en Glaucoma") as demo:
        gr.Markdown(
            "# 🔬 Desacuerdo Cross-Modal como Señal de Incertidumbre\n"
            "Dashboard interactivo del experimento: detección de errores de "
            "**MedGemma-4B** en detección de glaucoma (129 fondos de ojo), "
            "con una señal **training-free y single-pass** basada en divergencia KL "
            "entre representaciones visual y textual.\n\n" + DISCLAIMER
        )

        with gr.Tab("🏠 Panorama"):
            gr.Markdown(RESUMEN_EXPERIMENTO)
            with gr.Row():
                gr.Markdown(
                    f"### 🎯 AUROC ganador\n## {m0['auroc']:.3f}\n"
                    f"detección de errores (P1, n={m0['n']})"
                )
                gr.Markdown(
                    f"### 🧠 Accuracy MedGemma\n## {m0['accuracy_modelo']:.1%}\n"
                    f"{m0['n'] - m0['n_errors']}/{m0['n']} aciertos — hay "
                    f"{m0['n_errors']} errores que detectar"
                )
                gr.Markdown(
                    f"### ⚡ Costo\n## 1×\nuna sola pasada por imagen, "
                    f"modelo congelado, sin reentrenar"
                )
                gr.Markdown(
                    f"### 🖼️ Dataset\n## {len(master)}\n"
                    f"{int((master['label'] == 0).sum())} normales / "
                    f"{int((master['label'] == 1).sum())} glaucoma"
                )
            gr.Markdown("---\n" + PANELES["galeria"])
            with gr.Row():
                f_split = gr.Dropdown(["all", "train", "validation", "test"],
                                      value="all", label="Split", scale=1)
                f_clase = gr.Dropdown(["all", "Normal", "Glaucoma"], value="all",
                                      label="Etiqueta real", scale=1)
                f_acierto = gr.Dropdown(["all", "Aciertos", "Errores"], value="all",
                                        label="Resultado del modelo", scale=1)
            galeria = gr.Gallery(label="129 fondos de ojo (click para ver la ficha)",
                                 columns=6, rows=2, height=420, object_fit="cover",
                                 allow_preview=True)
            estado_meta = gr.State([])
            ficha = gr.Markdown("Haz click en una imagen para ver su ficha clínica.")

            for ctrl in (f_split, f_clase, f_acierto):
                ctrl.change(_gallery_items, [f_split, f_clase, f_acierto],
                            [galeria, estado_meta])
            galeria.select(detalle_imagen, estado_meta, ficha)
            demo.load(_gallery_items, [f_split, f_clase, f_acierto],
                      [galeria, estado_meta])

        with gr.Tab("📊 Explorador de señales"):
            gr.Markdown(
                "Cada imagen tiene un **u(x)**: cuanto mayor, menos confiable la "
                "respuesta del modelo. Aquí puedes comparar todas las variantes de "
                "la señal y los baselines de igual costo. Todo se recalcula al vuelo."
            )
            with gr.Row():
                e_prompt = gr.Radio(["P1", "P4"], value="P1", label="Prompt")
                e_familia = gr.Dropdown(
                    list(dd.NOMBRES_FAMILIA.keys()), value="kl_t_v",
                    label="Familia de señal",
                )
                e_pooling = gr.Dropdown(dd.POOLINGS, value="max",
                                        label="Pooling (solo familias KL/JSD/coseno)")
                e_tau = gr.Dropdown(dd.TAUS, value="1.0", label="τ (temperatura)")
                e_split = gr.Radio(["all", "train", "validation", "test"],
                                   value="all", label="Split")
            with gr.Row():
                e_metricas = gr.Markdown()
                with gr.Accordion("📖 ¿Qué es esta señal?", open=False):
                    e_concepto = gr.Markdown()
            with gr.Row():
                e_roc = gr.Plot()
                e_pr = gr.Plot()
            with gr.Accordion("📖 ¿Qué estoy viendo? (ROC / PR)", open=False):
                gr.Markdown(PANELES["roc_pr"])
            with gr.Row():
                e_box = gr.Plot()
                e_hist = gr.Plot()
            with gr.Row():
                with gr.Accordion("📖 ¿Qué estoy viendo? (boxplot)", open=False):
                    gr.Markdown(PANELES["boxplot"])
                with gr.Accordion("📖 ¿Qué estoy viendo? (histograma)", open=False):
                    gr.Markdown(PANELES["histograma"])

            entradas = [e_prompt, e_familia, e_pooling, e_tau, e_split]
            salidas = [e_roc, e_pr, e_box, e_hist, e_metricas, e_concepto]
            for ctrl in entradas:
                ctrl.change(actualizar_explorador, entradas, salidas)
            demo.load(actualizar_explorador, entradas, salidas)

        with gr.Tab("🗺️ Mapas de pooling"):
            maps_df = dd.load_pooling_maps()
            if maps_df is None:
                gr.Markdown(INSTRUCCIONES_MAPAS)
            else:
                gr.Markdown(PANELES["mapas"])
                imagenes_maps = sorted(maps_df["image_filename"].unique())
                prompts_maps = sorted(maps_df["prompt_id"].unique())
                with gr.Row():
                    mp_imagen = gr.Dropdown(
                        imagenes_maps, value=imagenes_maps[0],
                        label="Imagen", filterable=True,
                    )
                    mp_prompt = gr.Radio(prompts_maps, value=prompts_maps[0], label="Prompt")
                with gr.Row():
                    mp_fig = gr.Plot()
                    mp_thumb = gr.Image(label="El fundus como lo ve el modelo (estirado a 896×896)",
                                        height=560, width=560)
                with gr.Accordion("📖 ¿Qué hace cada técnica?", open=False):
                    gr.Markdown("\n\n".join(
                        f"- **{p}**: {MAPAS_POOLING[p]}" for p in dd.POOLING_MAPS_ORDEN
                    ))
                for ctrl in (mp_imagen, mp_prompt):
                    ctrl.change(actualizar_mapas, [mp_imagen, mp_prompt],
                                [mp_fig, mp_thumb])
                demo.load(actualizar_mapas, [mp_imagen, mp_prompt],
                          [mp_fig, mp_thumb])

        with gr.Tab("🚑 Simulador de triage"):
            gr.Markdown(
                "La aplicación clínica de u(x): **derivar al oftalmólogo los casos "
                "más inciertos** y aceptar automáticamente los más seguros. "
                "Mueve el slider y observa cómo cambian la accuracy del sistema y "
                "los errores capturados. Señal usada: la ganadora congelada "
                f"(`{dd.WINNER_NAME}`)."
            )
            t_prompt = gr.Radio(["P1", "P4"], value="P1", label="Prompt")
            t_slider = gr.Slider(0.5, 1.0, value=0.8, step=0.01,
                                 label="Cobertura (% de casos que acepta el sistema automáticamente)")
            t_resumen = gr.Markdown()
            with gr.Row():
                t_fig = gr.Plot()
                t_cuad = gr.Plot()
            with gr.Row():
                with gr.Accordion("📖 ¿Qué estoy viendo? (triage)", open=False):
                    gr.Markdown(PANELES["triage"])
                with gr.Accordion("📖 ¿Qué estoy viendo? (cuadrantes)", open=False):
                    gr.Markdown(PANELES["cuadrantes"])
            for ctrl in (t_slider, t_prompt):
                ctrl.change(actualizar_triage, [t_prompt, t_slider],
                            [t_fig, t_cuad, t_resumen])
            demo.load(actualizar_triage, [t_prompt, t_slider],
                      [t_fig, t_cuad, t_resumen])

        with gr.Tab("📑 Resultados y tablas"):
            gr.Markdown(
                "Las tablas del paper, con sus cifras exactas (calculadas con "
                "bootstrap BCa de 9.999 remuestreos en `src/evaluation.py`)."
            )
            with gr.Accordion("Tabla 1 — Resultados principales", open=True):
                gr.Dataframe(dd.load_tabla("tabla_t1_resultados.csv"))
            with gr.Accordion("Tabla 2 — Ablaciones (variantes de la señal)", open=False):
                gr.Dataframe(dd.load_tabla("tabla_t2_ablaciones.csv"))
            with gr.Accordion("Tabla 4 — Costo vs. beneficio", open=False):
                gr.Dataframe(dd.load_tabla("tabla_t4_costo_beneficio.csv"))
            with gr.Accordion("Tabla 5 — Discriminación + calibración", open=False):
                gr.Dataframe(dd.load_tabla("tabla_t5_calibracion.csv"))
            gr.Markdown("---\n## Calibración y severidad\n" + CONCEPTOS["calibracion"])
            gr.Markdown(
                "> **¿Por qué el diagrama usa u(x) calibrada y no la confianza del "
                "modelo?** MedGemma está extremadamente sobreconfiado (p_yes mediana "
                "≈ 0.9999): su confianza cruda casi no varía entre imágenes y un "
                "diagrama sobre ella se vería vacío. El reliability diagram oficial "
                "se construye sobre u(x) transformada con Platt scaling (ajustado "
                "solo en train) → P(error), igual que la Fig 10 del paper."
            )
            obs_rel = _observations_con_pred("P1")
            gr.Plot(fig_reliability(obs_rel))
            gr.Markdown(CONCEPTOS["h4"])
            gr.Plot(fig_h4("P1"))

        with gr.Tab("🤖 Asistente IA"):
            modo = ("**Modo IA completo** (LLM de Hugging Face conectado)."
                    if assistant.llm_disponible() else
                    "**Modo local** (sin `HF_TOKEN`): respondo con el glosario y las "
                    "cifras del estudio. Configura el secret `HF_TOKEN` en el Space "
                    "para activar el modo IA completo.")
            gr.Markdown(
                "Pregúntame lo que quieras sobre el experimento: cifras reales, "
                "conceptos (KL, AUROC, calibración, triage...) y límites del "
                "estudio. " + modo
            )
            gr.ChatInterface(
                fn=responder_chat,
                examples=assistant.EJEMPLOS_PREGUNTAS,
            )

        gr.Markdown("---\n" + DISCLAIMER)

    return demo


def _observations_con_pred(prompt: str) -> pd.DataFrame:
    frame = dd.get_u(prompt, dd.WINNER["family"], dd.WINNER["pooling"], dd.WINNER["tau"])
    return frame


app = build_app()

if __name__ == "__main__":
    app.launch()
