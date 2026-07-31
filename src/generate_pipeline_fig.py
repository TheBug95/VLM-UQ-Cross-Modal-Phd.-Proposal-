"""src/generate_pipeline_fig.py — Figura 1 estilo paper (Visual, elegante, sin textos largos ni flechas superpuestas).

Diagrama de arquitectura compacto y altamente visual para publicación.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from pathlib import Path
import numpy as np


# Paleta de colores refinada para paper
C_NAVY      = "#002D54"
C_NAVY_LT   = "#004080"
C_BLUE       = "#1F77B4"
C_BLUE_LT    = "#E6F0FA"
C_RED        = "#D62728"
C_RED_BG     = "#FDE8E8"
C_PURPLE     = "#8C564B"
C_PURPLE_BG  = "#F4EFEB"
C_PURPLE_DARK= "#6B3E2E"
C_GREEN      = "#2CA02C"
C_GREEN_BG   = "#EAF5EA"
C_GRAY       = "#444444"
C_GRAY_LT    = "#777777"
C_WHITE      = "#FFFFFF"
C_DARK       = "#1A1A2E"
C_BG         = "#FFFFFF"
C_BOX_BG     = "#F8F9FA"


def _box(ax, x, y, w, h, fc=C_BOX_BG, ec=C_NAVY, lw=1.5, ls="-", radius=0.12, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.04,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z
    )
    ax.add_patch(p)
    return p


def _text(ax, x, y, txt, fs=9, c="#111", w="normal", ha="center", va="center", style="normal", z=10):
    ax.text(x, y, txt, fontsize=fs, color=c, fontweight=w, ha=ha, va=va, fontstyle=style, zorder=z)


def _math(ax, x, y, expr, fs=9.5, c="#111", ha="center", va="center", z=10):
    ax.text(x, y, expr, fontsize=fs, color=c, ha=ha, va=va, zorder=z)


def _arrow(ax, x0, y0, x1, y1, c=C_NAVY, lw=1.8, style="-|>", rad=0.0, z=8):
    connection = f"arc3,rad={rad}"
    p = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=c, zorder=z,
        connectionstyle=connection
    )
    ax.add_patch(p)


def draw_pipeline_figure(output_path: Path):
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # -----------------------------------------------------------------------
    # TÍTULO PRINCIPAL (MINIMALISTA)
    # -----------------------------------------------------------------------
    _text(ax, 10.0, 9.5, "Figure 1: Overview of the Cross-Modal Uncertainty Quantification & Triage Framework",
          fs=13, c=C_NAVY, w="bold")

    # -----------------------------------------------------------------------
    # BLOQUE 1: MULTIMODAL INPUT (x: 0.5 .. 3.5, y: 5.2 .. 8.5)
    # -----------------------------------------------------------------------
    b1_x, b1_y, b1_w, b1_h = 0.5, 5.2, 3.0, 3.3
    _box(ax, b1_x, b1_y, b1_w, b1_h, fc="#F0F4F8", ec=C_NAVY, lw=1.5)
    _text(ax, b1_x + b1_w/2, b1_y + b1_h - 0.35, "1. Multimodal Input", fs=10, c=C_NAVY, w="bold")

    # Icono del fondo de ojo
    eye_x, eye_y, eye_r = b1_x + b1_w/2, b1_y + 1.85, 0.55
    ax.add_patch(Circle((eye_x, eye_y), eye_r, fc="#E8A87C", ec=C_NAVY, lw=1.2, zorder=4))
    ax.add_patch(Circle((eye_x + 0.12, eye_y), 0.18, fc="#F6D55C", ec="#C8963E", lw=0.8, zorder=5))
    ax.add_patch(Circle((eye_x + 0.12, eye_y), 0.08, fc="#FFF", ec="#C8963E", lw=0.6, zorder=6))
    for ang in [25, 145, 215, 335]:
        r = np.radians(ang)
        ax.plot([eye_x, eye_x + eye_r * 0.8 * np.cos(r)], [eye_y, eye_y + eye_r * 0.8 * np.sin(r)],
                color="#A83232", lw=0.8, zorder=5)

    _math(ax, b1_x + b1_w/2, b1_y + 1.05, r"$X_{img} \in R^{896 \times 896 \times 3}$", fs=8.5, c=C_GRAY)
    _text(ax, b1_x + b1_w/2, b1_y + 0.45, 'Prompt: "Does this image\nshow glaucoma?"', fs=8, c=C_GRAY, style="italic")

    # -----------------------------------------------------------------------
    # BLOQUE 2: VISION ENCODER & TOKENIZER (x: 4.1 .. 7.1, y: 5.2 .. 8.5)
    # -----------------------------------------------------------------------
    b2_x, b2_y, b2_w, b2_h = 4.1, 5.2, 3.0, 3.3
    _box(ax, b2_x, b2_y, b2_w, b2_h, fc="#EBF3FA", ec=C_NAVY_LT, lw=1.5)
    _text(ax, b2_x + b2_w/2, b2_y + b2_h - 0.35, "2. Vision & Text Encoders", fs=10, c=C_NAVY_LT, w="bold")

    # Vision Encoder box inside
    _box(ax, b2_x + 0.2, b2_y + 1.6, b2_w - 0.4, 1.0, fc=C_WHITE, ec=C_NAVY_LT, lw=1)
    _text(ax, b2_x + b2_w/2, b2_y + 2.3, "MedSigLIP (14x14 Patches)", fs=8.5, c=C_NAVY_LT, w="bold")
    _math(ax, b2_x + b2_w/2, b2_y + 1.85, r"256 tokens $\rightarrow R^{256 \times 2560}$", fs=8, c=C_GRAY)

    # Grid 4x4 visual tokens representation
    gx0, gy0 = b2_x + 0.35, b2_y + 0.4
    for row in range(3):
        for col in range(8):
            _box(ax, gx0 + col*0.28, gy0 + row*0.28, 0.24, 0.24, fc="#B3D4E5", ec=C_NAVY_LT, lw=0.5, radius=0.02)
    _text(ax, b2_x + b2_w/2, b2_y + 0.18, "Visual Soft Tokens (N=256)", fs=7.5, c=C_GRAY)

    # Arrow 1 -> 2
    _arrow(ax, b1_x + b1_w + 0.05, b1_y + b1_h/2, b2_x - 0.05, b2_y + b2_h/2, c=C_NAVY, lw=2)

    # -----------------------------------------------------------------------
    # BLOQUE 3: MEDGEMMA-4B FROZEN DECODER (x: 7.7 .. 11.2, y: 5.2 .. 8.5)
    # -----------------------------------------------------------------------
    b3_x, b3_y, b3_w, b3_h = 7.7, 5.2, 3.5, 3.3
    _box(ax, b3_x, b3_y, b3_w, b3_h, fc="#E8F1F8", ec=C_BLUE, lw=1.5)
    _text(ax, b3_x + b3_w/2, b3_y + b3_h - 0.35, "3. MedGemma-4B (Frozen)", fs=10, c=C_BLUE, w="bold")

    # Stack of layers diagram
    ly0 = b3_y + 0.7
    for i, (l_name, l_col) in enumerate([
        ("Layer 1..16", "#D5E8F0"), ("Layer 17", "#B3D4E5"),
        ("Layer 18..26", "#90C0DA"), ("Layer 27..33", "#6DADD0")
    ]):
        _box(ax, b3_x + 0.3, ly0 + (3-i)*0.4, b3_w - 0.6, 0.32, fc=l_col, ec=C_BLUE, lw=0.6, radius=0.04)
        _text(ax, b3_x + b3_w/2, ly0 + (3-i)*0.4 + 0.16, l_name, fs=7.5, c=C_GRAY)

    # Layer 34 highlighted block
    _box(ax, b3_x + 0.25, ly0 - 0.35, b3_w - 0.5, 0.42, fc=C_BLUE, ec=C_NAVY, lw=1.5, radius=0.06)
    _text(ax, b3_x + b3_w/2, ly0 - 0.14, "Layer 34 (Target Representations)", fs=8.5, c="white", w="bold")

    _text(ax, b3_x + b3_w/2, b3_y + 0.18, "Single Pass (1x) | Greedy Decoding", fs=7.5, c=C_GRAY, style="italic")

    # Arrow 2 -> 3
    _arrow(ax, b2_x + b2_w + 0.05, b2_y + b2_h/2, b3_x - 0.05, b3_y + b3_h/2, c=C_NAVY, lw=2)

    # -----------------------------------------------------------------------
    # BLOQUE 4B: CROSS-MODAL KL MODULE (x: 0.5 .. 8.0, y: 1.0 .. 4.2)
    # -----------------------------------------------------------------------
    b4b_x, b4b_y, b4b_w, b4b_h = 0.5, 1.0, 7.5, 3.2
    _box(ax, b4b_x, b4b_y, b4b_w, b4b_h, fc=C_RED_BG, ec=C_RED, lw=2.0)
    _text(ax, b4b_x + b4b_w/2, b4b_y + b4b_h - 0.35,
          "4. Cross-Modal Disagreement Module (Original Contribution 1)", fs=10, c=C_RED, w="bold")

    # Step A: High-precision Softmax
    _box(ax, b4b_x + 0.2, b4b_y + 1.7, 3.4, 0.95, fc=C_WHITE, ec=C_RED, lw=1)
    _text(ax, b4b_x + 1.9, b4b_y + 2.4, "High-Precision Softmax (float64)", fs=8, c=C_RED, w="bold")
    _math(ax, b4b_x + 1.9, b4b_y + 2.0, r"$p_{text},\; p_{vis}^{(i)} = \mathrm{softmax}(h/\tau)$", fs=8.5, c=C_DARK)

    # Step B: Asymmetric KL Divergence
    _box(ax, b4b_x + 3.9, b4b_y + 1.7, 3.4, 0.95, fc=C_WHITE, ec=C_RED, lw=1)
    _text(ax, b4b_x + 5.6, b4b_y + 2.4, "Asymmetric Divergence", fs=8, c=C_RED, w="bold")
    _math(ax, b4b_x + 5.6, b4b_y + 2.0, r"$D_{KL}(p_{text} \parallel p_{vis}^{(i)})\;\; \forall i \in \{1..256\}$", fs=8.5, c=C_DARK)

    # Step C: Spatial Max Pooling
    _box(ax, b4b_x + 1.5, b4b_y + 0.3, 4.5, 1.1, fc=C_WHITE, ec=C_RED, lw=1.5)
    _text(ax, b4b_x + 3.75, b4b_y + 1.15, "Spatial Max-Pooling (Avoids Dilution)", fs=8.5, c=C_RED, w="bold")
    _math(ax, b4b_x + 3.75, b4b_y + 0.65, r"$u_{KL}(x) = \max_{i \in \{1..256\}} D_{KL}(p_{text} \parallel p_{vis}^{(i)})$", fs=9.5, c=C_DARK)

    # Arrow inside 4b: A -> C and B -> C
    _arrow(ax, b4b_x + 1.9, b4b_y + 1.65, b4b_x + 2.5, b4b_y + 1.45, c=C_RED, lw=1.2)
    _arrow(ax, b4b_x + 5.6, b4b_y + 1.65, b4b_x + 5.0, b4b_y + 1.45, c=C_RED, lw=1.2)

    # -----------------------------------------------------------------------
    # BLOQUE 4A: OUTPUT LOGITS & MSP (x: 8.5 .. 12.0, y: 1.0 .. 4.2)
    # -----------------------------------------------------------------------
    b4a_x, b4a_y, b4a_w, b4a_h = 8.5, 1.0, 3.5, 3.2
    _box(ax, b4a_x, b4a_y, b4a_w, b4a_h, fc="#F5F5F5", ec=C_GRAY, lw=1.5)
    _text(ax, b4a_x + b4a_w/2, b4a_y + b4a_h - 0.35, "5. Output Logits UQ", fs=10, c=C_GRAY, w="bold")

    _text(ax, b4a_x + b4a_w/2, b4a_y + 2.3, "First-Token Probabilities", fs=8.5, c=C_GRAY, w="bold")
    _math(ax, b4a_x + b4a_w/2, b4a_y + 1.85, r"$P(yes),\; P(no)$", fs=9, c=C_DARK)

    _box(ax, b4a_x + 0.25, b4a_y + 0.4, b4a_w - 0.5, 1.0, fc=C_WHITE, ec=C_GRAY, lw=1)
    _text(ax, b4a_x + b4a_w/2, b4a_y + 1.15, "Output Uncertainty", fs=8, c=C_GRAY, w="bold")
    _math(ax, b4a_x + b4a_w/2, b4a_y + 0.75, r"$u_{MSP} = 1 - \max(P_{yes}, P_{no})$", fs=8.5, c=C_DARK)

    # -----------------------------------------------------------------------
    # BLOQUE 5: PARAMETER-FREE RANK FUSION (x: 12.5 .. 15.7, y: 1.0 .. 4.2)
    # -----------------------------------------------------------------------
    b5_x, b5_y, b5_w, b5_h = 12.5, 1.0, 3.2, 3.2
    _box(ax, b5_x, b5_y, b5_w, b5_h, fc=C_PURPLE_BG, ec=C_PURPLE, lw=2.0)
    _text(ax, b5_x + b5_w/2, b5_y + b5_h - 0.35, "6. Rank Fusion", fs=10, c=C_PURPLE_DARK, w="bold")
    _text(ax, b5_x + b5_w/2, b5_y + b5_h - 0.7, "(Contrib. 2)", fs=8, c=C_PURPLE_DARK, style="italic")

    _math(ax, b5_x + b5_w/2, b5_y + 2.0, r"$u_{combo} = \mathrm{rank}(u_{KL})$", fs=9, c=C_PURPLE_DARK)
    _math(ax, b5_x + b5_w/2, b5_y + 1.6, r"$+ \mathrm{rank}(u_{MSP})$", fs=9, c=C_PURPLE_DARK)

    _box(ax, b5_x + 0.2, b5_y + 0.4, b5_w - 0.4, 0.85, fc=C_WHITE, ec=C_PURPLE, lw=1)
    _text(ax, b5_x + b5_w/2, b5_y + 0.95, "AUROC Performance", fs=7.5, c=C_PURPLE, w="bold")
    _text(ax, b5_x + b5_w/2, b5_y + 0.65, "Combo: 0.698 (p<0.001)", fs=8.5, c=C_PURPLE_DARK, w="bold")

    # -----------------------------------------------------------------------
    # BLOQUE 6: SELECTIVE CLINICAL TRIAGE (x: 16.2 .. 19.5, y: 1.0 .. 8.5)
    # -----------------------------------------------------------------------
    b6_x, b6_y, b6_w, b6_h = 16.2, 1.0, 3.3, 7.5
    _box(ax, b6_x, b6_y, b6_w, b6_h, fc="#F4F6F9", ec=C_NAVY, lw=2.0)
    _text(ax, b6_x + b6_w/2, b6_y + b6_h - 0.4, "7. Selective Triage", fs=10.5, c=C_NAVY, w="bold")
    _math(ax, b6_x + b6_w/2, b6_y + b6_h - 0.9, r"Compare $u_{combo}$ vs $\theta$", fs=9, c=C_NAVY)

    # Branch 1: Low Uncertainty (Green)
    _box(ax, b6_x + 0.2, b6_y + 3.8, b6_w - 0.4, 2.9, fc=C_GREEN_BG, ec=C_GREEN, lw=1.8)
    _text(ax, b6_x + b6_w/2, b6_y + 6.3, "Low Uncertainty", fs=9.5, c=C_GREEN, w="bold")
    _math(ax, b6_x + b6_w/2, b6_y + 5.9, r"$u_{combo} < \theta$", fs=8.5, c=C_GREEN)
    _text(ax, b6_x + b6_w/2, b6_y + 5.3, "[ACCEPT DIAGNOSIS]", fs=8.5, c=C_GREEN, w="bold")
    _text(ax, b6_x + b6_w/2, b6_y + 4.7, "Autonomous VLM Output\nGlaucoma / Normal", fs=7.5, c="#1E8449")
    _text(ax, b6_x + b6_w/2, b6_y + 4.1, "Acc @ 70% coverage: 82.2%\nAcc @ 50% coverage: 89.1%", fs=7.5, c="#1E8449", w="bold")

    # Branch 2: High Uncertainty (Red)
    _box(ax, b6_x + 0.2, b6_y + 0.4, b6_w - 0.4, 3.1, fc=C_RED_BG, ec=C_RED, lw=1.8)
    _text(ax, b6_x + b6_w/2, b6_y + 3.1, "High Uncertainty", fs=9.5, c=C_RED, w="bold")
    _math(ax, b6_x + b6_w/2, b6_y + 2.7, r"$u_{combo} >= \theta$", fs=8.5, c=C_RED)
    _text(ax, b6_x + b6_w/2, b6_y + 2.1, "[REFER TO CLINICIAN]", fs=8.5, c=C_RED, w="bold")
    _text(ax, b6_x + b6_w/2, b6_y + 1.4, "Human Ophthalmologist\nTriage & Review", fs=7.5, c="#922B21")
    _text(ax, b6_x + b6_w/2, b6_y + 0.75, "Prevents misdiagnosis\non ambiguous cases", fs=7, c="#922B21", style="italic")

    # -----------------------------------------------------------------------
    # FLECHAS Y CONEXIONES LIMPIAS (SIN CRUCES SOBRE CAJAS)
    # -----------------------------------------------------------------------

    # Decoder (Bloque 3) -> Bloque 4b (KL Module)
    # sale de la base izquierda de 3 (x=8.5, y=5.2) y cae a 4b top (x=4.25, y=4.2)
    _arrow(ax, 8.5, 5.15, 4.25, 4.25, c=C_RED, lw=2.0, rad=0.1)
    _math(ax, 6.0, 4.85, r"Layer 34 states $(H_{vis},\, h_{text})$", fs=8.5, c=C_RED)

    # Decoder (Bloque 3) -> Bloque 4a (Output Logits)
    # sale de la base derecha de 3 (x=10.2, y=5.2) y cae a 4a top (x=10.2, y=4.2)
    _arrow(ax, 10.2, 5.15, 10.2, 4.25, c=C_GRAY, lw=1.8)
    _text(ax, 10.8, 4.7, "Logits", fs=8, c=C_GRAY, style="italic")

    # Bloque 4b (KL) -> Bloque 6 (Rank Fusion)
    # RUTA LIMPIA POR DEBAJO DE 4A (y = 0.55)
    _arrow(ax, 8.05, 1.5, 8.25, 0.55, c=C_RED, lw=1.8, style="-", rad=0.0)
    _arrow(ax, 8.25, 0.55, 12.25, 0.55, c=C_RED, lw=1.8, style="-", rad=0.0)
    _arrow(ax, 12.25, 0.55, 12.45, 1.5, c=C_RED, lw=1.8, style="-|>", rad=0.0)
    _math(ax, 10.25, 0.75, r"$u_{KL}(x)$", fs=9, c=C_RED)

    # Bloque 4a (MSP) -> Bloque 5 (Rank Fusion)
    _arrow(ax, 12.05, 2.6, 12.45, 2.6, c=C_GRAY, lw=1.8)
    _math(ax, 12.25, 2.85, r"$u_{MSP}$", fs=8.5, c=C_GRAY)

    # Bloque 5 (Rank Fusion) -> Bloque 6 (Triage)
    _arrow(ax, 15.75, 2.6, 16.15, 2.6, c=C_PURPLE, lw=2.2)

    # -----------------------------------------------------------------------
    # GUARDAR
    # -----------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"[fig1] Paper-style clean pipeline diagram saved to: {output_path}")


if __name__ == "__main__":
    out = Path("figures/fig1_pipeline.png")
    draw_pipeline_figure(out)
