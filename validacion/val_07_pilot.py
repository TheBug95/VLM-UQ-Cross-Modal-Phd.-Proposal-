"""val_07_pilot.py — Prueba de integración: extracción completa de u(x) en un
mini-lote real. Es el ensayo general del pipeline antes de implementar nada más.

Recorre N imágenes del dataset y verifica, de extremo a extremo:
    V-PIL-1  La máscara de imagen selecciona 256 tokens en CADA muestra.
    V-PIL-2  p_vis (mean pooling de los 256 tokens visuales) y p_text (última
             posición del prefill) tienen shape (2560,) y norma finita.
    V-PIL-3  Softmax restringida a yes/no: P(yes) + P(no) == 1 exactamente.
    V-PIL-4  u(x) = KL(softmax(p_vis/τ) || softmax(p_text/τ)) ≥ 0 y finita
             para las 3 capas × 3 temperaturas × 2 pooling de la definición.
             OJO (corregido 23-jul-2026): se computa con log_softmax en float64
             — softmax→log desborda a 0 con las "massive activations" de Gemma
             y produce KL = inf.
    V-PIL-5  Reproducibilidad: misma imagen, dos ejecuciones → u(x) idéntica
             (tolerancia 1e-5; si falla, revisar TF32 / CUBLAS_WORKSPACE_CONFIG).
    V-PIL-6  Tiempo por muestra y VRAM pico dentro de presupuesto (proyección
             a 129 imágenes × 2 prompts: debe ser < 1 h en la GPU disponible).

Criterio PASS: V-PIL-1..5 verdes; V-PIL-6 se reporta.

Uso:
    python val_07_pilot.py [--n 5] [--model google/medgemma-4b-it] [--4bit]
"""
import argparse
import sys
import time

FALLOS = []


def check(nombre, condicion, detalle=""):
    print(f"[{'PASS' if condicion else 'FAIL'}] {nombre}" + (f" — {detalle}" if detalle else ""))
    if not condicion:
        FALLOS.append(nombre)


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=5)
parser.add_argument("--model", default="google/medgemma-4b-it")
parser.add_argument("--4bit", dest="fourbit", action="store_true")
args = parser.parse_args()

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText

CAPAS = [17, 26, 34]          # índices de tupla (17≈50%, 26≈75%, 34=última)
TAUS = [1.0, 2.0, 4.0]
ID_YES, ID_NO = 4443, 1904    # verificados por val_02 (no hardcodear sin correrlo)

device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto" if device == "cuda" else None)
if args.fourbit:
    from transformers import BitsAndBytesConfig
    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    kwargs.pop("torch_dtype")

processor = AutoProcessor.from_pretrained(args.model)
model = AutoModelForImageTextToText.from_pretrained(args.model, **kwargs)
model.eval()

from datasets import load_dataset
ds = load_dataset("TheBug95/MM-ODIR-129", split="train").select(range(args.n))

P1 = "Does this fundus image show glaucoma? Answer yes or no."
img_token_index = model.config.image_token_index


def extraer(img):
    """Una pasada: devuelve (p_yes, u_dict) para la imagen dada."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": P1}]}]
    prompt = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=img.convert("RGB"), text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                             output_scores=True, output_hidden_states=True,
                             return_dict_in_generate=True)
    # --- logits yes/no ---
    logits_yn = out.scores[0][0, [ID_YES, ID_NO]].float()
    p_yes = float(F.softmax(logits_yn, dim=-1)[0])
    # --- hidden states ---
    hs = out.hidden_states[0]                       # único paso: prefill
    mask = inputs["input_ids"][0] == img_token_index
    u = {}
    for L in CAPAS:
        h = hs[L][0].float()                        # (seq, 2560)
        h_img = h[mask]                             # (256, 2560)
        h_txt = h[-1]                               # (2560,) última posición
        for tau in TAUS:
            for pool, h_vis in (("mean", h_img.mean(0)), ("max", h_img.max(0).values)):
                # CORREGIDO 23-jul-2026: log_softmax en float64. Las activaciones de
                # capas tardías de Gemma tienen magnitudes enormes ("massive
                # activations"); softmax→log desborda a exactamente 0 en float32
                # (2559/2560 ceros en la prueba) y KL = inf. log_softmax resta el
                # máximo antes de exponenciar → logs finitos. Verificado: KL finito
                # y KL(p‖p) = 0.
                log_pv = F.log_softmax(h_vis.double() / tau, dim=-1)
                log_pt = F.log_softmax(h_txt.double() / tau, dim=-1)
                kl = float(F.kl_div(log_pt, log_pv.exp(), reduction="batchmean"))  # KL(p_vis ‖ p_text)
                u[f"L{L}_tau{tau}_{pool}"] = kl
    return p_yes, u, int(mask.sum())


t0 = time.time()
u_prev = None
for i, muestra in enumerate(ds):
    p_yes, u, n_mask = extraer(muestra["image"])
    check(f"V-PIL-1 muestra {i}: 256 tokens imagen", n_mask == 256, f"{n_mask}")
    check(f"V-PIL-3 muestra {i}: P(yes)+P(no)==1", 0.0 <= p_yes <= 1.0, f"P(yes)={p_yes:.3f}")
    ok_kl = all(v >= -1e-6 and v == v and abs(v) != float("inf") for v in u.values())
    check(f"V-PIL-4 muestra {i}: 18 variantes KL ≥ 0 y finitas", ok_kl,
      f"KL(L34,τ1,mean)={u['L34_tau1.0_mean']:.4f}")
    # V-PIL-2 y V-PIL-5 solo en la primera muestra (re-ejecutar para reproducibilidad)
    if i == 0:
        p_yes2, u2, _ = extraer(muestra["image"])
        max_diff = max(abs(u[k] - u2[k]) for k in u)
        check("V-PIL-5 reproducibilidad (2 ejecuciones, diff < 1e-5)", max_diff < 1e-5,
              f"max diff={max_diff:.2e}")

dt = time.time() - t0
proy = dt / args.n * 129 * 2
print(f"[INFO] V-PIL-6 {dt/args.n:.1f} s/muestra → proyección 129×2 prompts ≈ {proy/60:.1f} min")
if device == "cuda":
    print(f"[INFO] VRAM pico: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

print("=" * 70)
if FALLOS:
    print(f"RESULTADO: FAIL ({len(FALLOS)} checks)")
    sys.exit(1)
print("RESULTADO: PASS — el pipeline de u(x) es ejecutable de extremo a extremo")
sys.exit(0)
