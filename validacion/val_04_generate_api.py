"""val_04_generate_api.py — Validación de la API de generate() y la extracción de
hidden states / logits. ES EL SCRIPT MÁS CRÍTICO DEL PLAN: de aquí salen las
convenciones que usa todo el pipeline de la señal u(x).

Qué valida (contra transformers >= 4.51.3, verificado en modeling_gemma3.py):
    V-API-1  Con max_new_tokens=1, outputs.hidden_states tiene UN SOLO elemento
             (el prefill). hidden_states[1] NO EXISTE (IndexError). El estado
             que condiciona el primer token yes/no es
             hidden_states[0][capa][:, -1, :] (última posición del prefill).
    V-API-2  Cada tupla por paso tiene 35 entradas: índice 0 = embeddings de
             entrada (escalados por sqrt(2560)), índices 1..34 = salidas de las
             34 capas del decoder. NO mezclar el índice 0 con capas en la KL.
    V-API-3  outputs.scores[0] tiene shape (1, 262208) y su argmax coincide con
             el token efectivamente generado (outputs.sequences[0, -1]).
    V-API-4  En greedy puro (do_sample=False, sin processors), scores[0] ≈ los
             logits crudos de la última posición de un forward manual
             (equivalencia numérica que legitima restringir softmax a yes/no).
    V-API-5  La máscara input_ids == config.image_token_index selecciona
             exactamente 256 posiciones.
    V-API-6  Variante con max_new_tokens=2: ahora SÍ existe hidden_states[1]
             (estado tras generar el primer token) — queda como ablación.
    V-API-7  VRAM pico medido (torch.cuda.max_memory_allocated) — verifica la
             factibilidad en la GPU disponible (~10-12 GB bf16 / ~4-5 GB NF4).

Criterio PASS: V-API-1..6 en verde. V-API-7 se reporta (informativo).

Requisitos: GPU (o CPU con mucha paciencia), transformers>=4.51.3, acceso al
modelo (licencia HAI-DEF para google/medgemma-4b-it; si no, espejo Gemma-3).

Uso:
    python val_04_generate_api.py [--model google/medgemma-4b-it] [--4bit]
"""
import argparse
import sys

FALLOS = []


def check(nombre, condicion, detalle=""):
    print(f"[{'PASS' if condicion else 'FAIL'}] {nombre}" + (f" — {detalle}" if detalle else ""))
    if not condicion:
        FALLOS.append(nombre)


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="google/medgemma-4b-it")
parser.add_argument("--4bit", dest="fourbit", action="store_true",
                    help="Cargar en NF4 (bitsandbytes) si la GPU tiene < 14 GB")
args = parser.parse_args()

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig

modelo = args.model
parcial = False
try:
    AutoConfig.from_pretrained(modelo)
except Exception:
    print(f"[WARN] Sin acceso a {modelo}; usando espejo unsloth/gemma-3-4b-it.")
    modelo, parcial = "unsloth/gemma-3-4b-it", True

print("=" * 70)
print(f"VAL-04 · API DE GENERATE · {modelo}")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto" if device == "cuda" else None)
if args.fourbit:
    from transformers import BitsAndBytesConfig
    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    kwargs.pop("torch_dtype")

processor = AutoProcessor.from_pretrained(modelo)
model = AutoModelForImageTextToText.from_pretrained(modelo, **kwargs)
model.eval()

# --- Muestra mínima: imagen sintética + pregunta P1 -----------------------------
from PIL import Image
import numpy as np
img = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype("uint8"))
msgs = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Does this fundus image show glaucoma? Answer yes or no."},
]}]
prompt = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                         output_scores=True, output_hidden_states=True,
                         return_dict_in_generate=True)

# --- V-API-1 --------------------------------------------------------------------
hs = out.hidden_states
check("V-API-1a len(hidden_states) == 1 con max_new_tokens=1", len(hs) == 1,
      f"obtenido: {len(hs)} — si es >1, la versión de transformers difiere")
try:
    _ = hs[1]
    check("V-API-1b hidden_states[1] inaccesible", False,
          "¡existe! revisar versión de transformers y actualizar el diseño")
except IndexError:
    check("V-API-1b hidden_states[1] inaccesible (IndexError esperado)", True)

# --- V-API-2 --------------------------------------------------------------------
n_entradas = len(hs[0])
check("V-API-2a 35 entradas en la tupla de prefill", n_entradas == 35, f"obtenido: {n_entradas}")
seq_len = inputs["input_ids"].shape[1]
ok_shapes = all(hs[0][i].shape == (1, seq_len, 2560) for i in range(n_entradas))
check(f"V-API-2b shapes (1, {seq_len}, 2560) en todas las entradas", ok_shapes)
p_text = hs[0][34][:, -1, :]  # última capa, última posición del prefill
check("V-API-2c p_text extraíble, shape (1, 2560)", p_text.shape == (1, 2560),
      f"shape={tuple(p_text.shape)}")

# --- V-API-3 --------------------------------------------------------------------
scores0 = out.scores[0]
check("V-API-3a scores[0].shape == (1, 262208)", tuple(scores0.shape) == (1, 262208),
      f"shape={tuple(scores0.shape)}")
gen_id = out.sequences[0, -1].item()
check("V-API-3b argmax(scores[0]) == token generado",
      int(scores0.argmax()) == gen_id,
      f"argmax={int(scores0.argmax())} vs generado={gen_id} "
      f"('{processor.tokenizer.decode([gen_id])}')")

# --- V-API-4 --------------------------------------------------------------------
with torch.inference_mode():
    fwd = model(**inputs)
logits_manual = fwd.logits[:, -1, :]
coseno = torch.nn.functional.cosine_similarity(scores0.float(), logits_manual.float()).item()
check("V-API-4 scores[0] ≈ logits crudos del forward (cos ≈ 1)", coseno > 0.9999,
      f"cos={coseno:.6f} — si difiere, hay logit processors activos; usar output_logits=True")

# --- V-API-5 --------------------------------------------------------------------
img_idx = model.config.image_token_index
mask = inputs["input_ids"] == img_idx
check("V-API-5 máscara de imagen == 256 posiciones", int(mask.sum()) == 256,
      f"image_token_index={img_idx}, posiciones={int(mask.sum())}")

# --- V-API-6 --------------------------------------------------------------------
with torch.inference_mode():
    out2 = model.generate(**inputs, max_new_tokens=2, do_sample=False,
                          output_hidden_states=True, return_dict_in_generate=True)
check("V-API-6a con max_new_tokens=2 existen 2 pasos", len(out2.hidden_states) == 2)
h_post = out2.hidden_states[1][34][:, -1, :]
check("V-API-6b estado post-token (ablación) shape (1, 2560)", h_post.shape == (1, 2560),
      f"shape={tuple(h_post.shape)}")

# --- V-API-7 --------------------------------------------------------------------
if device == "cuda":
    vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[INFO] V-API-7 VRAM pico: {vram:.2f} GB "
          f"({'4-bit NF4' if args.fourbit else 'bf16'})")

print("=" * 70)
if FALLOS:
    print(f"RESULTADO: FAIL ({FALLOS})")
    sys.exit(1)
print("RESULTADO: PASS" + (" (PARCIAL — espejo Gemma-3; re-correr con MedGemma)" if parcial else ""))
sys.exit(0)
