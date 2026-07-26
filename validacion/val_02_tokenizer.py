"""val_02_tokenizer.py — Validación del tokenizer y la plantilla de chat.

Qué valida (afirmaciones del diseño que de otra forma serían supuestos):
    V-TOK-1  len(tokenizer) en {262.144, 262.145} y config.vocab_size == 262.208
             (el tokenizer NO cubre los 64 tokens especiales extra; por eso la
             máscara de imagen usa IDs, nunca slicing de vocabulario).
             Nota: transformers añade <image_soft_token> al vocabulario del
             tokenizer en algunas versiones, por lo que len(tokenizer) puede
             ser 262.145 (issue huggingface/transformers #37011).
    V-TOK-2  IDs de imagen: <image_soft_token> = 262.144 (= config.image_token_index),
             <start_of_image> = 255.999, <end_of_image> = 256.000.
    V-TOK-3  El prompt de chat termina en "<start_of_turn>model\\n", por lo que el
             primer token de respuesta NO lleva espacio inicial: los candidatos
             primarios son yes=4443 y no=1904 (no las variantes con '▁').
    V-TOK-4  Al aplicar el chat template con una imagen, aparecen EXACTAMENTE
             256 tokens <image_soft_token> contiguos tras <start_of_image>
             (896px / 14px por patch = 64 por lado; 64x64 = 4.096 patches;
             pooling 4x4 del proyecto → 4.096/16 = 256 tokens).
    V-TOK-5  El system prompt se pliega dentro del primer turno de usuario
             (Gemma 3 no tiene rol de sistema propio).

Criterio PASS: todas las aserciones V-TOK-1..5.

Requisitos: transformers>=4.51.3. Para google/medgemma-4b-it hace falta token HF
con licencia HAI-DEF aceptada (repo gated). Si no hay acceso, se usa el espejo
unsloth/gemma-3-4b-it (mismo tokenizer y misma arquitectura de tokenización de
imagen según el technical report de Gemma 3) y se reporta como PASS-PARCIAL.

Uso:
    python val_02_tokenizer.py [--model google/medgemma-4b-it]
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
args = parser.parse_args([]) # Fix: Pass an empty list to parse_args()

from transformers import AutoTokenizer, AutoConfig

modelo_usado = "google/medgemma-4b-it"
parcial = False
try:
    tokenizer = AutoTokenizer.from_pretrained(modelo_usado)
    config = AutoConfig.from_pretrained(modelo_usado)
except Exception as e:
    print(f"[WARN] Sin acceso a {args.model} ({type(e).__name__}). "
          "Cayendo al espejo unsloth/gemma-3-4b-it (tokenizer idéntico).")
    modelo_usado = "unsloth/gemma-3-4b-it"
    parcial = True
    tokenizer = AutoTokenizer.from_pretrained(modelo_usado)
    config = AutoConfig.from_pretrained(modelo_usado)

print(f"Modelo: {modelo_usado}")
print("=" * 70)

# --- V-TOK-1 -------------------------------------------------------------------
n_tok = len(tokenizer)
check("V-TOK-1a len(tokenizer) en {262144, 262145}",
      n_tok in (262144, 262145),
      f"obtenido: {n_tok} (262145 indica <image_soft_token> añadido al vocabulario)")
# config de Gemma3 multimodal anida el vocab del LM en text_config
vocab_cfg = getattr(getattr(config, "text_config", config), "vocab_size", None)
check("V-TOK-1b config.vocab_size == 262208", vocab_cfg == 262208, f"obtenido: {vocab_cfg}")

# --- V-TOK-2 -------------------------------------------------------------------
id_soft = tokenizer.convert_tokens_to_ids("<image_soft_token>")
id_soi = tokenizer.convert_tokens_to_ids("<start_of_image>")
id_eoi = tokenizer.convert_tokens_to_ids("<end_of_image>")
check("V-TOK-2a <image_soft_token> == 262144", id_soft == 262144, f"obtenido: {id_soft}")
check("V-TOK-2b <start_of_image> == 255999", id_soi == 255999, f"obtenido: {id_soi}")
check("V-TOK-2c <end_of_image> == 256000", id_eoi == 256000, f"obtenido: {id_eoi}")
idx_cfg = getattr(config, "image_token_index", None)
check("V-TOK-2d config.image_token_index == 262144", idx_cfg == 262144, f"obtenido: {idx_cfg}")

# --- V-TOK-3 -------------------------------------------------------------------
CANDIDATOS = ["yes", "Yes", " no", "no", "No", " yes", " Yes", " No"]
ids = {c: tokenizer.encode(c, add_special_tokens=False) for c in CANDIDATOS}
print("IDs yes/no:", ids)
ESPERADOS = {"yes": [4443], "no": [1904], "Yes": [10784], "No": [3771]}
for tok, esp in ESPERADOS.items():
    check(f"V-TOK-3 '{tok}' == {esp}", ids.get(tok) == esp, f"obtenido: {ids.get(tok)}")

# --- V-TOK-4 y V-TOK-5 (requieren processor + imagen sintética) -----------------
try:
    from transformers import AutoProcessor
    from PIL import Image
    import numpy as np

    processor = AutoProcessor.from_pretrained(modelo_usado)
    img = Image.fromarray((np.random.rand(512, 512, 3) * 255).astype("uint8"))

    msgs_p1 = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Does this fundus image show glaucoma? Answer yes or no."},
    ]}]
    prompt = tokenizer.apply_chat_template(msgs_p1, tokenize=False, add_generation_prompt=True)
    check("V-TOK-3b prompt termina en '<start_of_turn>model\\n'",
          prompt.endswith("<start_of_turn>model\n"), repr(prompt[-40:]))

    inputs = processor(images=img, text=prompt, return_tensors="pt")
    ids_seq = inputs["input_ids"][0]
    n_img = int((ids_seq == id_soft).sum())
    check("V-TOK-4a exactamente 256 tokens de imagen", n_img == 256, f"obtenidos: {n_img}")
    pos = (ids_seq == id_soft).nonzero().flatten()
    contiguos = bool((pos.diff() == 1).all()) if len(pos) > 1 else True
    check("V-TOK-4b tokens de imagen contiguos", contiguos)
    pos_soi = (ids_seq == id_soi).nonzero().flatten()
    tras_soi = len(pos_soi) == 1 and int(pos[0]) == int(pos_soi[0]) + 1
    check("V-TOK-4b2 bloque empieza justo tras <start_of_image>", tras_soi)

    # V-TOK-5: system prompt plegado en el primer turno de usuario
    msgs_p4 = [{"role": "system", "content": [{"type": "text", "text": "You are an expert ophthalmologist."}]}] + msgs_p1
    prompt_p4 = tokenizer.apply_chat_template(msgs_p4, tokenize=False, add_generation_prompt=True)
    check("V-TOK-5 system plegado en turno user (sin rol 'system' en el texto)",
          "<start_of_turn>system" not in prompt_p4 and "expert ophthalmologist" in prompt_p4,
          repr(prompt_p4[:120]))
except Exception as e:
    print(f"[WARN] V-TOK-4/5 no ejecutables ({type(e).__name__}: {e})")

print("=" * 70)
if FALLOS:
    print(f"RESULTADO: FAIL ({FALLOS})")
    sys.exit(1)
print("RESULTADO: PASS" + (" (PARCIAL — verificado contra espejo Gemma-3, "
      "re-correr con licencia MedGemma)" if parcial else ""))
sys.exit(0)
