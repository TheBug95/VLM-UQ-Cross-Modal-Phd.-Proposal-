"""Verificación independiente de la 'zona verde' (39 casos, 30.2%).

Afirmación a verificar: ordenando las 129 imágenes de MENOS a MÁS incierto
según u(x) = rank(KL ganadora) + rank(1-MSP), los primeros 39 casos son
todos correctos (el primer error está en la posición 40).

Método: recomputar desde results_full.csv sin reutilizar el código que
generó el número, y cruzar contra la curva de la Fig 4.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("results/results_full.csv")
df["tau_s"] = df["tau"].astype(str)

# --- Datos base, cada uno por su propio camino ---
kl = df[(df.prompt_id == "P1") & (df.signal_type == "kl_t_v")
        & (df.pooling == "max") & (df.tau_s == "1.0")]
kl = kl[["image_filename", "value"]].rename(columns={"value": "kl"})
obs = (df[df.prompt_id == "P1"]
       .drop_duplicates("image_filename")
       [["image_filename", "msp_answer", "correct", "label", "split"]])
m = obs.merge(kl, on="image_filename")
assert len(m) == 129, f"esperaba 129, hay {len(m)}"

# --- u(x) recomputada a mano: rank fraccionario de KL + rank de (1-MSP) ---
n = len(m)
m["rank_kl"] = m["kl"].rank(method="average") / n
m["rank_msp"] = (1 - m["msp_answer"]).rank(method="average") / n
m["u"] = m["rank_kl"] + m["rank_msp"]

# --- Ordenar de MENOS a MÁS incierto y revisar la secuencia de errores ---
m = m.sort_values("u").reset_index(drop=True)
m["posicion"] = m.index + 1  # 1-indexed
errores = m[m.correct == 0]

print("=== Secuencia de errores en el ranking (posiciones de los 26 errores) ===")
print(sorted(errores.posicion.tolist()))

primera = int(errores.posicion.min())
print(f"\nPrimer error en la posición: {primera} (de 129, de menos a más incierto)")
print(f"Casos antes del primer error: {primera - 1} ({(primera - 1) / n:.1%})")
print(f"¿Todos correctos en los primeros {primera - 1}? ",
      bool(m.head(primera - 1).correct.sum() == primera - 1))

# --- Verificación cruzada contra la Fig 4 (misma lógica que figures.py) ---
print("\n=== Cruce con Fig 4 (accuracy reteniendo los MENOS inciertos) ===")
order_desc = np.argsort(-m.u.values)  # más incierto primero, como en figures.py
y = m.correct.values
for cov in [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.30, 0.32, 0.36, 0.50]:
    n_keep = int(np.ceil(n * cov))
    keep = order_desc[n - n_keep:]
    print(f"cobertura={cov:.2f} (n={n_keep:3d})  accuracy={y[keep].mean():.3f}")

# --- Muestra de los casos de la zona verde: ¿valores sensatos? ---
print("\n=== 5 casos al borde de la zona verde (posiciones 37-43) ===")
print(m.iloc[36:43][["posicion", "image_filename", "split", "kl",
                      "msp_answer", "u", "correct"]].to_string(index=False))
