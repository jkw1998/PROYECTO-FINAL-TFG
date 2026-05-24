# =============================================================
# TFG - MODELO IA: RNN (Recurrent Neural Network)
# =============================================================
# Autor: James Kagunda Wangari
# Grado en Ingenieria Electrica - Universidad de Malaga
#
# La red neuronal sustituye completamente al optimizador.
# Procesa los datos como secuencia temporal de 24 pasos.
#
# Entrada: (dias, 24 horas, 4 features) -> precio, pv, demanda, soc_ini
# Salida:  48 valores -> 24 p_bat + 24 soc
# =============================================================

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# --- Datos ---
df = pd.read_csv("dataset_vpp_ia.csv")
df.columns = df.columns.str.strip()
T = 24

cols_X = ['soc_inicial'] + [f'precio_h{i}' for i in range(T)] + \
         [f'pv_h{i}' for i in range(T)] + [f'demanda_h{i}' for i in range(T)]

cols_y = [f'p_bat_h{i}' for i in range(T)] + [f'soc_h{i}' for i in range(T)]

X = df[cols_X].values
y = df[cols_y].values

# --- Split ---
SPLIT = 800
X_train, X_test = X[:SPLIT], X[SPLIT:]
y_train, y_test = y[:SPLIT], y[SPLIT:]

# --- Normalizacion ---
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# --- Reshape a secuencia temporal (dias, 24, 4) ---
# Cada hora: [precio, pv, demanda, soc_inicial repetido]
def a_secuencia(X_sc):
    n       = X_sc.shape[0]
    soc_rep = np.repeat(X_sc[:, 0:1], T, axis=1)   # (n, 24)
    precios = X_sc[:, 1:25]
    pv      = X_sc[:, 25:49]
    demanda = X_sc[:, 49:73]
    return np.stack([precios, pv, demanda, soc_rep], axis=2)  # (n, 24, 4)

X_train_t = torch.FloatTensor(a_secuencia(X_train_sc))
y_train_t = torch.FloatTensor(y_train)
X_test_t  = torch.FloatTensor(a_secuencia(X_test_sc))

# --- Arquitectura RNN ---
# RNN(4 -> hidden 64, 2 capas) + cabeza 50->80->42->48
class RNN_VPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=4, hidden_size=64,
                          num_layers=2, batch_first=True)
        self.cabeza = nn.Sequential(
            nn.Linear(64, 50),  nn.ReLU(),
            nn.Linear(50, 80),  nn.ReLU(),
            nn.Linear(80, 42),  nn.ReLU(),
            nn.Linear(42, 48)
        )
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.cabeza(out[:, -1, :])   # ultimo paso temporal

model     = RNN_VPP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"RNN: (24,4) -> hidden64 -> 50 -> 80 -> 42 -> 48")
print(f"Parametros: {sum(p.numel() for p in model.parameters()):,}\n")

# --- Entrenamiento ---
EPOCHS = 1500
losses = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 150 == 0:
        print(f"  Epoca {epoch:5d}  |  MSE: {loss.item():.6f}")

print(f"\n  Loss final: {losses[-1]:.6f}")

# --- Prediccion ---
model.eval()
with torch.no_grad():
    t0     = time.time()
    y_pred = model(X_test_t).numpy()
    t_ms   = (time.time() - t0) * 1000

print(f"  Inferencia total: {t_ms:.2f} ms  |  por dia: {t_ms/len(X_test):.4f} ms")

# --- CSV de resultados ---
filas = []
for i in range(len(X_test)):
    dia = SPLIT + i
    for h in range(T):
        filas.append({
            "DIA"         : dia,
            "HORA"        : h,
            "PRECIO"      : round(df.loc[dia, f'precio_h{h}'],  4),
            "SOLAR"       : round(df.loc[dia, f'pv_h{h}'],      4),
            "DEMANDA"     : round(df.loc[dia, f'demanda_h{h}'], 4),
            "ACCION_REAL" : round(float(y_test[i, h]),          4),
            "ACCION_IA"   : round(float(y_pred[i, h]),          4),
            "SOC_REAL"    : round(float(y_test[i, T + h]),      4),
            "SOC_IA"      : round(float(y_pred[i, T + h]),      4),
            "DIFERENCIA"  : round(abs(float(y_test[i, T+h]) - float(y_pred[i, T+h])), 4),
        })

df_res = pd.DataFrame(filas)
df_res.to_csv("rnn_resultados.csv", index=False)

# --- Metricas ---
err_acc = df_res["ACCION_REAL"] - df_res["ACCION_IA"]
err_soc = df_res["SOC_REAL"]    - df_res["SOC_IA"]

print("\n" + "="*48)
print("  RESULTADOS RNN")
print("="*48)
print(f"  {'Metrica':<18} {'Accion(MW)':>10}  {'SOC':>10}")
print("-"*48)
print(f"  {'MAE':<18} {err_acc.abs().mean():>10.4f}  {err_soc.abs().mean():>10.4f}")
print(f"  {'RMSE':<18} {np.sqrt((err_acc**2).mean()):>10.4f}  {np.sqrt((err_soc**2).mean()):>10.4f}")
print(f"  {'BIAS':<18} {err_acc.mean():>10.4f}  {err_soc.mean():>10.4f}")
print(f"  {'MAX ERROR':<18} {err_acc.abs().max():>10.4f}  {err_soc.abs().max():>10.4f}")
print("="*48)

# --- Parámetros físicos del sistema (mismos que el optimizador) ---
P_BAT_RANGO = 0.80   # P_dis_max + P_ch_max = 0.50 + 0.30 MW
SOC_RANGO   = 0.70   # SOC_max - SOC_min = 0.90 - 0.20 p.u.
UMBRAL_MAPE = 0.01   # ignorar horas con |P_bat_real| < 1 kW (≈0)
 
# --- Errores base (ya calculados antes de este bloque) ---
# err_acc = df_res["ACCION_REAL"] - df_res["ACCION_IA"]
# err_soc = df_res["SOC_REAL"]    - df_res["SOC_IA"]
 
# --- nMAE normalizado por rango físico ---
# Independiente de la escala, siempre válido aunque P_bat = 0
# Interpretación: % del rango físico máximo de la variable
nMAE_acc = err_acc.abs().mean() / P_BAT_RANGO * 100
nMAE_soc = err_soc.abs().mean() / SOC_RANGO   * 100
 
nRMSE_acc = np.sqrt((err_acc**2).mean()) / P_BAT_RANGO * 100
nRMSE_soc = np.sqrt((err_soc**2).mean()) / SOC_RANGO   * 100
 
# --- MAPE clásico (filtrado: solo horas donde batería se mueve) ---
# Se excluyen las horas con |P_bat_real| < umbral para evitar
# divisiones por cero que distorsionan el resultado
mask_acc = df_res["ACCION_REAL"].abs() > UMBRAL_MAPE
mask_soc = df_res["SOC_REAL"].abs()    > UMBRAL_MAPE
 
n_validos_acc = mask_acc.sum()
n_validos_soc = mask_soc.sum()
n_total       = len(df_res)
 
if n_validos_acc > 0:
    mape_acc = (
        (df_res.loc[mask_acc, "ACCION_REAL"] -
         df_res.loc[mask_acc, "ACCION_IA"]).abs()
        / df_res.loc[mask_acc, "ACCION_REAL"].abs()
    ).mean() * 100
else:
    mape_acc = float('nan')
 
if n_validos_soc > 0:
    mape_soc = (
        (df_res.loc[mask_soc, "SOC_REAL"] -
         df_res.loc[mask_soc, "SOC_IA"]).abs()
        / df_res.loc[mask_soc, "SOC_REAL"].abs()
    ).mean() * 100
else:
    mape_soc = float('nan')
 
# --- Imprimir resultados ---
print("\n" + "="*58)
print("  ERROR PORCENTUAL")
print("="*58)
print(f"  {'Metrica':<28} {'Accion(MW)':>10}  {'SOC':>10}")
print("-"*58)
print(f"  {'nMAE  (% rango fisico)':<28} "
      f"{nMAE_acc:>9.2f}%  {nMAE_soc:>9.2f}%")
print(f"  {'nRMSE (% rango fisico)':<28} "
      f"{nRMSE_acc:>9.2f}%  {nRMSE_soc:>9.2f}%")
print("-"*58)
print(f"  {'MAPE clasico (filtrado)':<28} "
      f"{mape_acc:>9.2f}%  {mape_soc:>9.2f}%")
print(f"  {'  horas validas / total':<28} "
      f"{n_validos_acc:>7}/{n_total}   "
      f"{n_validos_soc:>7}/{n_total}")
print("="*58)
print()
print("  Interpretacion:")
print(f"  - nMAE P_bat: el error medio es {nMAE_acc:.1f}% del rango")
print(f"    fisico de la bateria ({P_BAT_RANGO} MW).")
print(f"  - nMAE SOC  : el error medio es {nMAE_soc:.1f}% del rango")
print(f"    operativo del SOC ({SOC_RANGO} p.u.).")
if not np.isnan(mape_acc):
    print(f"  - MAPE P_bat: {mape_acc:.1f}% de error relativo en las horas")
    print(f"    donde la bateria se mueve (|P_bat| > {UMBRAL_MAPE} MW).")
print("="*58)
 


# --- Graficas ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("RNN - Evaluacion", fontsize=13)

axes[0].plot(losses, color='steelblue', linewidth=0.8)
axes[0].set_title("Curva de perdida"); axes[0].set_xlabel("Epoca"); axes[0].grid(True, alpha=0.4)

dia0 = df_res[df_res["DIA"] == SPLIT]
axes[1].plot(dia0["HORA"], dia0["ACCION_REAL"], label="Optimizador", color='steelblue')
axes[1].plot(dia0["HORA"], dia0["ACCION_IA"],   label="RNN",  color='darkorange', linestyle='--')
axes[1].set_title("Accion bateria (dia 800)"); axes[1].set_xlabel("Hora"); axes[1].legend(); axes[1].grid(True, alpha=0.4)

axes[2].plot(dia0["HORA"], dia0["SOC_REAL"], label="Optimizador", color='steelblue')
axes[2].plot(dia0["HORA"], dia0["SOC_IA"],   label="RNN",  color='darkorange', linestyle='--')
axes[2].set_title("SOC (dia 800)"); axes[2].set_xlabel("Hora"); axes[2].legend(); axes[2].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("rnn_graficas.png", dpi=150)
plt.show()