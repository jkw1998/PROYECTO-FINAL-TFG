# =============================================================
# TFG - MLP | HORIZONTE T=24h
# =============================================================
# Autor: James Kagunda Wangari
# Grado en Ingenieria Electrica - Universidad de Malaga
#
# Arquitectura: MLP (red feedforward, vector plano)
# Horizonte:    T=24h — la IA ve el dia completo de golpe
#
# Entrada (73 features):
#   soc_inicial, precio*24, pv*24, demanda*24
# Salida (48 outputs):
#   p_bat*24, SOC*24  — las 24 horas completas
#
# Es el horizonte de maxima informacion disponible.
# Referencia contra la que se comparan T=1h, 2h y 4h.
# =============================================================

import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

DIR     = os.path.dirname(os.path.abspath(__file__))
DS_PATH = os.path.join(DIR, '..', 'dataset', 'dataset_vpp_ia.csv')

# =============================================================
# CARGA DEL DATASET — T=24h (una fila = un día completo)
# =============================================================

df = pd.read_csv(DS_PATH)
df.columns = df.columns.str.strip()
T = 24

cols_X = (['soc_inicial']
          + [f'precio_h{i}'  for i in range(T)]
          + [f'pv_h{i}'      for i in range(T)]
          + [f'demanda_h{i}' for i in range(T)])
cols_y = ([f'p_bat_h{i}' for i in range(T)]
          + [f'soc_h{i}'  for i in range(T)])

X_all = df[cols_X].values.astype(np.float32)
y_all = df[cols_y].values.astype(np.float32)

mask_tr = df['split'] == 'train'
mask_te = df['split'] == 'test'
X_train, y_train = X_all[mask_tr.values], y_all[mask_tr.values]
X_test,  y_test  = X_all[mask_te.values], y_all[mask_te.values]
dias_test = df[mask_te]['dia_id'].values

print(f"MLP T=24h | Train: {len(X_train)} dias  Test: {len(X_test)} dias  Entrada: {X_train.shape[1]} features")

# =============================================================
# NORMALIZACIÓN
# =============================================================

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
X_train_t  = torch.FloatTensor(X_train_sc)
X_test_t   = torch.FloatTensor(X_test_sc)
y_train_t  = torch.FloatTensor(y_train)

# =============================================================
# ARQUITECTURA MLP  73 → 128 → 256 → 128 → 48
# =============================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(73,  128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128,  48)
        )
    def forward(self, x): return self.net(x)

model     = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(f"Parametros: {sum(p.numel() for p in model.parameters()):,}\n")

# =============================================================
# ENTRENAMIENTO
# =============================================================

EPOCHS, losses = 1500, []
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train_t), y_train_t)
    loss.backward(); optimizer.step()
    losses.append(loss.item())
    if epoch % 150 == 0:
        print(f"  Epoca {epoch:5d}  |  MSE: {loss.item():.6f}")
print(f"\n  Loss final: {losses[-1]:.6f}")

# =============================================================
# PREDICCIÓN
# =============================================================

model.eval()
with torch.no_grad():
    t0 = time.time()
    y_pred = model(X_test_t).numpy()
    t_ms   = (time.time() - t0) * 1000
print(f"  Inferencia: {t_ms:.2f} ms total  |  {t_ms/len(X_test):.4f} ms/dia")

# =============================================================
# CSV RESULTADOS (expandido a nivel hora para comparativa)
# =============================================================

filas = []
for i, dia_id in enumerate(dias_test):
    for h in range(T):
        filas.append({
            'DIA': dia_id, 'HORA': h,
            'HORIZONTE': 'T=24h', 'MODELO': 'MLP',
            'ACCION_REAL': round(float(y_test[i, h]),     4),
            'ACCION_IA':   round(float(y_pred[i, h]),     4),
            'SOC_REAL':    round(float(y_test[i, T+h]),   4),
            'SOC_IA':      round(float(y_pred[i, T+h]),   4),
        })
df_res = pd.DataFrame(filas)
df_res.to_csv(os.path.join(DIR, 'MLP_T24h_resultados.csv'), index=False)

# =============================================================
# MÉTRICAS
# =============================================================

P_RNG, S_RNG = 0.80, 0.70
ea = df_res['ACCION_REAL'] - df_res['ACCION_IA']
es = df_res['SOC_REAL']    - df_res['SOC_IA']

mae_a, mae_s   = ea.abs().mean(), es.abs().mean()
rmse_a, rmse_s = np.sqrt((ea**2).mean()), np.sqrt((es**2).mean())

print(f"\n{'='*55}\n  MÉTRICAS MLP — T=24h\n{'='*55}")
print(f"  {'Metrica':<20} {'P_bat':>10}  {'SOC':>10}")
print(f"  {'MAE':<20} {mae_a:>10.4f}  {mae_s:>10.4f}")
print(f"  {'RMSE':<20} {rmse_a:>10.4f}  {rmse_s:>10.4f}")
print(f"  {'nMAE (%rango)':<20} {mae_a/P_RNG*100:>9.2f}%  {mae_s/S_RNG*100:>9.2f}%")
print(f"  {'nRMSE(%rango)':<20} {rmse_a/P_RNG*100:>9.2f}%  {rmse_s/S_RNG*100:>9.2f}%")
print(f"{'='*55}")

pd.DataFrame([{
    'MODELO':'MLP','HORIZONTE':'T=24h',
    'MAE_ACC':round(mae_a,4),   'RMSE_ACC':round(rmse_a,4),
    'MAE_SOC':round(mae_s,4),   'RMSE_SOC':round(rmse_s,4),
    'nMAE_ACC':round(mae_a/P_RNG*100,2), 'nRMSE_ACC':round(rmse_a/P_RNG*100,2),
    'nMAE_SOC':round(mae_s/S_RNG*100,2), 'nRMSE_SOC':round(rmse_s/S_RNG*100,2),
    'BIAS_ACC':round(ea.mean(),4), 'BIAS_SOC':round(es.mean(),4),
    'MAX_ERR_ACC':round(ea.abs().max(),4), 'MAX_ERR_SOC':round(es.abs().max(),4),
    'T_INF_MS':round(t_ms/len(X_test),4),
}]).to_csv(os.path.join(DIR, 'MLP_T24h_metricas.csv'), index=False)

# =============================================================
# GRÁFICAS
# =============================================================

DIA_VIS = dias_test[0]
d0 = df_res[df_res['DIA']==DIA_VIS].sort_values('HORA')

fig, axes = plt.subplots(1, 3, figsize=(15,4))
fig.suptitle('MLP — T=24h', fontsize=13)

axes[0].plot(losses, color='steelblue', lw=0.8)
axes[0].set_title('Curva de perdida'); axes[0].set_xlabel('Epoca')
axes[0].set_ylabel('MSE'); axes[0].grid(True, alpha=0.4)

axes[1].plot(d0['HORA'], d0['ACCION_REAL'], label='Optimizador', color='steelblue')
axes[1].plot(d0['HORA'], d0['ACCION_IA'],   label='MLP', color='darkorange', ls='--')
axes[1].set_title(f'Accion bateria (dia {DIA_VIS}) — T=24h')
axes[1].set_xlabel('Hora'); axes[1].set_ylabel('P_bat (MW)')
axes[1].legend(); axes[1].grid(True, alpha=0.4)

axes[2].plot(d0['HORA'], d0['SOC_REAL'], label='Optimizador', color='steelblue')
axes[2].plot(d0['HORA'], d0['SOC_IA'],   label='MLP', color='darkorange', ls='--')
axes[2].set_title(f'SOC (dia {DIA_VIS}) — T=24h')
axes[2].set_xlabel('Hora'); axes[2].set_ylabel('SOC (p.u.)')
axes[2].legend(); axes[2].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(DIR, 'MLP_T24h_graficas.png'), dpi=150)
plt.close()
print(f"\n  Archivos guardados en: {DIR}")
print("  *** MLP T=24h COMPLETADO ***\n")