# =============================================================
# TFG - GRU | HORIZONTE T=24h
# =============================================================
# Autor: James Kagunda Wangari
# Grado en Ingenieria Electrica - Universidad de Malaga
#
# Arquitectura: GRU (Gated Recurrent Unit)
# Horizonte:    T=24h — ve el dia completo como secuencia
#
# Entrada: secuencia (batch, seq_len=24, input_size=4)
#   cada paso t: [precio(t), pv(t), demanda(t), soc_inicial]
#   soc_inicial se repite en todos los pasos como contexto
# Salida (48 outputs): p_bat*24 + SOC*24
#
# GRU usa puertas de reset y actualizacion (sin celda separada),
# lo que lo hace mas eficiente que LSTM con resultados similares.
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

df = pd.read_csv(DS_PATH)
df.columns = df.columns.str.strip()
T = 24

filas_X, filas_y = [], []
for _, row in df.iterrows():
    soc0 = float(row['soc_inicial'])
    seq  = [[float(row[f'precio_h{t}']), float(row[f'pv_h{t}']),
              float(row[f'demanda_h{t}']), soc0] for t in range(T)]
    filas_X.append(seq)
    filas_y.append([float(row[f'p_bat_h{t}']) for t in range(T)]
                 + [float(row[f'soc_h{t}'])   for t in range(T)])

X_all = np.array(filas_X, dtype=np.float32)   # (1000, 24, 4)
y_all = np.array(filas_y, dtype=np.float32)   # (1000, 48)

mask_tr = df['split'] == 'train'
mask_te = df['split'] == 'test'
X_train, y_train = X_all[mask_tr.values], y_all[mask_tr.values]
X_test,  y_test  = X_all[mask_te.values], y_all[mask_te.values]
dias_test = df[mask_te]['dia_id'].values

print(f"GRU T=24h | Train: {len(X_train)} dias  Test: {len(X_test)} dias  Shape: {X_train.shape}")

N_tr, S, F = X_train.shape
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N_tr, S, F)
N_te = X_test.shape[0]
X_test_sc  = scaler.transform(X_test.reshape(-1, F)).reshape(N_te, S, F)

X_train_t = torch.FloatTensor(X_train_sc)
X_test_t  = torch.FloatTensor(X_test_sc)
y_train_t = torch.FloatTensor(y_train)

class GRUModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=48):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Sequential(nn.Linear(hidden_size, 256), nn.ReLU(),
                                 nn.Linear(256, output_size))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

model     = GRUModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(f"Parametros: {sum(p.numel() for p in model.parameters()):,}\n")

EPOCHS, losses = 1500, []
for epoch in range(EPOCHS):
    model.train(); optimizer.zero_grad()
    loss = criterion(model(X_train_t), y_train_t)
    loss.backward(); optimizer.step()
    losses.append(loss.item())
    if epoch % 150 == 0:
        print(f"  Epoca {epoch:5d}  |  MSE: {loss.item():.6f}")
print(f"\n  Loss final: {losses[-1]:.6f}")

model.eval()
with torch.no_grad():
    t0 = time.time()
    y_pred = model(X_test_t).numpy()
    t_ms   = (time.time() - t0) * 1000
print(f"  Inferencia: {t_ms:.2f} ms total  |  {t_ms/len(X_test):.4f} ms/dia")

filas = []
for i, dia_id in enumerate(dias_test):
    for h in range(T):
        filas.append({
            'DIA': dia_id, 'HORA': h, 'HORIZONTE': 'T=24h', 'MODELO': 'GRU',
            'ACCION_REAL': round(float(y_test[i, h]),   4),
            'ACCION_IA':   round(float(y_pred[i, h]),   4),
            'SOC_REAL':    round(float(y_test[i, T+h]), 4),
            'SOC_IA':      round(float(y_pred[i, T+h]), 4),
        })
df_res = pd.DataFrame(filas)
df_res.to_csv(os.path.join(DIR, 'GRU_T24h_resultados.csv'), index=False)

P_RNG, S_RNG = 0.80, 0.70
ea = df_res['ACCION_REAL'] - df_res['ACCION_IA']
es = df_res['SOC_REAL']    - df_res['SOC_IA']
mae_a, mae_s   = ea.abs().mean(), es.abs().mean()
rmse_a, rmse_s = np.sqrt((ea**2).mean()), np.sqrt((es**2).mean())

print(f"\n{'='*55}\n  METRICAS GRU - T=24h\n{'='*55}")
print(f"  {'MAE':<20} {mae_a:>10.4f}  {mae_s:>10.4f}")
print(f"  {'RMSE':<20} {rmse_a:>10.4f}  {rmse_s:>10.4f}")
print(f"  {'nMAE (%rango)':<20} {mae_a/P_RNG*100:>9.2f}%  {mae_s/S_RNG*100:>9.2f}%")
print(f"  {'nRMSE(%rango)':<20} {rmse_a/P_RNG*100:>9.2f}%  {rmse_s/S_RNG*100:>9.2f}%")

pd.DataFrame([{
    'MODELO':'GRU','HORIZONTE':'T=24h',
    'MAE_ACC':round(mae_a,4), 'RMSE_ACC':round(rmse_a,4),
    'MAE_SOC':round(mae_s,4), 'RMSE_SOC':round(rmse_s,4),
    'nMAE_ACC':round(mae_a/P_RNG*100,2), 'nRMSE_ACC':round(rmse_a/P_RNG*100,2),
    'nMAE_SOC':round(mae_s/S_RNG*100,2), 'nRMSE_SOC':round(rmse_s/S_RNG*100,2),
    'BIAS_ACC':round(ea.mean(),4), 'BIAS_SOC':round(es.mean(),4),
    'MAX_ERR_ACC':round(ea.abs().max(),4), 'MAX_ERR_SOC':round(es.abs().max(),4),
    'T_INF_MS':round(t_ms/len(X_test),4),
}]).to_csv(os.path.join(DIR, 'GRU_T24h_metricas.csv'), index=False)

DIA_VIS = dias_test[0]
d0 = df_res[df_res['DIA']==DIA_VIS].sort_values('HORA')
fig, axes = plt.subplots(1, 3, figsize=(15,4))
fig.suptitle('GRU - T=24h', fontsize=13)
axes[0].plot(losses, color='steelblue', lw=0.8)
axes[0].set_title('Curva de perdida'); axes[0].set_xlabel('Epoca'); axes[0].set_ylabel('MSE'); axes[0].grid(True, alpha=0.4)
axes[1].plot(d0['HORA'], d0['ACCION_REAL'], label='Optimizador', color='steelblue')
axes[1].plot(d0['HORA'], d0['ACCION_IA'], label='GRU', color='darkorange', ls='--')
axes[1].set_title(f'Accion bateria (dia {DIA_VIS}) - T=24h'); axes[1].set_xlabel('Hora'); axes[1].set_ylabel('P_bat (MW)'); axes[1].legend(); axes[1].grid(True, alpha=0.4)
axes[2].plot(d0['HORA'], d0['SOC_REAL'], label='Optimizador', color='steelblue')
axes[2].plot(d0['HORA'], d0['SOC_IA'], label='GRU', color='darkorange', ls='--')
axes[2].set_title(f'SOC (dia {DIA_VIS}) - T=24h'); axes[2].set_xlabel('Hora'); axes[2].set_ylabel('SOC (p.u.)'); axes[2].legend(); axes[2].grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'GRU_T24h_graficas.png'), dpi=150)
plt.close()
print(f"\n  Archivos guardados en: {DIR}")
print("  *** GRU T=24h COMPLETADO ***\n")