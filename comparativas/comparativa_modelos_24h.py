# =============================================================
# TFG - COMPARATIVA DE MODELOS IA — HORIZONTE T=24h
# =============================================================
# Autor: James Kagunda Wangari
# Grado en Ingenieria Electrica - Universidad de Malaga
#
# Descripcion:
# ------------
# Lee los 4 ficheros de resultados del horizonte T=24h
# generados por los scripts de entrenamiento:
#   ../MLP/MLP_T24h_resultados.csv
#   ../RNN/RNN_T24h_resultados.csv
#   ../LSTM/LSTM_T24h_resultados.csv
#   ../GRU/GRU_T24h_resultados.csv
#
# Figuras generadas en COMPARATIVAS/:
#   1) comparativa_mae_rmse_T24h.png   — barras MAE/RMSE
#   2) error_por_hora_T24h.png         — MAE por hora del dia
#   3) histograma_errores_T24h.png     — distribucion del error
#   4) prediccion_vs_real_T24h.png     — P_bat y SOC dia ref
#   5) scatter_real_predicho_T24h.png  — scatter real vs pred
#   6) distribucion_coste_T24h.png     — coste optimizador
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

torch_seed = 42

DIR      = os.path.dirname(os.path.abspath(__file__))
DIR_RAIZ = os.path.dirname(DIR)

MODELOS    = ['MLP', 'RNN', 'LSTM', 'GRU']
COLORES    = {'MLP': '#2196F3', 'RNN': '#FF9800',
              'LSTM': '#4CAF50', 'GRU': '#F44336'}
MARCADORES = {'MLP': 'o', 'RNN': 's', 'LSTM': '^', 'GRU': 'D'}

T     = 24
SPLIT = 800   # primer dia de test

# Rutas CSV de resultados (cada modelo en su carpeta)
CSV_MODELOS = {m: os.path.join(DIR_RAIZ, m, f'{m}_T24h_resultados.csv')
               for m in MODELOS}

# Dataset completo del optimizador para estadisticas de coste
CSV_COMPLETO = os.path.join(DIR_RAIZ, 'dataset', 'dataset_vpp_completo.csv')

# =============================================================
# CARGA DE DATOS
# =============================================================

print("Cargando resultados T=24h...\n")
dfs = {}
for modelo, ruta in CSV_MODELOS.items():
    if os.path.exists(ruta):
        dfs[modelo] = pd.read_csv(ruta)
        print(f"  OK  {modelo}: {len(dfs[modelo])} filas")
    else:
        print(f"  --  {modelo}: NO ENCONTRADO ({ruta})")

if not dfs:
    print("\nERROR: No se encontro ningun CSV. Ejecuta primero los scripts de entrenamiento.")
    exit(1)

modelos_ok = [m for m in MODELOS if m in dfs]

df_opt  = None
df_test = None
if os.path.exists(CSV_COMPLETO):
    df_opt  = pd.read_csv(CSV_COMPLETO)
    df_test = df_opt[df_opt['split'] == 'test'].reset_index(drop=True)
    print(f"\n  Dataset completo: {len(df_test)} escenarios de test")

# =============================================================
# CALCULO DE METRICAS
# =============================================================

P_RNG, S_RNG = 0.80, 0.70

metricas = {}
for modelo, df in dfs.items():
    ea = df['ACCION_REAL'] - df['ACCION_IA']
    es = df['SOC_REAL']    - df['SOC_IA']
    mask = df['ACCION_REAL'].abs() > 0.01
    mape = (ea[mask].abs() / df['ACCION_REAL'][mask].abs()).mean() * 100
    metricas[modelo] = {
        'MAE_acc' : ea.abs().mean(),
        'RMSE_acc': np.sqrt((ea**2).mean()),
        'MAE_soc' : es.abs().mean(),
        'RMSE_soc': np.sqrt((es**2).mean()),
        'BIAS_acc': ea.mean(),
        'MAX_acc' : ea.abs().max(),
        'MAPE_acc': mape,
        'nMAE_acc': ea.abs().mean() / P_RNG * 100,
        'nMAE_soc': es.abs().mean() / S_RNG * 100,
    }

# =============================================================
# TABLA COMPARATIVA EN CONSOLA
# =============================================================

print("\n" + "="*85)
print("  TABLA COMPARATIVA — T=24h")
print("="*85)
print(f"  {'Modelo':<6} {'MAE Pbat':>10} {'RMSE Pbat':>10} "
      f"{'MAE SOC':>10} {'RMSE SOC':>10} "
      f"{'nMAE Pbat':>11} {'nMAE SOC':>10} {'MAPE(%)':>9}")
print("-"*85)
for m in modelos_ok:
    mt = metricas[m]
    print(f"  {m:<6} {mt['MAE_acc']:>10.4f} {mt['RMSE_acc']:>10.4f} "
          f"{mt['MAE_soc']:>10.4f} {mt['RMSE_soc']:>10.4f} "
          f"{mt['nMAE_acc']:>10.2f}% {mt['nMAE_soc']:>9.2f}% "
          f"{mt['MAPE_acc']:>9.2f}")
print("="*85)

mejor = min(modelos_ok, key=lambda m: metricas[m]['MAE_acc'])
print(f"\n  Mejor modelo (menor MAE P_bat): {mejor}  "
      f"MAE={metricas[mejor]['MAE_acc']:.4f} MW")

# =============================================================
# FIGURA 1 — BARRAS MAE y RMSE
# =============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Comparativa de metricas - T=24h", fontsize=14, fontweight='bold')

for (clave, titulo), ax in zip(
    [('MAE_acc',  'MAE P_bat (MW)'),
     ('RMSE_acc', 'RMSE P_bat (MW)'),
     ('MAE_soc',  'MAE SOC (p.u.)'),
     ('RMSE_soc', 'RMSE SOC (p.u.)')],
    axes.flatten()):

    valores = [metricas[m][clave] for m in modelos_ok]
    bars = ax.barh(modelos_ok, valores,
                   color=[COLORES[m] for m in modelos_ok],
                   edgecolor='white', height=0.6)
    vmax = max(valores)
    for bar, val in zip(bars, valores):
        ax.text(val + vmax*0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    ax.set_xlabel(titulo)
    ax.set_title(titulo)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(DIR, 'comparativa_mae_rmse_T24h.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n  Guardada: comparativa_mae_rmse_T24h.png")

# =============================================================
# FIGURA 2 — MAE POR HORA DEL DIA
# =============================================================

fig, ax = plt.subplots(figsize=(12, 5))
for modelo, df in dfs.items():
    mae_hora = [(df[df['HORA']==h]['ACCION_REAL'] - df[df['HORA']==h]['ACCION_IA']).abs().mean()
                for h in range(T)]
    ax.plot(range(T), mae_hora, label=modelo,
            color=COLORES[modelo], marker=MARCADORES[modelo],
            linewidth=1.8, markersize=5)

ax.axvspan(7, 9,   alpha=0.08, color='gray')
ax.axvspan(18, 21, alpha=0.08, color='gray')
ax.set_xlabel('Hora del dia', fontsize=11)
ax.set_ylabel('MAE P_bat (MW)', fontsize=11)
ax.set_title('Error MAE de prediccion por hora - T=24h', fontsize=13)
ax.set_xticks(range(T))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'error_por_hora_T24h.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Guardada: error_por_hora_T24h.png")

# =============================================================
# FIGURA 3 — HISTOGRAMA DE ERRORES
# =============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Distribucion del error de prediccion P_bat - T=24h',
             fontsize=13, fontweight='bold')

for idx, modelo in enumerate(modelos_ok[:4]):
    ax  = axes[idx // 2][idx % 2]
    err = (dfs[modelo]['ACCION_REAL'] - dfs[modelo]['ACCION_IA']).values
    ax.hist(err, bins=60, color=COLORES[modelo],
            alpha=0.6, density=True, edgecolor='white')
    kde = gaussian_kde(err)
    xr  = np.linspace(err.min(), err.max(), 300)
    ax.plot(xr, kde(xr), color=COLORES[modelo], linewidth=2)
    ax.axvline(0,          color='black', linewidth=1.2, linestyle='--', label='Error=0')
    ax.axvline(err.mean(), color='red',   linewidth=1,   linestyle=':',
               label=f'Sesgo={err.mean():.4f}')
    ax.set_title(f'{modelo}  (sesgo={err.mean():.4f}, std={err.std():.4f})')
    ax.set_xlabel('Error (MW)'); ax.set_ylabel('Densidad')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(DIR, 'histograma_errores_T24h.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Guardada: histograma_errores_T24h.png")

# =============================================================
# FIGURA 4 — PREDICCION VS REAL (dia de referencia)
# =============================================================

# Primer dia de test disponible
DIA_REF = dfs[modelos_ok[0]]['DIA'].unique()[0]

n_mod = len(modelos_ok)
fig, axes = plt.subplots(2, n_mod, figsize=(4*n_mod, 8))
fig.suptitle(f'Prediccion vs. solucion optima - dia {DIA_REF} - T=24h',
             fontsize=13, fontweight='bold')

for idx, modelo in enumerate(modelos_ok):
    df_dia = dfs[modelo][dfs[modelo]['DIA'] == DIA_REF].sort_values('HORA')

    ax = axes[0][idx]
    ax.plot(df_dia['HORA'], df_dia['ACCION_REAL'],
            label='MPC optimo', color='steelblue', linewidth=2)
    ax.plot(df_dia['HORA'], df_dia['ACCION_IA'],
            label=modelo, color=COLORES[modelo], linewidth=1.5, ls='--')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_title(modelo)
    ax.set_ylabel('P_bat (MW)' if idx == 0 else '')
    ax.set_xlabel('Hora')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1][idx]
    ax.plot(df_dia['HORA'], df_dia['SOC_REAL'],
            label='MPC optimo', color='steelblue', linewidth=2)
    ax.plot(df_dia['HORA'], df_dia['SOC_IA'],
            label=modelo, color=COLORES[modelo], linewidth=1.5, ls='--')
    ax.axhline(0.20, color='red', linewidth=0.8, linestyle=':', label='SOC min/max')
    ax.axhline(0.90, color='red', linewidth=0.8, linestyle=':')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('SOC (p.u.)' if idx == 0 else '')
    ax.set_xlabel('Hora')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DIR, 'prediccion_vs_real_T24h.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Guardada: prediccion_vs_real_T24h.png")

# =============================================================
# FIGURA 5 — SCATTER REAL VS PREDICHO
# =============================================================

fig, axes = plt.subplots(2, n_mod, figsize=(4*n_mod, 8))
fig.suptitle('Scatter real vs. predicho - T=24h', fontsize=13, fontweight='bold')

for idx, modelo in enumerate(modelos_ok):
    df    = dfs[modelo]
    horas = df['HORA'].values

    ax = axes[0][idx]
    ax.scatter(df['ACCION_REAL'], df['ACCION_IA'],
               c=horas, cmap='viridis', alpha=0.3, s=3)
    lim = [df['ACCION_REAL'].min(), df['ACCION_REAL'].max()]
    ax.plot(lim, lim, 'r--', linewidth=1, label='y=x')
    ax.set_title(modelo)
    ax.set_xlabel('P_bat real (MW)' if idx == 0 else '')
    ax.set_ylabel('P_bat predicho (MW)' if idx == 0 else '')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1][idx]
    ax.scatter(df['SOC_REAL'], df['SOC_IA'],
               c=horas, cmap='viridis', alpha=0.3, s=3)
    lim2 = [0.15, 0.95]
    ax.plot(lim2, lim2, 'r--', linewidth=1, label='y=x')
    ax.set_xlabel('SOC real (p.u.)' if idx == 0 else '')
    ax.set_ylabel('SOC predicho (p.u.)' if idx == 0 else '')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DIR, 'scatter_real_predicho_T24h.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Guardada: scatter_real_predicho_T24h.png")

# =============================================================
# FIGURA 6 — DISTRIBUCION COSTE OPTIMIZADOR
# =============================================================

if df_test is not None:
    costes_dia = []
    for _, row in df_test.iterrows():
        c = 0
        for h in range(T):
            p_red = row[f'p_red_h{h}']
            lam   = row[f'precio_h{h}']
            p_dg  = row[f'p_dg_h{h}']
            p_ch  = max(0,  row[f'p_bat_h{h}'])
            p_dis = max(0, -row[f'p_bat_h{h}'])
            c += (max(0,  p_red) * lam
                - max(0, -p_red) * lam * 0.80
                + 10.0 * p_dg
                + 2.0  * (p_ch + p_dis))
        costes_dia.append(c)
    costes_dia = np.array(costes_dia)

    print(f"\n  Estadisticas optimizador T=24h ({len(costes_dia)} escenarios test):")
    print(f"  Media   : {costes_dia.mean():.2f} euro/dia")
    print(f"  Mediana : {np.median(costes_dia):.2f} euro/dia")
    print(f"  Std     : {costes_dia.std():.2f} euro/dia")
    print(f"  Min/Max : {costes_dia.min():.2f} / {costes_dia.max():.2f} euro/dia")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(costes_dia, bins=30, color='steelblue',
            alpha=0.7, edgecolor='white', density=False)
    kde = gaussian_kde(costes_dia)
    xr  = np.linspace(costes_dia.min(), costes_dia.max(), 300)
    ax2 = ax.twinx()
    ax2.plot(xr, kde(xr), color='steelblue', linewidth=2)
    ax2.set_ylabel('Densidad KDE')
    ax.axvline(costes_dia.mean(),    color='red',   linestyle='--', linewidth=1.5,
               label=f'Media = {costes_dia.mean():.1f} euro/dia')
    ax.axvline(np.median(costes_dia), color='green', linestyle='--', linewidth=1.5,
               label=f'Mediana = {np.median(costes_dia):.1f} euro/dia')
    ax.set_xlabel('Coste optimo J* (euro/dia)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribucion del coste optimo - optimizador T=24h')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR, 'distribucion_coste_T24h.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Guardada: distribucion_coste_T24h.png")

# =============================================================
# RESUMEN FINAL
# =============================================================

print("\n" + "="*60)
print("  COMPARATIVA T=24h COMPLETADA")
print("="*60)
print("  Figuras en: COMPARATIVAS/")
print("    1) comparativa_mae_rmse_T24h.png")
print("    2) error_por_hora_T24h.png")
print("    3) histograma_errores_T24h.png")
print("    4) prediccion_vs_real_T24h.png")
print("    5) scatter_real_predicho_T24h.png")
print("    6) distribucion_coste_T24h.png")
print("="*60)