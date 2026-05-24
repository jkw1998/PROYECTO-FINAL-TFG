# =============================================================
# TFG - COMPARATIVA DE MODELOS IA — HORIZONTE T=24h
# =============================================================
# Autor: James Kagunda Wangari
# Grado en Ingenieria Electrica - Universidad de Malaga
#
# Descripcion:
# ------------
# Lee los 4 archivos de resultados del horizonte T=24h
# (mlp_resultados.csv, rnn_resultados.csv,
#  lstm_resultados.csv, gru_resultados.csv)
# y genera todas las tablas y figuras del Capitulo 7 para
# la comparativa entre modelos.
#
# Figuras generadas (todas en el mismo directorio que el script):
# ---------------------------------------------------------------
#  1) comparativa_mae_rmse_T24h.png
#     Barras agrupadas: MAE y RMSE de P_bat y SOC por modelo
#
#  2) error_por_hora_T24h.png
#     MAE de P_bat por hora del dia (0-23) para cada modelo
#
#  3) histograma_errores_T24h.png
#     Distribucion del error e = P_bat_real - P_bat_pred (4 subplots)
#
#  4) prediccion_vs_real_T24h.png
#     P_bat y SOC predicho vs real para un dia del test (8 subplots)
#
#  5) scatter_real_predicho_T24h.png
#     Scatter real vs predicho para P_bat y SOC (8 subplots)
#
#  6) distribucion_coste_T24h.png
#     Histograma del coste optimo J* del optimizador (200 escenarios)
#
# Tablas impresas en consola (para copiar a la memoria LaTeX):
# ------------------------------------------------------------
#  - Tabla comparativa MAE, RMSE, BIAS, MAX_ERROR, t_inf
#  - Tabla de estadisticas del optimizador (coste, P_bat, etc.)
#
# REQUISITO: ejecutar primero los 4 scripts de entrenamiento IA
# para T=24h y el optimizador T=24h para tener los CSV.
# Los CSV deben estar en el mismo directorio que este script
# O se puede ajustar DIR_DATOS al directorio donde esten.
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

# =============================================================
# CONFIGURACION DE RUTAS
# =============================================================

# Directorio de este script (comparativas/)
DIR = os.path.dirname(os.path.abspath(__file__))

# Directorio raiz del proyecto (un nivel arriba de comparativas/)
DIR_RAIZ = os.path.dirname(DIR)

# Las figuras se guardan en comparativas/ (junto a este script)
# Los CSV de resultados T=24h estan en la carpeta raiz
DIR_DATOS = DIR_RAIZ

MODELOS    = ['MLP', 'RNN', 'LSTM', 'GRU']
COLORES    = {'MLP': '#2196F3', 'RNN': '#FF9800',
              'LSTM': '#4CAF50', 'GRU': '#F44336'}
MARCADORES = {'MLP': 'o', 'RNN': 's', 'LSTM': '^', 'GRU': 'D'}

CSV_MODELOS = {
    'MLP' : os.path.join(DIR_DATOS, 'mlp_resultados.csv'),
    'RNN' : os.path.join(DIR_DATOS, 'rnn_resultados.csv'),
    'LSTM': os.path.join(DIR_DATOS, 'lstm_resultados.csv'),
    'GRU' : os.path.join(DIR_DATOS, 'gru_resultados.csv'),
}

CSV_COMPLETO = os.path.join(DIR_DATOS, 'dataset_vpp_completo.csv')

T     = 24
SPLIT = 800

# =============================================================
# CARGA DE DATOS
# =============================================================

print("Cargando resultados T=24h...")
dfs = {}
for modelo, ruta in CSV_MODELOS.items():
    if os.path.exists(ruta):
        dfs[modelo] = pd.read_csv(ruta)
        print(f"  {modelo}: {len(dfs[modelo])} filas cargadas")
    else:
        print(f"  [AVISO] No encontrado: {ruta}")

if not dfs:
    print("\nERROR: No se encontro ningun CSV de resultados.")
    print("Ejecuta primero los scripts de entrenamiento IA para T=24h.")
    exit(1)

modelos_ok = list(dfs.keys())
print(f"  Modelos disponibles: {modelos_ok}\n")

# Dataset completo del optimizador (para estadisticas de coste)
df_opt = None
if os.path.exists(CSV_COMPLETO):
    df_opt = pd.read_csv(CSV_COMPLETO)
    df_test = df_opt[df_opt['split'] == 'test'].reset_index(drop=True)
    print(f"  Dataset completo: {len(df_test)} escenarios de test")

# =============================================================
# CALCULO DE METRICAS
# =============================================================

metricas = {}
for modelo, df in dfs.items():
    err_acc = df['ACCION_REAL'] - df['ACCION_IA']
    err_soc = df['SOC_REAL']    - df['SOC_IA']

    # Evitar division por cero en MAPE
    mask = df['ACCION_REAL'].abs() > 0.01
    mape = (err_acc[mask].abs() / df['ACCION_REAL'][mask].abs()).mean() * 100

    metricas[modelo] = {
        'MAE_acc'  : err_acc.abs().mean(),
        'RMSE_acc' : np.sqrt((err_acc**2).mean()),
        'MAE_soc'  : err_soc.abs().mean(),
        'RMSE_soc' : np.sqrt((err_soc**2).mean()),
        'BIAS_acc' : err_acc.mean(),
        'MAX_acc'  : err_acc.abs().max(),
        'MAPE_acc' : mape,
    }

# =============================================================
# TABLA COMPARATIVA EN CONSOLA
# =============================================================

print("\n" + "="*80)
print("  TABLA COMPARATIVA — T=24h (para copiar a LaTeX)")
print("="*80)
print(f"  {'Modelo':<6} {'MAE_Pbat':>10} {'RMSE_Pbat':>10} "
      f"{'MAE_SOC':>10} {'RMSE_SOC':>10} {'MAPE(%)':>9} {'t_inf':>8}")
print("-"*80)
for m in modelos_ok:
    mt = metricas[m]
    print(f"  {m:<6} {mt['MAE_acc']:>10.4f} {mt['RMSE_acc']:>10.4f} "
          f"{mt['MAE_soc']:>10.4f} {mt['RMSE_soc']:>10.4f} "
          f"{mt['MAPE_acc']:>9.2f} {'<1 ms':>8}")
print("="*80)

# Estadisticas del optimizador
if df_opt is not None:
    costes = []
    for _, row in df_test.iterrows():
        c = sum(row[f'p_red_h{h}'] * row[f'precio_h{h}'] * (
                1 if row[f'p_red_h{h}'] >= 0 else -0.8)
                for h in range(T))
        costes.append(c)
    costes = np.array(costes)
    print(f"\n  Estadisticas optimizador T=24h (200 escenarios test):")
    print(f"  Media coste  : {costes.mean():>8.2f} €/dia")
    print(f"  Mediana coste: {np.median(costes):>8.2f} €/dia")
    print(f"  Desv. tipica : {costes.std():>8.2f} €/dia")
    print(f"  Min / Max    : {costes.min():>8.2f} / {costes.max():>8.2f} €/dia\n")

# =============================================================
# FIGURA 1 — COMPARATIVA MAE Y RMSE (barras agrupadas)
# =============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Comparativa de métricas — T=24h", fontsize=14, fontweight='bold')

metricas_plot = [
    ('MAE_acc',  'MAE $P_{bat}$ (MW)',   axes[0, 0]),
    ('RMSE_acc', 'RMSE $P_{bat}$ (MW)',  axes[0, 1]),
    ('MAE_soc',  'MAE SOC (p.u.)',       axes[1, 0]),
    ('RMSE_soc', 'RMSE SOC (p.u.)',      axes[1, 1]),
]

for clave, titulo, ax in metricas_plot:
    valores = [metricas[m][clave] for m in modelos_ok]
    bars = ax.barh(modelos_ok, valores,
                   color=[COLORES[m] for m in modelos_ok],
                   edgecolor='white', height=0.6)
    for bar, val in zip(bars, valores):
        ax.text(val + max(valores)*0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    ax.set_xlabel(titulo)
    ax.set_title(titulo)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
ruta = os.path.join(DIR, 'comparativa_mae_rmse_T24h.png')
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardada: comparativa_mae_rmse_T24h.png")

# =============================================================
# FIGURA 2 — ERROR MAE POR HORA DEL DIA
# =============================================================

fig, ax = plt.subplots(figsize=(12, 5))

for modelo, df in dfs.items():
    mae_hora = []
    for h in range(T):
        df_h  = df[df['HORA'] == h]
        err_h = (df_h['ACCION_REAL'] - df_h['ACCION_IA']).abs().mean()
        mae_hora.append(err_h)
    ax.plot(range(T), mae_hora,
            label=modelo, color=COLORES[modelo],
            marker=MARCADORES[modelo], linewidth=1.8,
            markersize=5)

# Zonas de transicion de precio
ax.axvspan(7, 9,  alpha=0.08, color='gray', label='Transicion precio')
ax.axvspan(18, 21, alpha=0.08, color='gray')

ax.set_xlabel('Hora del día', fontsize=11)
ax.set_ylabel('MAE $P_{bat}$ (MW)', fontsize=11)
ax.set_title('Error MAE de predicción por hora — T=24h', fontsize=13)
ax.set_xticks(range(T))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
ruta = os.path.join(DIR, 'error_por_hora_T24h.png')
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardada: error_por_hora_T24h.png")

# =============================================================
# FIGURA 3 — HISTOGRAMA DE ERRORES (distribucion)
# =============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Distribución del error de predicción $P_{bat}$ — T=24h',
             fontsize=13, fontweight='bold')

for idx, modelo in enumerate(modelos_ok[:4]):
    ax  = axes[idx // 2][idx % 2]
    df  = dfs[modelo]
    err = (df['ACCION_REAL'] - df['ACCION_IA']).values

    ax.hist(err, bins=60, color=COLORES[modelo],
            alpha=0.6, density=True, edgecolor='white')

    # Curva KDE
    kde  = gaussian_kde(err)
    x_r  = np.linspace(err.min(), err.max(), 300)
    ax.plot(x_r, kde(x_r), color=COLORES[modelo], linewidth=2)

    ax.axvline(0, color='black', linewidth=1.2, linestyle='--',
               label='Error = 0')
    ax.axvline(err.mean(), color='red', linewidth=1,
               linestyle=':', label=f'Sesgo={err.mean():.4f}')

    ax.set_title(f'{modelo}  (sesgo={err.mean():.4f}, '
                 f'std={err.std():.4f})')
    ax.set_xlabel('Error (MW)')
    ax.set_ylabel('Densidad')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
ruta = os.path.join(DIR, 'histograma_errores_T24h.png')
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardada: histograma_errores_T24h.png")

# =============================================================
# FIGURA 4 — PREDICCION VS REAL (dia de referencia: dia 800)
# =============================================================

DIA_REF = SPLIT   # primer dia del test

n_mod   = len(modelos_ok)
fig, axes = plt.subplots(2, n_mod, figsize=(4*n_mod, 8))
fig.suptitle(f'Predicción vs. solución óptima — día {DIA_REF} — T=24h',
             fontsize=13, fontweight='bold')

for idx, modelo in enumerate(modelos_ok):
    df_dia = dfs[modelo][dfs[modelo]['DIA'] == DIA_REF]

    # Fila 1: P_bat
    ax = axes[0][idx]
    ax.plot(df_dia['HORA'], df_dia['ACCION_REAL'],
            label='MPC óptimo', color='steelblue', linewidth=2)
    ax.plot(df_dia['HORA'], df_dia['ACCION_IA'],
            label=modelo, color=COLORES[modelo],
            linewidth=1.5, linestyle='--')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_title(f'{modelo}')
    ax.set_ylabel('$P_{bat}$ (MW)' if idx == 0 else '')
    ax.set_xlabel('Hora')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Fila 2: SOC
    ax = axes[1][idx]
    ax.plot(df_dia['HORA'], df_dia['SOC_REAL'],
            label='MPC óptimo', color='steelblue', linewidth=2)
    ax.plot(df_dia['HORA'], df_dia['SOC_IA'],
            label=modelo, color=COLORES[modelo],
            linewidth=1.5, linestyle='--')
    ax.axhline(0.20, color='red', linewidth=0.8,
               linestyle=':', label='SOC min/max')
    ax.axhline(0.90, color='red', linewidth=0.8, linestyle=':')
    ax.set_ylim(0, 1)
    ax.set_ylabel('SOC (p.u.)' if idx == 0 else '')
    ax.set_xlabel('Hora')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
ruta = os.path.join(DIR, 'prediccion_vs_real_T24h.png')
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardada: prediccion_vs_real_T24h.png")

# =============================================================
# FIGURA 5 — SCATTER REAL VS PREDICHO
# =============================================================

fig, axes = plt.subplots(2, n_mod, figsize=(4*n_mod, 8))
fig.suptitle('Scatter real vs. predicho — T=24h', fontsize=13, fontweight='bold')

for idx, modelo in enumerate(modelos_ok):
    df = dfs[modelo]
    horas = df['HORA'].values   # para colorear por hora

    # Fila 1: P_bat
    ax = axes[0][idx]
    sc = ax.scatter(df['ACCION_REAL'], df['ACCION_IA'],
                    c=horas, cmap='viridis',
                    alpha=0.3, s=3)
    lim = [df['ACCION_REAL'].min(), df['ACCION_REAL'].max()]
    ax.plot(lim, lim, 'r--', linewidth=1, label='y=x')
    ax.set_title(f'{modelo}')
    ax.set_xlabel('$P_{bat}$ real (MW)' if idx == 0 else '')
    ax.set_ylabel('$P_{bat}$ predicho (MW)' if idx == 0 else '')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Fila 2: SOC
    ax = axes[1][idx]
    ax.scatter(df['SOC_REAL'], df['SOC_IA'],
               c=horas, cmap='viridis',
               alpha=0.3, s=3)
    lim = [0.15, 0.95]
    ax.plot(lim, lim, 'r--', linewidth=1, label='y=x')
    ax.set_xlabel('SOC real (p.u.)' if idx == 0 else '')
    ax.set_ylabel('SOC predicho (p.u.)' if idx == 0 else '')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
ruta = os.path.join(DIR, 'scatter_real_predicho_T24h.png')
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardada: scatter_real_predicho_T24h.png")

# =============================================================
# FIGURA 6 — DISTRIBUCION COSTE OPTIMIZADOR
# =============================================================

if df_opt is not None:
    # Calcular coste por escenario desde el dataset completo
    costes_dia = []
    for _, row in df_test.iterrows():
        c = 0
        for h in range(T):
            p_red = row[f'p_red_h{h}']
            lam   = row[f'precio_h{h}']
            p_dg  = row[f'p_dg_h{h}']
            p_ch  = max(0,  row[f'p_bat_h{h}'])
            p_dis = max(0, -row[f'p_bat_h{h}'])
            c += (max(0, p_red) * lam
                  - max(0, -p_red) * lam * 0.80
                  + 10.0 * p_dg
                  + 2.0  * (p_ch + p_dis))
        costes_dia.append(c)
    costes_dia = np.array(costes_dia)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(costes_dia, bins=30, color='steelblue',
            alpha=0.7, edgecolor='white', density=False)
    kde  = gaussian_kde(costes_dia)
    x_r  = np.linspace(costes_dia.min(), costes_dia.max(), 300)
    ax2  = ax.twinx()
    ax2.plot(x_r, kde(x_r), color='steelblue', linewidth=2)
    ax2.set_ylabel('Densidad KDE')
    ax.axvline(costes_dia.mean(), color='red', linestyle='--',
               linewidth=1.5, label=f'Media = {costes_dia.mean():.1f} €/día')
    ax.axvline(np.median(costes_dia), color='green', linestyle='--',
               linewidth=1.5, label=f'Mediana = {np.median(costes_dia):.1f} €/día')
    ax.set_xlabel('Coste óptimo $\\mathcal{J}^*$ (€/día)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución del coste óptimo — optimizador T=24h')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(DIR, 'distribucion_coste_T24h.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardada: distribucion_coste_T24h.png")

# =============================================================
# RESUMEN FINAL
# =============================================================

print("\n" + "="*60)
print("  COMPARATIVA T=24h COMPLETADA")
print("="*60)
print("  Figuras generadas:")
print("    1) comparativa_mae_rmse_T24h.png")
print("    2) error_por_hora_T24h.png")
print("    3) histograma_errores_T24h.png")
print("    4) prediccion_vs_real_T24h.png")
print("    5) scatter_real_predicho_T24h.png")
if df_opt is not None:
    print("    6) distribucion_coste_T24h.png")
print("="*60)