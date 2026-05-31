# =============================================================
# TFG - COMPARATIVA CRUZADA HORIZONTES x MODELOS
# =============================================================
# Autor: James Kagunda Wangari
# Grado en Ingenieria Electrica - Universidad de Malaga
#
# Descripcion:
# ------------
# Lee los 16 ficheros de resultados (4 modelos x 4 horizontes)
# generados por los scripts de entrenamiento y produce las
# figuras y tablas del analisis comparativo del Cap. 7.
#
# Estructura esperada (relativa a este script en COMPARATIVAS/):
#   ../MLP/MLP_T1h_resultados.csv
#   ../MLP/MLP_T2h_resultados.csv  ... etc
#   ../RNN/RNN_T1h_resultados.csv  ... etc
#   ../LSTM/LSTM_T1h_resultados.csv ... etc
#   ../GRU/GRU_T1h_resultados.csv  ... etc
#   ../dataset/dataset_vpp_completo.csv
#
# Figuras generadas en COMPARATIVAS/:
#   1) matriz_MAE_Pbat.png
#   2) matriz_MAE_SOC.png
#   3) efecto_horizonte_MAE.png
#   4) efecto_horizonte_RMSE.png
#   5) barras_comparativa_completa.png
#   6) error_hora_todos_horizontes.png
#   7) tabla_comparativa_visual.png
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR      = os.path.dirname(os.path.abspath(__file__))
DIR_RAIZ = os.path.dirname(DIR)

MODELOS    = ['MLP', 'RNN', 'LSTM', 'GRU']
HORIZONTES = ['T=1h', 'T=2h', 'T=4h', 'T=24h']
H_VALS     = [1, 2, 4, 24]
T          = 24

COLORES_MOD = {'MLP': '#2196F3', 'RNN': '#FF9800',
               'LSTM': '#4CAF50', 'GRU': '#F44336'}
COLORES_HOR = {'T=1h': '#E91E63', 'T=2h': '#9C27B0',
               'T=4h': '#FF9800', 'T=24h': '#2196F3'}
MARCADORES  = {'MLP': 'o', 'RNN': 's', 'LSTM': '^', 'GRU': 'D'}

P_RNG, S_RNG = 0.80, 0.70

# Sufijos de los ficheros CSV por horizonte
SUFIJOS = {'T=1h': 'T1h', 'T=2h': 'T2h', 'T=4h': 'T4h', 'T=24h': 'T24h'}

# =============================================================
# CARGA DE TODOS LOS RESULTADOS
# =============================================================

print("Cargando resultados de los 16 scripts...\n")

# datos[horizonte][modelo] = DataFrame
datos = {h: {} for h in HORIZONTES}

for horizonte in HORIZONTES:
    suf = SUFIJOS[horizonte]
    for modelo in MODELOS:
        ruta = os.path.join(DIR_RAIZ, modelo,
                            f'{modelo}_{suf}_resultados.csv')
        if os.path.exists(ruta):
            datos[horizonte][modelo] = pd.read_csv(ruta)
            print(f"  OK  {horizonte} / {modelo}")
        else:
            print(f"  --  {horizonte} / {modelo}: NO ENCONTRADO")

print()

# =============================================================
# CALCULO DE METRICAS — MATRIZ 4 HORIZONTES x 4 MODELOS
# =============================================================

mat_mae_acc  = np.full((len(HORIZONTES), len(MODELOS)), np.nan)
mat_rmse_acc = np.full((len(HORIZONTES), len(MODELOS)), np.nan)
mat_mae_soc  = np.full((len(HORIZONTES), len(MODELOS)), np.nan)
mat_rmse_soc = np.full((len(HORIZONTES), len(MODELOS)), np.nan)
mat_nmae_acc = np.full((len(HORIZONTES), len(MODELOS)), np.nan)
mat_nmae_soc = np.full((len(HORIZONTES), len(MODELOS)), np.nan)

for i, horizonte in enumerate(HORIZONTES):
    for j, modelo in enumerate(MODELOS):
        if modelo not in datos[horizonte]:
            continue
        df = datos[horizonte][modelo]
        ea = df['ACCION_REAL'] - df['ACCION_IA']
        es = df['SOC_REAL']    - df['SOC_IA']
        mat_mae_acc[i, j]  = ea.abs().mean()
        mat_rmse_acc[i, j] = np.sqrt((ea**2).mean())
        mat_mae_soc[i, j]  = es.abs().mean()
        mat_rmse_soc[i, j] = np.sqrt((es**2).mean())
        mat_nmae_acc[i, j] = ea.abs().mean() / P_RNG * 100
        mat_nmae_soc[i, j] = es.abs().mean() / S_RNG * 100

# =============================================================
# TABLAS EN CONSOLA
# =============================================================

def imprimir_matriz(titulo, matriz, unidad=''):
    print(f"\n{'='*65}")
    print(f"  {titulo}")
    print(f"{'='*65}")
    print(f"  {'Horizonte':<10}" + "".join(f"{m:>12}" for m in MODELOS))
    print("-"*65)
    for i, h in enumerate(HORIZONTES):
        fila = f"  {h:<10}"
        for j in range(len(MODELOS)):
            v = matriz[i, j]
            fila += f"{v:>11.4f} " if not np.isnan(v) else f"{'--':>12}"
        print(fila)
    print("="*65)

imprimir_matriz("MATRIZ MAE P_bat (MW)",  mat_mae_acc)
imprimir_matriz("MATRIZ RMSE P_bat (MW)", mat_rmse_acc)
imprimir_matriz("MATRIZ MAE SOC (p.u.)",  mat_mae_soc)
imprimir_matriz("MATRIZ nMAE P_bat (%rango)", mat_nmae_acc)

# Ranking de las 16 combinaciones
print(f"\n{'='*60}")
print("  RANKING — 16 combinaciones por MAE P_bat")
print(f"{'='*60}")
ranking = []
for i, h in enumerate(HORIZONTES):
    for j, m in enumerate(MODELOS):
        if not np.isnan(mat_mae_acc[i, j]):
            ranking.append((mat_mae_acc[i, j], h, m,
                            mat_mae_soc[i, j], mat_nmae_acc[i, j]))
ranking.sort()
for rank, (v, h, m, vs, nv) in enumerate(ranking, 1):
    print(f"  {rank:2d}. {h} + {m:<5}  MAE={v:.4f} MW  "
          f"MAE_SOC={vs:.4f}  nMAE={nv:.1f}%")
print("="*60)

# =============================================================
# FIGURA 1 y 2 — HEATMAPS
# =============================================================

def heatmap(matriz, titulo, fname, cmap='RdYlGn_r', fmt='.4f'):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = np.where(~np.isnan(matriz), matriz, 0)
    vmin, vmax = np.nanmin(matriz), np.nanmax(matriz)
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(MODELOS)));    ax.set_xticklabels(MODELOS, fontsize=11)
    ax.set_yticks(range(len(HORIZONTES))); ax.set_yticklabels(HORIZONTES, fontsize=11)
    ax.set_xlabel('Modelo', fontsize=11); ax.set_ylabel('Horizonte', fontsize=11)
    ax.set_title(titulo, fontsize=12, fontweight='bold')

    med = np.nanmedian(matriz)
    for i in range(len(HORIZONTES)):
        for j in range(len(MODELOS)):
            v = matriz[i, j]
            txt   = f'{v:{fmt}}' if not np.isnan(v) else '--'
            color = 'white' if v > med else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    # Marcar minimo con borde verde
    idx_min = np.unravel_index(np.nanargmin(matriz), matriz.shape)
    ax.add_patch(plt.Rectangle(
        (idx_min[1]-0.5, idx_min[0]-0.5), 1, 1,
        fill=False, edgecolor='lime', linewidth=3, zorder=5))

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardada: {fname}")

heatmap(mat_mae_acc,  'Matriz MAE P_bat (MW) - horizonte x modelo',
        'matriz_MAE_Pbat.png')
heatmap(mat_mae_soc,  'Matriz MAE SOC (p.u.) - horizonte x modelo',
        'matriz_MAE_SOC.png')
heatmap(mat_nmae_acc, 'Matriz nMAE P_bat (% rango) - horizonte x modelo',
        'matriz_nMAE_Pbat.png', fmt='.1f')

# =============================================================
# FIGURA 3 y 4 — LINEAS: METRICA vs HORIZONTE POR MODELO
# =============================================================

def lineas_horizonte(matriz, ylabel, titulo, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, modelo in enumerate(MODELOS):
        y    = matriz[:, j]
        mask = ~np.isnan(y)
        if not mask.any():
            continue
        ax.plot(np.array(H_VALS)[mask], y[mask],
                label=modelo, color=COLORES_MOD[modelo],
                marker=MARCADORES[modelo], linewidth=2, markersize=8)
        for hv, yv in zip(np.array(H_VALS)[mask], y[mask]):
            ax.annotate(f'{yv:.3f}', xy=(hv, yv),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=8, color=COLORES_MOD[modelo])

    ax.set_xlabel('Horizonte temporal (h)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.set_xticks(H_VALS); ax.set_xticklabels(HORIZONTES)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardada: {fname}")

lineas_horizonte(mat_mae_acc,  'MAE P_bat (MW)',
                 'Efecto del horizonte sobre MAE P_bat por modelo',
                 'efecto_horizonte_MAE.png')
lineas_horizonte(mat_rmse_acc, 'RMSE P_bat (MW)',
                 'Efecto del horizonte sobre RMSE P_bat por modelo',
                 'efecto_horizonte_RMSE.png')
lineas_horizonte(mat_nmae_acc, 'nMAE P_bat (% rango fisico)',
                 'Efecto del horizonte sobre nMAE P_bat por modelo',
                 'efecto_horizonte_nMAE.png')

# =============================================================
# FIGURA 5 — BARRAS AGRUPADAS COMPARATIVA COMPLETA
# =============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Comparativa completa - MAE P_bat y MAE SOC',
             fontsize=13, fontweight='bold')

x      = np.arange(len(HORIZONTES))
ancho  = 0.18
offset = np.linspace(-(len(MODELOS)-1)/2,
                      (len(MODELOS)-1)/2, len(MODELOS)) * ancho

for ax, (matriz, ylabel) in zip(axes,
    [(mat_mae_acc, 'MAE P_bat (MW)'),
     (mat_mae_soc, 'MAE SOC (p.u.)')]):
    for j, modelo in enumerate(MODELOS):
        vals = matriz[:, j]
        mask = ~np.isnan(vals)
        ax.bar(x[mask] + offset[j], vals[mask],
               width=ancho, label=modelo,
               color=COLORES_MOD[modelo], alpha=0.85, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(HORIZONTES, fontsize=10)
    ax.set_ylabel(ylabel); ax.set_xlabel('Horizonte temporal')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(DIR, 'barras_comparativa_completa.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Guardada: barras_comparativa_completa.png")

# =============================================================
# FIGURA 6 — MAE POR HORA PARA TODOS LOS HORIZONTES
# =============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('MAE P_bat por hora del dia - comparativa de horizontes',
             fontsize=13, fontweight='bold')

for idx, horizonte in enumerate(HORIZONTES):
    ax = axes[idx // 2][idx % 2]
    if not datos[horizonte]:
        ax.set_title(f'{horizonte} - sin datos'); continue

    for modelo, df in datos[horizonte].items():
        mae_hora = [(df[df['HORA']==h]['ACCION_REAL']
                     - df[df['HORA']==h]['ACCION_IA']).abs().mean()
                    for h in range(T)]
        ax.plot(range(T), mae_hora, label=modelo,
                color=COLORES_MOD[modelo], marker=MARCADORES[modelo],
                linewidth=1.5, markersize=4)

    ax.axvspan(7, 9,   alpha=0.08, color='gray')
    ax.axvspan(18, 21, alpha=0.08, color='gray')
    ax.set_title(f'Horizonte {horizonte}')
    ax.set_xlabel('Hora del dia'); ax.set_ylabel('MAE P_bat (MW)')
    ax.set_xticks(range(0, T, 2))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(DIR, 'error_hora_todos_horizontes.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Guardada: error_hora_todos_horizontes.png")

# =============================================================
# FIGURA 7 — TABLA VISUAL (imagen PNG de la tabla resumen)
# =============================================================

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

cols = ['Horizonte', 'Modelo', 'MAE P_bat\n(MW)', 'RMSE P_bat\n(MW)',
        'MAE SOC\n(p.u.)', 'RMSE SOC\n(p.u.)',
        'nMAE P_bat\n(%rango)', 'nMAE SOC\n(%rango)']

filas_tabla = []
for i, h in enumerate(HORIZONTES):
    for j, m in enumerate(MODELOS):
        if np.isnan(mat_mae_acc[i, j]):
            continue
        filas_tabla.append([
            h, m,
            f"{mat_mae_acc[i,j]:.4f}",
            f"{mat_rmse_acc[i,j]:.4f}",
            f"{mat_mae_soc[i,j]:.4f}",
            f"{mat_rmse_soc[i,j]:.4f}",
            f"{mat_nmae_acc[i,j]:.2f}%",
            f"{mat_nmae_soc[i,j]:.2f}%",
        ])

tabla = ax.table(cellText=filas_tabla, colLabels=cols,
                 loc='center', cellLoc='center')
tabla.auto_set_font_size(False)
tabla.set_fontsize(8)
tabla.scale(1.1, 1.6)

# Cabecera
for j in range(len(cols)):
    tabla[0, j].set_facecolor('#2C3E50')
    tabla[0, j].set_text_props(color='white', fontweight='bold')

# Colorear por horizonte
color_fila = {'T=1h': '#FADBD8', 'T=2h': '#FAE5D3',
              'T=4h': '#D5F5E3', 'T=24h': '#D6EAF8'}
for k, (h, *_) in enumerate(filas_tabla):
    for j in range(len(cols)):
        tabla[k+1, j].set_facecolor(color_fila.get(h, 'white'))

# Marcar minimo de cada columna metrica en negrita
for col_idx in range(2, len(cols)):
    vals = []
    for k, fila in enumerate(filas_tabla):
        try:
            vals.append((float(fila[col_idx].replace('%','')), k))
        except:
            vals.append((float('inf'), k))
    if vals:
        min_k = min(vals, key=lambda x: x[0])[1]
        tabla[min_k+1, col_idx].set_text_props(fontweight='bold', color='darkgreen')

ax.set_title('Tabla resumen — todas las combinaciones horizonte x modelo',
             fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(DIR, 'tabla_comparativa_visual.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Guardada: tabla_comparativa_visual.png")

# =============================================================
# RESUMEN FINAL
# =============================================================

print("\n" + "="*65)
print("  COMPARATIVA CRUZADA COMPLETADA")
print("="*65)
if not np.all(np.isnan(mat_mae_acc)):
    idx_min = np.unravel_index(np.nanargmin(mat_mae_acc), mat_mae_acc.shape)
    print(f"\n  MEJOR COMBINACION (menor MAE P_bat):")
    print(f"    Horizonte : {HORIZONTES[idx_min[0]]}")
    print(f"    Modelo    : {MODELOS[idx_min[1]]}")
    print(f"    MAE P_bat : {mat_mae_acc[idx_min]:.4f} MW")
    print(f"    MAE SOC   : {mat_mae_soc[idx_min]:.4f} p.u.")
    print(f"    nMAE      : {mat_nmae_acc[idx_min]:.2f}%")
print("\n  Figuras en: COMPARATIVAS/")
for i, f in enumerate([
    'matriz_MAE_Pbat.png', 'matriz_MAE_SOC.png', 'matriz_nMAE_Pbat.png',
    'efecto_horizonte_MAE.png', 'efecto_horizonte_RMSE.png', 'efecto_horizonte_nMAE.png',
    'barras_comparativa_completa.png', 'error_hora_todos_horizontes.png',
    'tabla_comparativa_visual.png'], 1):
    print(f"    {i}) {f}")
print("="*65)