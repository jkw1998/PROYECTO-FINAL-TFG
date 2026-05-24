# =============================================================
# TFG - COMPARATIVA CRUZADA HORIZONTES x MODELOS
# =============================================================
# Autor: James Kagunda Wangari
# Grado en Ingenieria Electrica - Universidad de Malaga
#
# Descripcion:
# ------------
# Lee los resultados de los 4 horizontes (T=1h, T=2h, T=4h,
# T=24h) y los 4 modelos (MLP, RNN, LSTM, GRU) y genera
# las tablas y figuras del analisis comparativo del Cap. 7.
#
# Estructura de CSV esperada:
# ----------------------------
# Para T=24h los CSV estan en el mismo directorio que este script:
#   mlp_resultados.csv, rnn_resultados.csv,
#   lstm_resultados.csv, gru_resultados.csv
#
# Para T=1h, T=2h, T=4h estan en subcarpetas:
#   ../1 hora dataset/mlp_resultados_T1h.csv   (etc.)
#   ../2 horas dataset/mlp_resultados_T2h.csv  (etc.)
#   ../4 horas dataset/mlp_resultados_T4h.csv  (etc.)
#
# AJUSTA LAS RUTAS al final del bloque de configuracion si
# tu estructura de carpetas es diferente.
#
# Figuras generadas:
# ------------------
#  1) matriz_MAE_Pbat.png
#     Heatmap de la matriz 4x4 horizonte x modelo (MAE P_bat)
#
#  2) matriz_MAE_SOC.png
#     Heatmap de la matriz 4x4 horizonte x modelo (MAE SOC)
#
#  3) efecto_horizonte_MAE.png
#     Lineas: MAE P_bat vs horizonte para cada modelo
#
#  4) efecto_horizonte_RMSE.png
#     Lineas: RMSE P_bat vs horizonte para cada modelo
#
#  5) boxplot_coste_horizontes.png
#     Boxplot del coste J* del optimizador por horizonte +
#     incremento porcentual respecto a T=24h
#
#  6) barras_comparativa_completa.png
#     Barras agrupadas: MAE P_bat para todas las combinaciones
#
#  7) error_hora_todos_horizontes.png
#     MAE por hora del dia para el mejor modelo en cada horizonte
#
# Tablas impresas en consola:
# ---------------------------
#  - Matriz MAE P_bat (4 horizontes x 4 modelos)
#  - Matriz RMSE P_bat
#  - Matriz MAE SOC
#  - Tabla ranking de las 16 combinaciones por MAE P_bat
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =============================================================
# CONFIGURACION DE RUTAS
# =============================================================

# Directorio de este script (comparativas/)
DIR = os.path.dirname(os.path.abspath(__file__))

# Directorio raiz del proyecto (un nivel arriba de comparativas/)
DIR_RAIZ = os.path.dirname(DIR)

# Las figuras se guardan en comparativas/ (junto a este script)
# Los CSV de resultados estan en las carpetas de cada horizonte
DIRS_HORIZONTES = {
    'T=1h' : os.path.join(DIR_RAIZ, '1 hora dataset'),
    'T=2h' : os.path.join(DIR_RAIZ, '2 horas dataset'),
    'T=4h' : os.path.join(DIR_RAIZ, '4 horas dataset'),
    'T=24h': DIR_RAIZ,
}

SUFIJOS = {
    'T=1h' : '_T1h',
    'T=2h' : '_T2h',
    'T=4h' : '_T4h',
    'T=24h': '',
}

# Datasets completos para comparar coste del optimizador
DIRS_COMPLETO = {
    'T=1h' : os.path.join(DIR_RAIZ, '1 hora dataset', 'dataset_vpp_completo_T1h.csv'),
    'T=2h' : os.path.join(DIR_RAIZ, '2 horas dataset', 'dataset_vpp_completo_T2h.csv'),
    'T=4h' : os.path.join(DIR_RAIZ, '4 horas dataset', 'dataset_vpp_completo_T4h.csv'),
    'T=24h': os.path.join(DIR_RAIZ, 'dataset_vpp_completo.csv'),
}

MODELOS    = ['MLP', 'RNN', 'LSTM', 'GRU']
HORIZONTES = ['T=1h', 'T=2h', 'T=4h', 'T=24h']
H_VALS     = [1, 2, 4, 24]   # valores numericos para ejes X

COLORES_MOD = {'MLP': '#2196F3', 'RNN': '#FF9800',
               'LSTM': '#4CAF50', 'GRU': '#F44336'}
COLORES_HOR = {'T=1h': '#E91E63', 'T=2h': '#9C27B0',
               'T=4h': '#FF9800', 'T=24h': '#2196F3'}
MARCADORES  = {'MLP': 'o', 'RNN': 's', 'LSTM': '^', 'GRU': 'D'}

T     = 24
SPLIT = 800

# =============================================================
# CARGA DE TODOS LOS RESULTADOS
# =============================================================

print("Cargando resultados de todos los horizontes...\n")

# datos[horizonte][modelo] = DataFrame
datos = {h: {} for h in HORIZONTES}

for horizonte in HORIZONTES:
    dir_h  = DIRS_HORIZONTES[horizonte]
    sufijo = SUFIJOS[horizonte]
    for modelo in MODELOS:
        nombre = f"{modelo.lower()}_resultados{sufijo}.csv"
        ruta   = os.path.join(dir_h, nombre)
        if os.path.exists(ruta):
            datos[horizonte][modelo] = pd.read_csv(ruta)
            print(f"  OK  {horizonte} / {modelo}: {ruta}")
        else:
            print(f"  --  {horizonte} / {modelo}: NO ENCONTRADO ({ruta})")

print()

# =============================================================
# CALCULO DE METRICAS PARA TODAS LAS COMBINACIONES
# =============================================================

# mat_mae_acc[h_idx][m_idx] = MAE P_bat
mat_mae_acc  = np.full((len(HORIZONTES), len(MODELOS)), np.nan)
mat_rmse_acc = np.full((len(HORIZONTES), len(MODELOS)), np.nan)
mat_mae_soc  = np.full((len(HORIZONTES), len(MODELOS)), np.nan)
mat_rmse_soc = np.full((len(HORIZONTES), len(MODELOS)), np.nan)

for i, horizonte in enumerate(HORIZONTES):
    for j, modelo in enumerate(MODELOS):
        if modelo not in datos[horizonte]:
            continue
        df = datos[horizonte][modelo]
        err_acc = df['ACCION_REAL'] - df['ACCION_IA']
        err_soc = df['SOC_REAL']    - df['SOC_IA']
        mat_mae_acc[i, j]  = err_acc.abs().mean()
        mat_rmse_acc[i, j] = np.sqrt((err_acc**2).mean())
        mat_mae_soc[i, j]  = err_soc.abs().mean()
        mat_rmse_soc[i, j] = np.sqrt((err_soc**2).mean())

# =============================================================
# TABLAS EN CONSOLA
# =============================================================

def imprimir_matriz(titulo, matriz, fmt='.4f'):
    print(f"\n{'='*65}")
    print(f"  {titulo}")
    print(f"{'='*65}")
    header = f"  {'Horizonte':<10}" + "".join(f"{m:>10}" for m in MODELOS)
    print(header)
    print("-"*65)
    for i, h in enumerate(HORIZONTES):
        fila = f"  {h:<10}"
        for j in range(len(MODELOS)):
            v = matriz[i, j]
            fila += f"{v:>10{fmt}}" if not np.isnan(v) else f"{'--':>10}"
        print(fila)
    print("="*65)

imprimir_matriz("MATRIZ MAE P_bat (MW)  — horizonte x modelo", mat_mae_acc)
imprimir_matriz("MATRIZ RMSE P_bat (MW) — horizonte x modelo", mat_rmse_acc)
imprimir_matriz("MATRIZ MAE SOC (p.u.)  — horizonte x modelo", mat_mae_soc)
imprimir_matriz("MATRIZ RMSE SOC (p.u.) — horizonte x modelo", mat_rmse_soc)

# Ranking de las combinaciones
print(f"\n{'='*65}")
print("  RANKING — 16 combinaciones por MAE P_bat (menor = mejor)")
print(f"{'='*65}")
filas_ranking = []
for i, h in enumerate(HORIZONTES):
    for j, m in enumerate(MODELOS):
        v = mat_mae_acc[i, j]
        if not np.isnan(v):
            filas_ranking.append((v, h, m))
filas_ranking.sort()
for rank, (v, h, m) in enumerate(filas_ranking, 1):
    print(f"  {rank:2d}. {h} + {m:<5}  MAE = {v:.4f} MW")
print("="*65)

# =============================================================
# FIGURA 1 — HEATMAP MATRIZ MAE P_bat
# =============================================================

def heatmap_matriz(matriz, titulo, nombre_archivo, fmt='.4f', cmap='RdYlGn_r'):
    fig, ax = plt.subplots(figsize=(7, 4))
    mask = ~np.isnan(matriz)
    data_plot = np.where(mask, matriz, 0)

    # Solo colorear celdas con datos
    im = ax.imshow(data_plot, cmap=cmap, aspect='auto',
                   vmin=np.nanmin(matriz), vmax=np.nanmax(matriz))

    ax.set_xticks(range(len(MODELOS)))
    ax.set_yticks(range(len(HORIZONTES)))
    ax.set_xticklabels(MODELOS, fontsize=11)
    ax.set_yticklabels(HORIZONTES, fontsize=11)
    ax.set_xlabel('Modelo', fontsize=11)
    ax.set_ylabel('Horizonte', fontsize=11)
    ax.set_title(titulo, fontsize=12, fontweight='bold')

    for i in range(len(HORIZONTES)):
        for j in range(len(MODELOS)):
            v = matriz[i, j]
            txt = f'{v:{fmt}}' if not np.isnan(v) else '--'
            color = 'white' if v > np.nanmedian(matriz) else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    # Marcar la celda minima con borde verde
    idx_min = np.unravel_index(np.nanargmin(matriz), matriz.shape)
    rect = plt.Rectangle((idx_min[1]-0.5, idx_min[0]-0.5), 1, 1,
                          fill=False, edgecolor='lime',
                          linewidth=3, zorder=5)
    ax.add_patch(rect)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    ruta = os.path.join(DIR, nombre_archivo)
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardada: {nombre_archivo}")

heatmap_matriz(mat_mae_acc,
               'Matriz MAE $P_{bat}$ (MW) — horizonte × modelo',
               'matriz_MAE_Pbat.png')
heatmap_matriz(mat_mae_soc,
               'Matriz MAE SOC (p.u.) — horizonte × modelo',
               'matriz_MAE_SOC.png', fmt='.4f', cmap='RdYlGn_r')

# =============================================================
# FIGURA 2 — LINEAS: MAE vs HORIZONTE POR MODELO
# =============================================================

def lineas_horizonte(matriz, ylabel, titulo, nombre_archivo):
    fig, ax = plt.subplots(figsize=(9, 5))

    for j, modelo in enumerate(MODELOS):
        y_vals = matriz[:, j]
        mask   = ~np.isnan(y_vals)
        if mask.any():
            ax.plot(np.array(H_VALS)[mask], y_vals[mask],
                    label=modelo,
                    color=COLORES_MOD[modelo],
                    marker=MARCADORES[modelo],
                    linewidth=2, markersize=8)
            # Anotar valores
            for hv, yv in zip(np.array(H_VALS)[mask], y_vals[mask]):
                ax.annotate(f'{yv:.3f}',
                            xy=(hv, yv),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center', fontsize=8,
                            color=COLORES_MOD[modelo])

    ax.set_xlabel('Horizonte temporal (h)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.set_xticks(H_VALS)
    ax.set_xticklabels(['T=1h', 'T=2h', 'T=4h', 'T=24h'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    ruta = os.path.join(DIR, nombre_archivo)
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardada: {nombre_archivo}")

lineas_horizonte(mat_mae_acc,
                 'MAE $P_{bat}$ (MW)',
                 'Efecto del horizonte sobre MAE $P_{bat}$ por modelo',
                 'efecto_horizonte_MAE.png')

lineas_horizonte(mat_rmse_acc,
                 'RMSE $P_{bat}$ (MW)',
                 'Efecto del horizonte sobre RMSE $P_{bat}$ por modelo',
                 'efecto_horizonte_RMSE.png')

# =============================================================
# FIGURA 3 — BOXPLOT COSTE OPTIMIZADOR POR HORIZONTE
# =============================================================

costes_por_horizonte = {}
for horizonte, ruta_csv in DIRS_COMPLETO.items():
    if not os.path.exists(ruta_csv):
        print(f"  [AVISO] No encontrado dataset completo: {ruta_csv}")
        continue
    df_full = pd.read_csv(ruta_csv)
    df_t    = df_full[df_full['split'] == 'test'].reset_index(drop=True)

    # Si el dataset tiene columna coste_total la usamos directamente
    if 'coste_total' in df_t.columns:
        costes_por_horizonte[horizonte] = df_t['coste_total'].values
    else:
        # Recalcular desde p_red y precio
        costes = []
        for _, row in df_t.iterrows():
            c = 0
            for h in range(T):
                p_red = row[f'p_red_h{h}']
                lam   = row[f'precio_h{h}']
                p_dg  = row.get(f'p_dg_h{h}', 0)
                p_ch  = max(0,  row[f'p_bat_h{h}'])
                p_dis = max(0, -row[f'p_bat_h{h}'])
                c += (max(0, p_red) * lam
                      - max(0, -p_red) * lam * 0.80
                      + 10.0 * p_dg
                      + 2.0  * (p_ch + p_dis))
            costes.append(c)
        costes_por_horizonte[horizonte] = np.array(costes)

if len(costes_por_horizonte) >= 2:
    h_disponibles = list(costes_por_horizonte.keys())
    ref = costes_por_horizonte.get('T=24h', None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Efecto del horizonte sobre el coste del optimizador',
                 fontsize=13, fontweight='bold')

    # Subplot 1: boxplot
    data_box   = [costes_por_horizonte[h] for h in h_disponibles]
    labels_box = h_disponibles
    bp = axes[0].boxplot(data_box, labels=labels_box,
                         patch_artist=True, notch=False)
    for patch, h in zip(bp['boxes'], h_disponibles):
        patch.set_facecolor(COLORES_HOR.get(h, 'steelblue'))
        patch.set_alpha(0.7)
    axes[0].set_ylabel('Coste $\\mathcal{J}^*$ (€/día)')
    axes[0].set_xlabel('Horizonte')
    axes[0].set_title('Distribución del coste óptimo')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Subplot 2: incremento porcentual respecto a T=24h
    if ref is not None:
        ref_media = ref.mean()
        incrementos = {h: (costes_por_horizonte[h].mean() - ref_media)
                          / ref_media * 100
                       for h in h_disponibles}
        h_plot = [h for h in h_disponibles if h != 'T=24h']
        inc_plot = [incrementos[h] for h in h_plot]
        bars = axes[1].bar(h_plot, inc_plot,
                           color=[COLORES_HOR.get(h, 'gray') for h in h_plot],
                           alpha=0.8, edgecolor='white')
        for bar, val in zip(bars, inc_plot):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.1,
                         f'+{val:.1f}%', ha='center', fontsize=10)
        axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[1].set_ylabel('Incremento coste respecto T=24h (%)')
        axes[1].set_xlabel('Horizonte')
        axes[1].set_title('Sobrecoste vs. T=24h (referencia óptima)')
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    ruta = os.path.join(DIR, 'boxplot_coste_horizontes.png')
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardada: boxplot_coste_horizontes.png")

# =============================================================
# FIGURA 4 — BARRAS AGRUPADAS COMPARATIVA COMPLETA
# =============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Comparativa completa — MAE $P_{bat}$ y MAE SOC',
             fontsize=13, fontweight='bold')

x      = np.arange(len(HORIZONTES))
ancho  = 0.18
offset = np.linspace(-(len(MODELOS)-1)/2, (len(MODELOS)-1)/2,
                     len(MODELOS)) * ancho

for k, ax, (matriz, ylabel) in enumerate(zip(
        axes, [(mat_mae_acc, 'MAE $P_{bat}$ (MW)'),
               (mat_mae_soc, 'MAE SOC (p.u.)')])):
    for j, modelo in enumerate(MODELOS):
        vals = matriz[:, j]
        mask = ~np.isnan(vals)
        ax.bar(x[mask] + offset[j], vals[mask],
               width=ancho, label=modelo,
               color=COLORES_MOD[modelo], alpha=0.85,
               edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONTES, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Horizonte temporal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
ruta = os.path.join(DIR, 'barras_comparativa_completa.png')
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardada: barras_comparativa_completa.png")

# =============================================================
# FIGURA 5 — MAE POR HORA PARA TODOS LOS HORIZONTES
#            (usando el mejor modelo de cada horizonte)
# =============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('MAE $P_{bat}$ por hora del día — comparativa de horizontes',
             fontsize=13, fontweight='bold')

for idx, horizonte in enumerate(HORIZONTES):
    ax = axes[idx // 2][idx % 2]
    if not datos[horizonte]:
        ax.set_title(f'{horizonte} — sin datos')
        continue

    for modelo, df in datos[horizonte].items():
        mae_hora = []
        for h in range(T):
            df_h  = df[df['HORA'] == h]
            if len(df_h) == 0:
                mae_hora.append(np.nan)
            else:
                err_h = (df_h['ACCION_REAL'] - df_h['ACCION_IA']).abs().mean()
                mae_hora.append(err_h)
        ax.plot(range(T), mae_hora,
                label=modelo, color=COLORES_MOD[modelo],
                marker=MARCADORES[modelo], linewidth=1.5,
                markersize=4)

    ax.axvspan(7, 9,   alpha=0.08, color='gray')
    ax.axvspan(18, 21, alpha=0.08, color='gray')
    ax.set_title(f'Horizonte {horizonte}')
    ax.set_xlabel('Hora del día')
    ax.set_ylabel('MAE $P_{bat}$ (MW)')
    ax.set_xticks(range(0, T, 2))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
ruta = os.path.join(DIR, 'error_hora_todos_horizontes.png')
plt.savefig(ruta, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardada: error_hora_todos_horizontes.png")

# =============================================================
# RESUMEN FINAL
# =============================================================

print("\n" + "="*65)
print("  COMPARATIVA CRUZADA HORIZONTES x MODELOS COMPLETADA")
print("="*65)
print("  Figuras generadas (en la carpeta T=24h):")
print("    1) matriz_MAE_Pbat.png")
print("    2) matriz_MAE_SOC.png")
print("    3) efecto_horizonte_MAE.png")
print("    4) efecto_horizonte_RMSE.png")
if len(costes_por_horizonte) >= 2:
    print("    5) boxplot_coste_horizontes.png")
print("    6) barras_comparativa_completa.png")
print("    7) error_hora_todos_horizontes.png")
print()

# Mejor combinacion global
if not np.all(np.isnan(mat_mae_acc)):
    idx_min = np.unravel_index(np.nanargmin(mat_mae_acc), mat_mae_acc.shape)
    print(f"  MEJOR COMBINACION (menor MAE P_bat):")
    print(f"    Horizonte : {HORIZONTES[idx_min[0]]}")
    print(f"    Modelo    : {MODELOS[idx_min[1]]}")
    print(f"    MAE P_bat : {mat_mae_acc[idx_min]:.4f} MW")
    print(f"    MAE SOC   : {mat_mae_soc[idx_min]:.4f} p.u.")
print("="*65)