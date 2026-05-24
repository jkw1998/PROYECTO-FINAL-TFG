# ==============================================================
# TFG - GENERACIÓN DE DATASET MEDIANTE OPTIMIZACIÓN MPC/MILP
# HORIZONTE TEMPORAL: T = 1 hora
# ==============================================================
#
# Autor: James Kagunda Wangari
# Grado en Ingeniería Eléctrica - Universidad de Málaga
#
# Descripción:
# ------------
# Genera el dataset de 1000 escenarios diarios resolviendo
# 24 problemas MILP INDEPENDIENTES de 1 hora cada uno.
#
# Estrategia de horizonte T=1h:
# ------------------------------
# Cada hora del día se optimiza de forma independiente, con
# información únicamente de esa hora. No hay anticipación.
# El SOC final de la hora h se pasa como SOC inicial de la
# hora h+1, garantizando la continuidad física de la batería.
#
# Esquema por día (24 ventanas de 1h):
#
#   Ventana 1:  hora [0]      -> SOC_ini = SOC_0 (aleatorio)
#   Ventana 2:  hora [1]      -> SOC_ini = SOC_final(ventana 1)
#   Ventana 3:  hora [2]      -> SOC_ini = SOC_final(ventana 2)
#   ...
#   Ventana 24: hora [23]     -> SOC_ini = SOC_final(ventana 23)
#
# Cada ventana es un problema MILP independiente.
# La solución de cada ventana (p_bat, SOC) se concatena para
# reconstruir el perfil completo del día (24 valores).
#
# NEUTRALIDAD ENERGÉTICA DIARIA:
# La restricción SOC_fin(día) >= SOC_0 se aplica SOLO en la
# última hora (h=23), no en cada hora individual. Esto permite
# que la batería cargue y descargue libremente durante el día,
# garantizando únicamente que al final no quede más descargada
# que al principio. Sin esta corrección, la restricción hora a
# hora impediría cualquier descarga (dataset trivial con P_bat=0).
#
# Comparación con T=24h:
# -----------------------
# T=24h optimiza el día completo de golpe → máxima anticipación.
# T=1h optimiza hora a hora sin visión futura → reactivo puro.
# Se espera un coste de operación más alto que T=24h por la
# falta de anticipación (no puede planificar arbitraje).
#
# Estructura del dataset generado (igual que T=24h):
# ---------------------------------------------------
#   dataset_vpp_completo_T1h.csv  -> todas las variables
#   dataset_vpp_ia_T1h.csv        -> entradas/salidas IA
#
# Cada fila = 1 día completo (24h reconstruidas).
# Columnas idénticas al dataset T=24h para comparación directa.
#
# ==============================================================

import os
import numpy as np
import pandas as pd
from pyomo.environ import *

# Directorio donde está este script — los CSV se guardarán aquí
DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================
# SEMILLA ALEATORIA (misma que T=24h para escenarios idénticos)
# ==============================================================

np.random.seed(42)

# ==============================================================
# CONFIGURACIÓN GENERAL
# ==============================================================

DIAS       = 1000   # número total de escenarios diarios
T_DIA      = 24     # horas totales por día
T_VEN      = 1      # horas por ventana de optimización
DELTA_T    = 1      # resolución temporal (h)

DIAS_TRAIN = 800    # escenarios para entrenamiento
DIAS_TEST  = 200    # escenarios para evaluación

# ==============================================================
# PARÁMETROS DEL SISTEMA
# (idénticos a T=24h para comparación directa)
# ==============================================================

# --- Batería (BESS) ---
SOC_MIN    = 0.20
SOC_MAX    = 0.90
P_CH_MAX   = 0.30
P_DIS_MAX  = 0.50
ETA_CH     = 0.95
ETA_DIS    = 0.95
E_BAT_CAP  = 1.0
C_DEG      = 2.0

# --- Red eléctrica ---
P_GRID_MAX   = 10.0
FACTOR_VENTA = 0.80

# --- Turbina de gas (respaldo) ---
P_DG_MIN  = 1.0
P_DG_MAX  = 5.0
DG_RAMP   = 2.0
C_DG_B    = 10.0

# ==============================================================
# GENERACIÓN DE ESCENARIOS
# (función idéntica a T=24h — misma semilla → mismos escenarios)
# ==============================================================

def generar_escenario():
    """
    Genera un escenario diario completo (24h) con perfiles de
    precios, generación PV, demanda y SOC inicial aleatorio.
    Función idéntica a la versión T=24h para garantizar que
    ambos datasets contienen exactamente los mismos escenarios
    y las comparaciones de coste son válidas.
    """
    t = np.arange(T_DIA)

    shift   = np.random.uniform(-3, 3)
    precios = (
        50
        + 30 * np.sin((t - 7 + shift) / 24 * 2 * np.pi)
        + np.random.normal(0, 4, T_DIA)
    )
    precios = np.clip(precios, 5, 120)

    pv = np.zeros(T_DIA)
    for h in range(T_DIA):
        if 7 <= h <= 18:
            pv[h] = 6 * np.sin((h - 6) / 12 * np.pi)
    pv *= np.random.uniform(0.6, 1.2)

    demanda = (
        1.5
        + 0.5 * np.sin((t - 6) / 24 * 2 * np.pi)
        + np.random.normal(0, 0.15, T_DIA)
    )
    demanda = np.clip(demanda, 0.5, 3.0)

    soc0 = np.random.uniform(SOC_MIN + 0.10, SOC_MAX - 0.10)

    return precios, pv, demanda, soc0

# ==============================================================
# MODELO DE OPTIMIZACIÓN MILP — VENTANA DE 1 HORA
# ==============================================================

def optimizar_ventana_1h(precio_h, pv_h, demanda_h, soc_ini,
                          p_dg_prev=None,
                          es_ultima_hora=False, soc0_dia=None):
    """
    Resuelve el problema MILP para UNA sola hora.

    Parámetros
    ----------
    precio_h       : float - precio del mercado en esa hora (€/MWh)
    pv_h           : float - generación fotovoltaica en esa hora (MW)
    demanda_h      : float - demanda eléctrica en esa hora (MW)
    soc_ini        : float - SOC inicial de esta hora (p.u.)
    p_dg_prev      : float - potencia DG hora anterior (MW) para rampa
    es_ultima_hora : bool  - True si es la hora 23 del día
    soc0_dia       : float - SOC inicial del día (para neutralidad)

    Retorna
    -------
    resultado : dict con p_bat, p_ch, p_dis, soc_fin, p_red,
                p_dg, coste — o None si el solver falla
    """

    model = ConcreteModel()
    model.T = RangeSet(0, 0)   # una sola hora

    # Variables continuas
    model.P_ch        = Var(model.T, bounds=(0, P_CH_MAX))
    model.P_dis       = Var(model.T, bounds=(0, P_DIS_MAX))
    model.SOC         = Var(model.T, bounds=(SOC_MIN, SOC_MAX))
    model.P_grid_buy  = Var(model.T, bounds=(0, P_GRID_MAX))
    model.P_grid_sell = Var(model.T, bounds=(0, P_GRID_MAX))
    model.P_dg        = Var(model.T, bounds=(P_DG_MIN, P_DG_MAX))

    # Variables binarias (no simultaneidad carga/descarga)
    model.u_ch  = Var(model.T, within=Binary)
    model.u_dis = Var(model.T, within=Binary)

    # Función objetivo: minimizar coste de esta hora.
    # En la última hora del día se añade una penalización blanda
    # que incentiva devolver el SOC a su valor inicial del día.
    # Se usa una variable auxiliar s_neg >= 0 que captura el
    # déficit SOC_fin - soc0_dia cuando es negativo:
    #   s_neg >= soc0_dia - SOC[0]   (s_neg = max(0, soc0-SOC_fin))
    # La penalización es P_NEU * s_neg con coeficiente alto para
    # que el optimizador prefiera cargar antes de terminar el día.
    # Esto evita la infactibilidad de la restricción dura cuando
    # el SOC encadenado llega bajo a la hora 23.
    P_NEU = 500.0   # €/p.u. — penalización por déficit de neutralidad

    if es_ultima_hora and soc0_dia is not None:
        model.s_neg = Var(bounds=(0, None))   # déficit SOC >= 0

    def objective_rule(m):
        t = 0
        coste = (
              precio_h * m.P_grid_buy[t]  * DELTA_T
            - FACTOR_VENTA * precio_h * m.P_grid_sell[t] * DELTA_T
            + C_DG_B * m.P_dg[t] * DELTA_T
            + C_DEG  * (m.P_ch[t] + m.P_dis[t]) * DELTA_T
        )
        if es_ultima_hora and soc0_dia is not None:
            coste += P_NEU * m.s_neg
        return coste

    model.obj = Objective(rule=objective_rule, sense=minimize)

    model.constraints = ConstraintList()
    t = 0

    # Balance de potencia
    model.constraints.add(
        pv_h + model.P_dis[t] + model.P_grid_buy[t] + model.P_dg[t]
        ==
        demanda_h + model.P_ch[t] + model.P_grid_sell[t]
    )

    # Límites de carga/descarga vinculados a binarias
    model.constraints.add(model.P_ch[t]  <= P_CH_MAX  * model.u_ch[t])
    model.constraints.add(model.P_dis[t] <= P_DIS_MAX * model.u_dis[t])

    # No simultaneidad
    model.constraints.add(model.u_ch[t] + model.u_dis[t] <= 1)

    # Dinámica del SOC
    model.constraints.add(
        model.SOC[t]
        == soc_ini
           + (ETA_CH * model.P_ch[t] * DELTA_T
              - model.P_dis[t] * DELTA_T / ETA_DIS) / E_BAT_CAP
    )

    # Restricción de rampa de la turbina (si no es la primera hora)
    if p_dg_prev is not None:
        model.constraints.add(
            model.P_dg[t] - p_dg_prev <= DG_RAMP
        )
        model.constraints.add(
            p_dg_prev - model.P_dg[t] <= DG_RAMP
        )

    # Neutralidad energética diaria (blanda): solo en la última hora.
    # s_neg >= soc0_dia - SOC[0]  →  s_neg captura el déficit de SOC.
    # La penalización en la función objetivo incentiva s_neg = 0,
    # es decir, que SOC_fin >= soc0_dia, sin causar infactibilidad.
    if es_ultima_hora and soc0_dia is not None:
        model.constraints.add(
            model.s_neg >= soc0_dia - model.SOC[0]
        )

    # Resolución con HiGHS
    solver = SolverFactory('highs')
    try:
        resultado = solver.solve(model, tee=False)
        if (resultado.solver.termination_condition
                != TerminationCondition.optimal):
            return None
    except Exception:
        # Versiones nuevas de Pyomo lanzan excepción en lugar de
        # devolver un status de infactibilidad
        return None

    return {
        'p_ch'   : value(model.P_ch[0]),
        'p_dis'  : value(model.P_dis[0]),
        'p_bat'  : value(model.P_ch[0]) - value(model.P_dis[0]),
        'soc'    : value(model.SOC[0]),
        'p_buy'  : value(model.P_grid_buy[0]),
        'p_sell' : value(model.P_grid_sell[0]),
        'p_red'  : value(model.P_grid_buy[0]) - value(model.P_grid_sell[0]),
        'p_dg'   : value(model.P_dg[0]),
        'coste'  : value(model.obj),
    }

# ==============================================================
# BUCLE PRINCIPAL DE GENERACIÓN
# ==============================================================

dataset_completo = []
dataset_ia       = []
fallos           = 0

print(f"\nGenerando {DIAS} escenarios con horizonte T=1h")
print(f"  Estrategia: 24 problemas MILP independientes por día")
print(f"  SOC encadenado: SOC_fin(h) -> SOC_ini(h+1)")
print(f"  Solver: HiGHS (MILP — optimalidad global)\n")

for d in range(DIAS):

    precios, pv, demanda, soc0 = generar_escenario()

    # Arrays para almacenar la solución del día completo
    p_bat_dia  = np.zeros(T_DIA)
    soc_dia    = np.zeros(T_DIA)
    p_red_dia  = np.zeros(T_DIA)
    p_dg_dia   = np.zeros(T_DIA)
    coste_dia  = 0.0

    soc_actual  = soc0
    p_dg_prev   = None
    fallo_dia   = False

    # --- 24 ventanas de 1 hora ---
    for h in range(T_DIA):

        res = optimizar_ventana_1h(
            precio_h       = precios[h],
            pv_h           = pv[h],
            demanda_h      = demanda[h],
            soc_ini        = soc_actual,
            p_dg_prev      = p_dg_prev,
            es_ultima_hora = (h == T_DIA - 1),
            soc0_dia       = soc0
        )

        if res is None:
            fallo_dia = True
            break

        # Guardar solución de esta hora
        p_bat_dia[h] = round(res['p_bat'], 4)
        soc_dia[h]   = round(res['soc'],   4)
        p_red_dia[h] = round(res['p_red'], 4)
        p_dg_dia[h]  = round(res['p_dg'],  4)
        coste_dia   += res['coste']

        # Encadenar: SOC final -> SOC inicial siguiente hora
        soc_actual = res['soc']
        p_dg_prev  = res['p_dg']

    if fallo_dia:
        fallos += 1
        print(f"  [AVISO] Escenario {d} con fallo en alguna hora — omitido")
        continue

    split = 'train' if d < DIAS_TRAIN else 'test'

    # ==========================================================
    # DATASET COMPLETO
    # ==========================================================

    fila_full = {
        'dia_id'      : d,
        'split'       : split,
        'soc_inicial' : round(soc0, 4),
        'coste_total' : round(coste_dia, 4),
        'horizonte'   : 1,
    }
    for h in range(T_DIA):
        fila_full[f'precio_h{h}']  = round(precios[h], 4)
    for h in range(T_DIA):
        fila_full[f'pv_h{h}']      = round(pv[h], 4)
    for h in range(T_DIA):
        fila_full[f'demanda_h{h}'] = round(demanda[h], 4)
    for h in range(T_DIA):
        fila_full[f'p_bat_h{h}']   = p_bat_dia[h]
    for h in range(T_DIA):
        fila_full[f'soc_h{h}']     = soc_dia[h]
    for h in range(T_DIA):
        fila_full[f'p_red_h{h}']   = p_red_dia[h]
    for h in range(T_DIA):
        fila_full[f'p_dg_h{h}']    = p_dg_dia[h]

    dataset_completo.append(fila_full)

    # ==========================================================
    # DATASET IA (misma estructura que T=24h)
    # Entradas: soc_inicial + precio(24) + pv(24) + demanda(24)
    # Salidas:  p_bat(24) + soc(24)
    # ==========================================================

    fila_ia = {
        'dia_id'      : d,
        'split'       : split,
        'soc_inicial' : round(soc0, 4),
    }
    for h in range(T_DIA):
        fila_ia[f'precio_h{h}']  = round(precios[h], 4)
    for h in range(T_DIA):
        fila_ia[f'pv_h{h}']      = round(pv[h], 4)
    for h in range(T_DIA):
        fila_ia[f'demanda_h{h}'] = round(demanda[h], 4)
    for h in range(T_DIA):
        fila_ia[f'p_bat_h{h}']   = p_bat_dia[h]
    for h in range(T_DIA):
        fila_ia[f'soc_h{h}']     = soc_dia[h]

    dataset_ia.append(fila_ia)

    if (d + 1) % 100 == 0:
        print(f"  Escenario {d+1:4d}/{DIAS} completado  [{split}]")

# ==============================================================
# EXPORTACIÓN CSV
# ==============================================================

df_full = pd.DataFrame(dataset_completo)
df_ia   = pd.DataFrame(dataset_ia)

df_full.to_csv(os.path.join(DIR, "dataset_vpp_completo_T1h.csv"), index=False)
df_ia.to_csv(  os.path.join(DIR, "dataset_vpp_ia_T1h.csv"),       index=False)

# ==============================================================
# RESUMEN FINAL
# ==============================================================

escenarios_ok = DIAS - fallos

print("\n" + "=" * 58)
print("  DATASET T=1h GENERADO")
print("=" * 58)
print(f"  Horizonte por ventana : T = 1 h")
print(f"  Ventanas por día      : 24 (una por hora)")
print(f"  Estrategia SOC        : encadenado hora a hora")
print(f"  Escenarios solicitados: {DIAS}")
print(f"  Resueltos con éxito   : {escenarios_ok}")
print(f"  Fallos omitidos       : {fallos}")
print(f"  Train / Test          : {DIAS_TRAIN} / {DIAS_TEST}")
print()
print("  Archivos generados:")
print("    1) dataset_vpp_completo_T1h.csv")
print("    2) dataset_vpp_ia_T1h.csv")
print()
print("  Estructura dataset IA (igual que T=24h):")
print("    Entradas (73): soc_inicial + precio×24 + pv×24 + demanda×24")
print("    Salidas  (48): p_bat×24 + soc×24")
print("=" * 58)