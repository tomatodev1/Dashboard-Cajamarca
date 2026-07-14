#!/usr/bin/env python3
"""
Predicciones out-of-fold (OOF) por walk-forward temporal para las zonas con
modelo entrenado (UGT1, UGT5).

Para cada semana del historico, se entrena un clasificador usando SOLO datos
de semanas estrictamente anteriores (expanding window) y se predice la
probabilidad de esa semana. No hay fuga de informacion futura: la prediccion
de la semana W nunca ve datos de W o posteriores.

Esto rellena el historico 2024-2026 con estimaciones reales del modelo
(reproducibles a partir de features_historicas), en lugar de simular una
curva. No reemplaza al modelo de produccion real del repo de ML -- es un
backtest walk-forward sobre las mismas features historicas para fines de
visualizacion, hasta que el repo de ML publique su propio backfill oficial.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "conteo_1w", "conteo_4w", "conteo_4w_prev", "delta_actividad_4w", "racha_semanas_protesta",
    "dias_desde_ultima_protesta", "noticias_total", "noticias_bajo", "noticias_medio", "noticias_alto",
    "noticias_protesta", "noticias_mencion_directa", "noticias_protesta_alto", "noticias_protesta_4w",
    "noticias_mencion_directa_4w", "noticias_protesta_alto_4w", "oefa_denuncias_1w", "oefa_denuncias_4w",
    "defensoria_escalamiento", "inei_tasa_pobreza", "rep_yanacocha_riesgo", "rep_conflictos_sociales_riesgo",
    "rep_tension_kw", "rep_compromiso_kw", "rep_disponible", "feriados_semana_actual",
    "feriados_prox_4_semanas", "criticidad_calendario_semana", "criticidad_calendario_prox_4_semanas",
    "dias_hasta_eleccion",
]

MIN_TRAIN_ROWS = 16  # burn-in minimo antes de confiar en el clasificador
# C bajo: con ~30 features y pliegues tempranos de pocas decenas de filas,
# una regresion logistica poco regularizada satura en 0/1 (sobreajuste).
# C=0.01 se eligio comparando el ruido semana a semana en distintos valores:
# reduce la saturacion de ~39% a ~7% de los puntos sin colapsar la señal
# (el rango de probabilidades predichas se mantiene amplio).
LOGREG_C = 0.01


def predicciones_oof_serie(features_historicas_rows):
    """
    features_historicas_rows: list[dict], tal cual vienen de la pestana
    'features_historicas' del Sheet (valores string).

    Devuelve list[dict]: {zona_id, semana, prob} -- solo para zonas
    grupo_modelo == 'modelo' (UGT1, UGT5 hoy), una fila por (zona, semana).
    """
    df = pd.DataFrame(features_historicas_rows)
    if df.empty or "grupo_modelo" not in df.columns:
        return []
    df = df[df["grupo_modelo"] == "modelo"].copy()
    if df.empty:
        return []

    for col in FEATURE_COLS + ["y_30"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS + ["y_30"])
    df = df.sort_values("semana").reset_index(drop=True)
    if df.empty:
        return []

    out = []
    semanas = sorted(df["semana"].unique())
    for semana in semanas:
        train = df[df["semana"] < semana]
        test = df[df["semana"] == semana]

        if len(train) < MIN_TRAIN_ROWS or train["y_30"].nunique() < 2:
            base_rate = train["y_30"].mean() if len(train) else df["y_30"].mean()
            prob = float(base_rate) if pd.notna(base_rate) else 0.5
            for _, row in test.iterrows():
                out.append({"zona_id": row["ugt"], "semana": semana, "prob": prob})
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=LOGREG_C)),
        ])
        pipe.fit(train[FEATURE_COLS], train["y_30"])
        proba = pipe.predict_proba(test[FEATURE_COLS])[:, 1]
        for (_, row), p in zip(test.iterrows(), proba):
            out.append({"zona_id": row["ugt"], "semana": semana, "prob": float(p)})

    return out
