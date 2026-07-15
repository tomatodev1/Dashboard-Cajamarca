#!/usr/bin/env python3
"""
Exporta el Google Sheet publico (solo lectura, sin credenciales) del modelo
predictivo de conflictos sociales de Cajamarca a data/data.json para el
dashboard estatico.

El Sheet debe estar compartido como "Cualquier persona con el enlace: Lector".
No requiere ninguna cuenta de servicio ni clave privada -- si en algun momento
se necesitara acceso privado, este script tendria que cambiar a la API de
Sheets con credenciales inyectadas por variable de entorno (nunca hardcodeadas).

Uso:
    python scripts/export_sheet.py
"""
import csv
import io
import json
import re
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oof_predictions import predicciones_oof_serie

SHEET_ID = "1Jb8eP7XTAlxj2S2nFc9jZ_j5C9Hj59EqOZF9VE9SRuo"

TABS = {
    "scoring_modelo": "0",
    "scoring_separadas": "1316050240",
    "features_historicas": "2034826287",
    "calibracion_modelo": "777887404",
    "calendario_criticidad": "778093189",
    # Pestana opcional con contenido editorial (nombre, provincia, descripcion,
    # accion_recomendada, mapa_x, mapa_y) por zona -- se agrega aparte por el
    # equipo. Si no existe todavia, el export sigue funcionando sin ella.
    "zonas_meta": None,
}

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "data.json"

NIVEL_ORDER = ["BAJO", "MEDIO", "MEDIO ALTO", "ALTO"]

# El dashboard solo cubre las 2 zonas con modelo predictivo real. Las otras
# 3 UGT del Sheet original (sin pronostico, solo heuristica) se descartan
# por completo -- no aparecen en ningun lado del tablero. La UGT que en el
# Sheet crudo es "UGT5" se renumera a "UGT2" en el dashboard (ya no hay
# UGT2 viejo con quien confundirse, porque se elimina).
ZONA_ID_REMAP = {"UGT1": "UGT1", "UGT5": "UGT2"}  # id crudo del Sheet -> id del dashboard
ALL_ZONAS = list(dict.fromkeys(ZONA_ID_REMAP.values()))  # ["UGT1", "UGT2"]
MODELO_ZONAS = set(ALL_ZONAS)  # las 2 zonas activas tienen modelo predictivo

# Nombres/provincias reales por UGT (id ya remapeado). Se usan solo si la
# pestana opcional "zonas_meta" del Sheet no trae el dato.
ZONAS_META_FALLBACK = {
    "UGT1": {"nombre": "UGT1 (Proyecto WTP / BECHTL)", "provincia": "Cajamarca"},
    "UGT2": {"nombre": "UGT2 (AISD)", "provincia": "Celendín, Hualgayoc"},
}


def remap_rows(rows, field="ugt"):
    """Renombra el campo de zona segun ZONA_ID_REMAP y descarta las filas de
    zonas que ya no estan en el dashboard (no estan en el mapeo)."""
    out = []
    for row in rows:
        nuevo_id = ZONA_ID_REMAP.get(row.get(field))
        if nuevo_id is None:
            continue
        row = dict(row)
        row[field] = nuevo_id
        out.append(row)
    return out


def remap_ugts_field(rows):
    """Igual que remap_rows pero para la columna 'ugts' de scoring_separadas,
    que trae una lista separada por comas o el literal 'TODAS'."""
    out = []
    for row in rows:
        crudo = (row.get("ugts") or "").strip()
        if crudo == "TODAS":
            zonas = list(ALL_ZONAS)
        else:
            zonas = [ZONA_ID_REMAP[p.strip()] for p in crudo.split(",") if p.strip() in ZONA_ID_REMAP]
        if not zonas:
            continue
        row = dict(row)
        row["ugts"] = ",".join(zonas)
        out.append(row)
    return out


def fetch_csv(gid):
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(raw)))


def fetch_tab(name):
    gid = TABS.get(name)
    if gid is None:
        return []
    try:
        return fetch_csv(gid)
    except Exception as exc:  # pestana opcional (zonas_meta) puede no existir aun
        print(f"[export_sheet] no se pudo leer la pestana '{name}': {exc}")
        return []


def to_float(v, default=None):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except ValueError:
        return default


def con_ewm(serie, span=3):
    """EWM (adjust=False) por zona_id, agrega 'prob_suavizado' sin tocar 'prob'.

    Suavizado SOLO para la linea del grafico -- el modelo real es binario
    semana a semana (class_weight="balanced" con pocos datos produce
    probabilidades polarizadas, ver ROADMAP.md de ML-Cajamarca Fase 8) y
    esa es la senal correcta, no un error. El tooltip debe seguir usando
    'prob' crudo: mentir sobre el valor exacto seria peor que un grafico
    con saltos.
    """
    alpha = 2.0 / (span + 1)
    por_zona = {}
    for p in serie:
        por_zona.setdefault(p["zona_id"], []).append(p)
    for pts in por_zona.values():
        pts.sort(key=lambda p: p["semana"])
        prev = None
        for p in pts:
            if p["prob"] is None:
                p["prob_suavizado"] = prev
                continue
            prev = p["prob"] if prev is None else alpha * p["prob"] + (1 - alpha) * prev
            p["prob_suavizado"] = round(prev, 6)
    return serie


def next_occurrence(mmdd, today):
    """Proxima fecha (>= hoy) de un evento recurrente anual 'MM-DD'."""
    month, day = (int(x) for x in mmdd.split("-"))
    candidate = date(today.year, month, day)
    if candidate < today:
        candidate = date(today.year + 1, month, day)
    return candidate


def compute_proxima_fecha_critica(scoring_separadas, today):
    """Para cada zona, el proximo evento de calendario (recurrente o electoral)."""
    per_zona = {z: [] for z in ALL_ZONAS}
    for row in scoring_separadas:
        zonas_field = (row.get("ugts") or "").strip()
        if zonas_field == "TODAS":
            zonas = ALL_ZONAS
        else:
            zonas = [z.strip() for z in zonas_field.split(",") if z.strip()]

        if row.get("tipo") == "electoral":
            fecha_str = (row.get("fecha") or "").strip()
            if not fecha_str:
                continue
            try:
                fecha = datetime.strptime(fecha_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if fecha < today:
                continue
            entry = {
                "evento": row.get("evento"),
                "tipo": "electoral",
                "criticidad": row.get("criticidad") or None,
                "fecha": fecha.isoformat(),
            }
        else:
            inicio = (row.get("inicio_mes_dia") or "").strip()
            if not inicio:
                continue
            fecha = next_occurrence(inicio, today)
            entry = {
                "evento": row.get("evento"),
                "tipo": "fecha_critica",
                "criticidad": row.get("criticidad") or None,
                "fecha": fecha.isoformat(),
            }

        for z in zonas:
            if z in per_zona:
                per_zona[z].append(entry)

    proximas = {}
    for z, entries in per_zona.items():
        entries.sort(key=lambda e: e["fecha"])
        proximas[z] = entries[0] if entries else None
    return proximas


def latest_features_by_zona(features_historicas):
    latest = {}
    for row in features_historicas:
        z = row.get("ugt")
        semana = row.get("semana")
        if not z or not semana:
            continue
        if z not in latest or semana > latest[z]["semana"]:
            latest[z] = row
    return latest


def build_serie_tiempo(scoring_modelo, features_historicas):
    serie = []
    for row in scoring_modelo:
        prob_crudo = to_float(row["prob"])
        # prob_calibrado es la probabilidad ajustada para leerse como un %
        # real (ver ROADMAP calibracion). Si por algun motivo una fila no la
        # trae aun, usamos el prob crudo como respaldo en vez de romper.
        prob_calibrado = to_float(row.get("prob_calibrado"))
        serie.append({
            "zona_id": row["ugt"],
            "fecha_scoring": row["fecha_scoring"],
            "semana": row["semana"],
            "prob": prob_calibrado if prob_calibrado is not None else prob_crudo,
            "prob_original": prob_crudo,
            "nivel": row["nivel"],
            "origen": "scoring_modelo",
        })
    # de-dup por (zona_id, semana). Un backfill historico masivo comparte un
    # unico fecha_scoring entre cientos de filas; una corrida de scoring en
    # vivo normal solo agrega un puñado de filas (una por zona) con su propio
    # fecha_scoring. Si dos filas compiten por la misma (zona, semana),
    # preferimos la que viene de la corrida "mas chica" (probablemente scoring
    # en vivo real) sobre la de la corrida "mas grande" (backfill masivo); en
    # caso de empate de tamaño, gana el fecha_scoring mas reciente (re-corrida
    # legitima).
    batch_size = {}
    for pt in serie:
        batch_size[pt["fecha_scoring"]] = batch_size.get(pt["fecha_scoring"], 0) + 1

    dedup = {}
    for pt in serie:
        key = (pt["zona_id"], pt["semana"])
        if key not in dedup:
            dedup[key] = pt
            continue
        current = dedup[key]
        pt_is_smaller_batch = batch_size[pt["fecha_scoring"]] < batch_size[current["fecha_scoring"]]
        same_batch_size_and_newer = (
            batch_size[pt["fecha_scoring"]] == batch_size[current["fecha_scoring"]]
            and pt["fecha_scoring"] > current["fecha_scoring"]
        )
        if pt_is_smaller_batch or same_batch_size_and_newer:
            dedup[key] = pt

    # Historico 2024+ real: no hay scoring en vivo tan atras, asi que se
    # rellena con predicciones OOF walk-forward (entrenadas solo con semanas
    # anteriores a cada punto -- sin fuga de futuro). Donde ya existe un
    # score en vivo para esa (zona, semana), ese tiene prioridad.
    for oof in predicciones_oof_serie(features_historicas):
        key = (oof["zona_id"], oof["semana"])
        if key in dedup:
            continue
        dedup[key] = {
            "zona_id": oof["zona_id"],
            "fecha_scoring": None,
            "semana": oof["semana"],
            "prob": oof["prob"],
            "prob_original": None,
            "nivel": None,
            "origen": "oof_backfill",
        }

    out = list(dedup.values())
    out.sort(key=lambda p: (p["zona_id"], p["semana"]))
    return con_ewm(out)


def build_zonas(scoring_modelo, calendario_criticidad, latest_features, proximas, zonas_meta):
    meta_by_id = {row["zona_id"]: row for row in zonas_meta if row.get("zona_id")}

    # ultimo score por zona (track modelo)
    latest_score = {}
    for row in scoring_modelo:
        z = row["ugt"]
        if z not in latest_score or row["semana"] > latest_score[z]["semana"]:
            latest_score[z] = row
    prev_score = {}
    by_zona_scores = {}
    for row in scoring_modelo:
        by_zona_scores.setdefault(row["ugt"], []).append(row)
    for z, rows in by_zona_scores.items():
        rows_sorted = sorted(rows, key=lambda r: r["semana"])
        if len(rows_sorted) >= 2:
            prev_score[z] = rows_sorted[-2]

    # ultimo dato de "separado" por zona
    latest_separado = {}
    for row in calendario_criticidad:
        z = row["ugt"]
        if z not in latest_separado or row["semana"] > latest_separado[z]["semana"]:
            latest_separado[z] = row

    zonas = []
    for z in ALL_ZONAS:
        meta = meta_by_id.get(z, {})
        fallback = ZONAS_META_FALLBACK.get(z, {})
        feat = latest_features.get(z, {})
        base = {
            "zona_id": z,
            "track": "modelo" if z in MODELO_ZONAS else "separado",
            "nombre": meta.get("nombre") or fallback.get("nombre"),
            "provincia": meta.get("provincia") or fallback.get("provincia"),
            "descripcion": meta.get("descripcion") or None,
            "accion_recomendada": meta.get("accion_recomendada") or None,
            "mapa_x": to_float(meta.get("mapa_x")),
            "mapa_y": to_float(meta.get("mapa_y")),
            "eventos_1w": to_float(feat.get("conteo_1w")),
            "eventos_4w": to_float(feat.get("conteo_4w")),
            "delta_actividad_4w": to_float(feat.get("delta_actividad_4w")),
            "dias_desde_ultima_protesta": to_float(feat.get("dias_desde_ultima_protesta")),
            "racha_semanas_protesta": to_float(feat.get("racha_semanas_protesta")),
            "proxima_fecha_critica": proximas.get(z),
        }
        if z in MODELO_ZONAS:
            cur = latest_score.get(z)
            prev = prev_score.get(z)

            def prob_legible(row):
                if row is None:
                    return None
                calibrado = to_float(row.get("prob_calibrado"))
                return calibrado if calibrado is not None else to_float(row["prob"])

            prob_actual = prob_legible(cur)
            prob_prev = prob_legible(prev)
            base.update({
                "prob_actual": prob_actual,
                "prob_original": to_float(cur["prob"]) if cur else None,
                "nivel_riesgo": cur["nivel"] if cur else None,
                "semana": cur["semana"] if cur else None,
                "tendencia_delta": (
                    round(prob_actual - prob_prev, 4)
                    if prob_actual is not None and prob_prev is not None else None
                ),
            })
        else:
            sep = latest_separado.get(z)
            base.update({
                "prob_actual": None,
                "nivel_riesgo": None,
                "semana": sep["semana"] if sep else None,
                "tasa_historica": to_float(sep["tasa_historica"]) if sep else None,
                "oefa_denuncias_4w": to_float(sep["oefa_denuncias_4w"]) if sep else None,
            })
        zonas.append(base)
    return zonas


def main():
    scoring_modelo = remap_rows(fetch_tab("scoring_modelo"))
    scoring_separadas = remap_ugts_field(fetch_tab("scoring_separadas"))
    features_historicas = remap_rows(fetch_tab("features_historicas"))
    calibracion_modelo = fetch_tab("calibracion_modelo")  # agregado por nivel, no por zona
    calendario_criticidad = remap_rows(fetch_tab("calendario_criticidad"))
    zonas_meta = remap_rows(fetch_tab("zonas_meta"), field="zona_id")

    today = date.today()
    latest_features = latest_features_by_zona(features_historicas)
    proximas = compute_proxima_fecha_critica(scoring_separadas, today)
    serie_tiempo = build_serie_tiempo(scoring_modelo, features_historicas)
    zonas = build_zonas(scoring_modelo, calendario_criticidad, latest_features, proximas, zonas_meta)

    fechas_scoring = [r["fecha_scoring"] for r in scoring_modelo if r.get("fecha_scoring")]
    semanas = [r["semana"] for r in scoring_modelo if r.get("semana")]

    calibracion = [
        {
            "nivel": row["nivel"],
            "n": int(to_float(row["n"], 0)),
            "tasa_acierto": to_float(row["tasa_acierto"]),
        }
        for row in calibracion_modelo
    ]
    calibracion.sort(key=lambda r: NIVEL_ORDER.index(r["nivel"]) if r["nivel"] in NIVEL_ORDER else 99)

    data = {
        "meta": {
            "generado_en": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fuente": f"Google Sheets {SHEET_ID} (scoring_modelo, scoring_separadas, "
                      f"features_historicas, calibracion_modelo, calendario_criticidad)",
            "fecha_scoring_mas_reciente": max(fechas_scoring) if fechas_scoring else None,
            "semana_mas_reciente": max(semanas) if semanas else None,
        },
        "zonas": zonas,
        "serie_tiempo": serie_tiempo,
        "calibracion_modelo": calibracion,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK -> {OUT_PATH} ({len(zonas)} zonas, {len(serie_tiempo)} puntos de serie)")


if __name__ == "__main__":
    main()
