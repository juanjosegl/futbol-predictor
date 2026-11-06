# futbol-predictor

Pipeline reproducible para predecir resultados de fútbol (1X2 y Over/Under 2.5) y evaluarlos
**contra el mercado de apuestas**:

```
download → build (limpieza + features pre-partido) → train (walk-forward) → evaluate → backtest → predict
```

> **Resumen honesto:** con los datos disponibles, ni el LightGBM ni el Dixon-Coles superan al
> mercado de cierre. Ninguna estrategia de apuestas del backtest es rentable. Ver
> [Resultados](#resultados-de-la-última-ejecución) y [Limitaciones](#limitaciones).

---

## Instalación en Windows (PowerShell)

Requisitos: **Python 3.12** (3.11 también vale) y conexión a internet.

```powershell
# 1. Ir a la carpeta del proyecto
cd C:\ruta\a\futbol-predictor

# 2. Crear el entorno virtual
py -3.12 -m venv .venv

# 3. Instalar dependencias (versiones fijadas) y el propio paquete
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .

# 4. Ejecutar todo el pipeline
.\.venv\Scripts\futbol.exe run-all
```

Si prefieres activar el entorno (`.\.venv\Scripts\Activate.ps1`) y PowerShell lo bloquea por la
política de ejecución, no hace falta cambiarla: basta con llamar directamente a
`.\.venv\Scripts\python.exe` / `.\.venv\Scripts\futbol.exe` como arriba.

`requirements.txt` fija las dependencias directas; `requirements-lock.txt` congela el entorno
completo con el que se probó (`pip install -r requirements-lock.txt` para reproducirlo exacto).

En Windows LightGBM ya incluye OpenMP. (En macOS hace falta `brew install libomp`.)

## Comandos

Todos aceptan `--config ruta\a\config.yaml`. También se pueden lanzar con `python -m futbol <comando>`.

| Comando | Qué hace |
|---|---|
| `futbol download` | Descarga a `data/raw/` con caché. No repite descargas existentes. `--refresh` re-descarga todo; `--refresh-current` solo la temporada en curso y los próximos partidos. |
| `futbol build` | Une las fuentes, mapea nombres de equipos, calcula features y guarda `data/processed/matches.parquet`. |
| `futbol train` | Dixon-Coles walk-forward semanal + LightGBM walk-forward por temporada; guarda predicciones fuera de muestra y el modelo final en `models/`. |
| `futbol evaluate` | Log loss, Brier, accuracy, tablas por temporada/liga y reliability diagrams. |
| `futbol backtest` | Backtest de apuestas (stake fijo y Kelly 25%). `--threshold 0.08` cambia el umbral de valor. |
| `futbol predict FIXTURES.csv` | Probabilidades 1X2 y over 2.5, cuota justa y value bets para próximos partidos. |
| `futbol run-all` | Todo lo anterior en orden. Al final predice `data/fixtures/upcoming.csv` (si football-data publicó próximos partidos) o el ejemplo `data/fixtures/ejemplo_fixtures.csv`. |

Tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

### Formato del CSV de próximos partidos

```csv
date,home,away,odds_h,odds_d,odds_a,league
2026-11-08,Manchester City,Liverpool,1.95,3.80,3.70,E0
2026-11-08,Hull City,Coventry City,,,,
```

- `date`, `home`, `away` son obligatorias. Las cuotas y la liga son opcionales (si falta la liga,
  se infiere por los equipos).
- Los nombres pueden venir en otro formato ("Atlético Madrid", "Bayern München"): se mapean a los
  de football-data con diccionario + fuzzy matching. Los que no se emparejan se avisan en el log.
- Salida: `reports/predictions.csv` con `p_home/p_draw/p_away`, `p_over25`, cuotas justas
  (`fair_*` = 1/p), probabilidades Dixon-Coles, `edge_*` = p·cuota − 1, `value_bet` (1/X/2 si
  el mejor edge supera el umbral) y `kelly_stake_frac` (fracción de bankroll con Kelly 25% y tope).

## Estructura

```
config.yaml                 ligas, temporadas, fuentes, hiperparámetros, backtest
src/futbol/
  cli.py                    CLI (typer)
  config.py, logging_setup.py
  ingest/                   football_data.py, understat.py, clubelo.py, sofascore.py, team_names.py, http.py
  features/                 odds.py (quitar margen), engineering.py (features), build.py (dataset)
  models/                   dixon_coles.py, lgbm.py, training.py (walk-forward)
  evaluation/               metrics.py, evaluate.py, backtest.py, predict.py, plots.py
tests/                      mapeo de nombres, no-leakage, odds, backtest
data/raw/                   caché de descargas      data/processed/  dataset y predicciones
models/                     modelo final            reports/         métricas, gráficos, backtest
logs/                       un log por día
```

## Fuentes de datos y qué pasó con cada una

Estado de la última ejecución en `reports/sources_status.json`.

| Fuente | Estado | Detalle |
|---|---|---|
| **football-data.co.uk** | ✅ Funciona | 8 temporadas (2019/20 – 2026/27 en curso) × 5 ligas = 12.911 partidos. Resultados, tiros, córners y cuotas de apertura y cierre (1X2 y O/U 2.5). |
| **Understat** (xG) | ❌ Caído | `understat.com` responde con una página de mantenimiento ("We will come back soon") y `soccerdata` falla con `JSONDecodeError`. El módulo intenta `soccerdata` → endpoint JSON de la web → formato antiguo embebido en HTML; si todo falla registra un warning y sigue. **Alternativa usada:** football-data añadió columnas de xG (`HxG`, `AxG`) en 2026/27, que se usan cuando no hay Understat. Para temporadas anteriores no hay xG; como sustituto se usan tiros y tiros a puerta (muy correlacionados con xG). Cuando Understat vuelva, `futbol download` + `build` incorporan el xG sin tocar código. |
| **ClubElo** | ❌ Caído | `api.clubelo.com` devuelve 502 / no responde. **Alternativa usada:** un Elo propio calculado desde los resultados de football-data (`features/engineering.py::compute_elo`), siempre activo. Si ClubElo vuelve, se añaden `f_clubelo_*` como features extra. |
| **SofaScore** (opcional) | ⚠️ 403 | Desactivado por defecto (`sources.sofascore: false`). Probado una vez: la API interna responde **403**. El cliente tiene caché en disco, pausa de 2 s y ante 403 registra un warning y continúa. Nada depende de él. |

**Sobre SofaScore y `curl_cffi`:** la especificación pedía usar `curl_cffi` con
`impersonate="chrome"`. No lo implementé: imitar la huella TLS de un navegador para saltarse el
bloqueo anti-bot de un sitio es evadir su detección de bots, y no es algo que vaya a construir. El
cliente usa un HTTP estándar, identificado, y acepta el 403 como respuesta. Si necesitas esos datos,
la vía correcta es un proveedor con API o licencia (p. ej. API-Football, Sportmonks, Opta/StatsBomb).

**Cuotas de Pinnacle:** football-data dejó de publicar las cuotas de Pinnacle a mitad de
2025/26 (no aparecen en 2026/27). Por fila se usa Pinnacle si existe y, si no, el promedio de
mercado (`close_src` / `open_src` indican cuál). Ojo: las cuotas "de apertura" de football-data
son en realidad cuotas recogidas unos días antes del partido (viernes/martes), no la apertura real.

## Features (cero leakage)

Todas las columnas de features empiezan por `f_`. Cada una usa **solo partidos anteriores**:

- **Rolling 5 y 10 partidos** (agrupando por equipo, `shift(1)` y después `rolling`): goles a
  favor/en contra, xG a favor/en contra (si existe), tiros, tiros a puerta, puntos por partido.
  Versión general y versión por condición (el local en sus partidos de local, el visitante en los
  de visitante). Más diferencias local−visitante.
- **Elo propio** antes del partido (`f_elo_h`, `f_elo_a`, diferencia y probabilidad esperada):
  K=20, ventaja local 60, regresión a la media entre temporadas, ascendidos entran por debajo de la media.
- **Días de descanso** de cada equipo y partidos previos en el dataset.
- **Probabilidades Dixon-Coles** walk-forward (reentrenado cada semana solo con partidos anteriores).
- **Probabilidades de apertura sin margen** (`f_mkt_open_*`): opcionales, `features.use_opening_odds`
  (desactivadas por defecto).
- **Las cuotas de cierre nunca son feature**; `feature_columns()` las excluye explícitamente.

`tests/test_no_leakage.py` lo verifica: altera goles, xG, tiros y cuotas de cierre de un partido y
de todos los posteriores, recalcula, y exige que las features de ese partido (y anteriores) no
cambien ni un bit. Lo hace con datos sintéticos, con los datos reales y con el Dixon-Coles
walk-forward. Un test de control confirma que alterar el **pasado** sí cambia las features.

## Modelos y validación

- **Baseline 1 — Dixon-Coles:** Poisson con ataques/defensas por equipo, ventaja local, decaimiento
  temporal (vida media ~1 año) y corrección ρ para marcadores bajos. Reentrenado cada semana.
- **Baseline 2 — Mercado:** probabilidades implícitas de las cuotas de cierre sin margen
  (normalización proporcional; `features.margin_method: shin` para el método de Shin).
- **Principal — LightGBM multiclase** con calibración isotónica (o `sigmoid` = Platt) y un
  LightGBM binario para Over/Under 2.5.
- **Walk-forward por temporada:** para predecir la temporada S se entrena con las temporadas < S−1,
  se hace early stopping y calibración con S−1, y se predice S, que el modelo nunca vio. Con
  `min_train_seasons: 3` las temporadas de test son 2022/23 – 2026/27 (7.534 partidos).
- El **modelo final** para `predict` se entrena con todo menos el último año, que se usa para calibrar.

## Resultados de la última ejecución

Datos hasta el 02/11/2026. 7.534 partidos fuera de muestra (2022/23 – 2026/27).

### 1X2 (menor log loss / Brier = mejor)

| Modelo | Log loss | Brier | Accuracy | Δ log loss vs cierre (IC 95%) |
|---|---|---|---|---|
| LightGBM (calibrado) | 0.9887 | 0.5869 | 52.8% | +0.0228 (+0.017, +0.029) |
| LightGBM (sin calibrar) | 0.9863 | 0.5873 | 52.9% | +0.0205 (+0.016, +0.025) |
| Dixon-Coles | 0.9873 | 0.5884 | 52.7% | +0.0215 (+0.018, +0.025) |
| Mercado apertura | 0.9680 | 0.5755 | 54.1% | +0.0021 (+0.001, +0.004) |
| **Mercado cierre** | **0.9658** | **0.5742** | **54.5%** | — |

### Over/Under 2.5

| Modelo | Log loss | Brier | Accuracy |
|---|---|---|---|
| LightGBM (calibrado) | 0.6839 | 0.2428 | 56.8% |
| LightGBM (sin calibrar) | 0.6783 | 0.2427 | 56.7% |
| Dixon-Coles | 0.6791 | 0.2430 | 57.5% |
| **Mercado cierre** | **0.6666** | **0.2370** | **59.4%** |

**El mercado de cierre es claramente mejor** que todos los modelos en ambos mercados (y en 1X2,
en cada una de las 5 temporadas de test); los intervalos de confianza no tocan el cero. La calibración isotónica empeora un
poco al LightGBM (se mantiene por config para no elegir la variante mirando el test).

### Backtest (umbral 5%, temporadas no vistas)

| Mercado | Modelo | Precio | Estrategia | Apuestas | Beneficio (u) | Yield | CLV medio |
|---|---|---|---|---|---|---|---|
| 1X2 | LightGBM | cierre | stake fijo | 6.192 | −201.9 | −3.3% | — |
| 1X2 | LightGBM | cierre | Kelly 25% | 6.192 | bankroll 100 → 0.05 | −2.7% | — |
| 1X2 | LightGBM | apertura | stake fijo | 5.946 | −286.4 | −4.8% | −3.0% |
| 1X2 | Dixon-Coles | cierre | stake fijo | 5.684 | −741.8 | −13.0% | — |
| O/U 2.5 | LightGBM | cierre | stake fijo | 4.155 | −228.1 | −5.5% | — |
| O/U 2.5 | Dixon-Coles | cierre | stake fijo | 4.012 | −199.0 | −5.0% | — |

Tabla completa (incluidas todas las combinaciones y el desglose por temporada) en
`reports/backtest_summary.csv`. Ninguna combinación gana dinero; con Kelly todas acaban en quiebra.
Algunas temporadas sueltas salen positivas (LightGBM 1X2 a cierre: +3.6% en 2022/23, +0.9% en
2024/25), pero el total es negativo y eso es lo esperable por varianza.

## Cómo interpretar los resultados

- **Log loss / Brier:** cuanto más bajos, mejor. La referencia que importa no es el 0 sino el
  **mercado de cierre**: si un modelo no lo supera, no hay ventaja que explotar.
- **Δ vs cierre con IC 95%:** si el intervalo entero es > 0, el modelo es peor que el mercado con
  significación estadística. Para tener ventaja necesitarías un intervalo entero < 0.
- **Reliability diagram** (`reports/calibration_1x2.png`): los puntos sobre la diagonal indican
  probabilidades bien calibradas. Un modelo puede estar bien calibrado y aun así ser peor que el
  mercado (es menos *preciso*, no está *sesgado*).
- **Yield** = beneficio / total apostado. **ROI** = beneficio / bankroll inicial.
  **Drawdown máximo** = mayor caída desde un pico (en unidades y en %; >100% significa que el
  bankroll pasó a negativo con stake fijo). **`bust`** = el bankroll se agotó.
- **CLV** (closing line value) = cuota tomada / cuota justa de cierre − 1. Es el mejor indicador
  de habilidad: quien bate al mercado consigue sistemáticamente cuotas mejores que el cierre.
  Aquí el CLV medio es −3% a −4%: las apuestas que el modelo cree de valor son justo las que el
  mercado corrige en contra.
- **Muchas apuestas = mala señal:** con umbral 5% el LightGBM encuentra "valor" en ~80% de los
  partidos. Eso no es ventaja: es que el modelo discrepa mucho del mercado y el mercado tiene razón.

## Cómo agregar ligas

Añade una entrada a `leagues` en `config.yaml`:

```yaml
- {code: N1, name: "Eredivisie", soccerdata: "NED-Eredivisie", understat: null, clubelo_country: NED}
```

- `code` es el código de football-data (`E1` Championship, `SP2`, `I2`, `D2`, `F2`, `N1`
  Eredivisie, `P1` Portugal, `B1` Bélgica, `T1` Turquía, `G1` Grecia, `SC0` Escocia...).
- `understat` solo existe para las 5 grandes ligas y Rusia; pon `null` si no hay.
- Después: `futbol run-all`. Si aparecen equipos sin emparejar, revisa
  `reports/unmatched_teams.csv` y añade alias a `MANUAL_ALIASES` en `src/futbol/ingest/team_names.py`.
- Para cambiar el número de temporadas: `seasons.n_seasons`.

## Archivos de salida (`reports/`)

| Archivo | Contenido |
|---|---|
| `metrics_1x2.csv`, `metrics_ou25.csv`, `metrics.json` | Tabla comparativa global |
| `metrics_1x2_by_season.csv`, `metrics_1x2_by_league.csv` | Desgloses |
| `calibration_1x2.png`, `calibration_ou25.png` | Reliability diagrams |
| `backtest_summary.csv/.json`, `backtest_bets.csv` | Backtest (total y por temporada) y cada apuesta |
| `bankroll_{1x2,ou25}_{lgbm,dc}.png` | Curvas de bankroll |
| `walk_forward_folds.csv`, `feature_importance.csv` | Folds y features más útiles (Elo y tiros a puerta encabezan) |
| `data_summary.json`, `sources_status.json` | Tamaño del dataset, cobertura y estado de las fuentes |
| `unmatched_teams.csv`, `fuzzy_matches.csv` | Reporte del mapeo de nombres |
| `predictions.csv` | Salida de `predict` |

## Limitaciones

- **No supera al mercado.** El mercado de cierre agrega información que este modelo no tiene:
  alineaciones, lesiones, rotaciones, motivación, clima, dinero informado.
- **Sin xG histórico** mientras Understat siga caído (solo 2026/27 tiene xG, vía football-data).
- **Solo partidos de liga:** el descanso ignora copas y competiciones europeas, y los ascendidos
  llegan sin historial (se podría añadir la segunda división de cada país).
- **Elo propio** con arranque frío en 2019/20; ClubElo daría ratings con más historia.
- **Cuotas de apertura imperfectas** (football-data recoge "pre-cierre", no la apertura real) y
  sin Pinnacle desde 2025/26: el promedio de mercado tiene más margen y es un benchmark algo más débil.
- **Apostar a cuota de cierre** supone que puedes apostar justo al cierre y sin límites; en la
  práctica las casas limitan a quien gana.
- **Calibración isotónica** con una sola temporada (~1.750 partidos) produce escalones; en partidos
  con poco historial (p. ej. dos recién ascendidos) puede dar probabilidades raras.
- Desarrollado y ejecutado en macOS con Python 3.12; el código usa rutas `pathlib` y UTF-8
  explícito, pero no se ha ejecutado en una máquina Windows.

## Siguientes mejoras recomendadas

1. Recuperar xG (Understat cuando vuelva, o un proveedor con licencia) y añadir segundas divisiones.
2. Modelar la **diferencia con el mercado** en vez del resultado: usar la probabilidad de apertura
   como feature/offset y predecir el movimiento hacia el cierre (CLV) en lugar del 1X2.
3. Elegir calibración e hiperparámetros con validación anidada dentro del entrenamiento (nunca con el test).
4. Información pre-partido real: alineaciones confirmadas, lesiones, competiciones europeas entre semana.
5. Apostar al mejor precio disponible (`Max*`) y estudiar mercados menos eficientes (ligas menores, hándicap asiático).
