# futbol-predictor

Pipeline reproducible para predecir resultados de fútbol (1X2 y Over/Under 2.5) y evaluarlos
**contra el mercado de apuestas**:

```
download → build (limpieza + features pre-partido) → train (walk-forward) → evaluate → backtest → predict
```

> **¿Solo quieres usarlo?** Lee [GUIA.md](GUIA.md): qué mirar cada día, cómo anotar apuestas y cómo medir.
>
> **Resumen honesto:** con los datos disponibles ningún modelo supera al mercado de cierre. El más
> preciso (regresión logística anclada a las cuotas de apertura) **empata** con el mercado de
> apertura. Ninguna estrategia de apuestas del backtest es rentable. Ver
> [Resultados](#resultados-de-la-última-ejecución), [Guía para empezar](#guía-para-empezar-hoy) y
> [Limitaciones](#limitaciones).

---

## Instalación en macOS

```bash
cd ~/Documents/University/futbol-predictor
brew install libomp                      # LightGBM lo necesita en macOS
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
.venv/bin/futbol run-all
```

(O `source .venv/bin/activate` y después simplemente `futbol run-all`.)

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

Todos aceptan `--config ruta/a/config.yaml`. También se pueden lanzar con `python -m futbol <comando>`.

| Comando | Qué hace |
|---|---|
| `futbol download` | Descarga a `data/raw/` con caché. `--refresh` re-descarga todo; `--refresh-current` solo la temporada en curso y los próximos partidos. Si football-data.co.uk no responde, avisa y usa la caché; con token, completa con football-data.org. |
| `futbol build` | Une las fuentes, mapea nombres de equipos, calcula features y guarda `data/processed/matches.parquet`. |
| `futbol train` | Walk-forward por temporada de Dixon-Coles, LightGBM y las dos regresiones logísticas; guarda predicciones fuera de muestra y el modelo final. |
| `futbol evaluate` | Log loss, Brier, accuracy (con IC 95% vs el mercado), tablas por temporada/liga y reliability diagrams. |
| `futbol backtest` | Backtest de apuestas (stake fijo y Kelly 25%) para cada modelo. `--threshold 0.08` cambia el umbral de valor. |
| `futbol exam DESDE [HASTA]` | **Examen**: predice partidos YA JUGADOS entrenando solo con lo anterior a `DESDE` y compara con los resultados reales y con el mercado. `--liga E0` filtra. |
| `futbol predict FIXTURES.csv` | Probabilidades 1X2 y O/U 2.5, cuota justa y value bets. Cada predicción se guarda en `reports/ledger.csv` (`--no-ledger` para no guardarla). |
| `futbol settle` | Liquida el ledger con los resultados reales (aciertos, log loss vs mercado, beneficio de las apuestas) y actualiza `reports/seguimiento.xlsx`. Si no hay datos nuevos, acepta marcadores escritos a mano en el ledger (`fthg`, `ftag`). |
| `futbol informe` | Regenera `reports/seguimiento.xlsx` (predicciones, aciertos/fallos y análisis) desde el ledger. |
| `futbol actualizar` | **Rutina semanal en un solo comando:** descarga → build → train → settle → predict de `data/fixtures/upcoming.csv` (o `--fixtures ARCHIVO.csv`) → documento de seguimiento. `--no-train` para ir más rápido. |
| `futbol hoy` | **Pronósticos multideporte del día en un solo comando** (ver [Multideporte](#multideporte-futbol-hoy)): actualiza datos, liquida lo pendiente, predice y actualiza el ledger y el Excel. `--dias 3`, `--top 40`, `--liga nba --liga atp`, `--no-ledger`. |
| `futbol ver` | Abre `reports/hoy.xlsx` (selección del día) y `reports/seguimiento.xlsx` (aciertos, fallos y apuestas). |
| `futbol apuesta EQUIPO --cuota 1.35 --monto 10000` | Anota una apuesta simple sobre un partido pendiente (por defecto, al pronóstico). `--papel` = simulada, sin dinero. Se liquida sola. |
| `futbol programar` | Ejecuta `futbol hoy` solo cada día (macOS, 07:00 y 17:00; `--hora`, `--estado`, `--probar`, `--quitar`). |
| `futbol deportes-descargar` | Descarga/actualiza el historial de las ligas de `deportes.ligas` (ESPN). `--refresh` para rehacerlo. |
| `futbol deportes-evaluar` | Valida el modelo multideporte en años no vistos (`reports/deportes_metricas.csv`, `deportes_calibracion.csv`). |
| `futbol run-all` | download → build → train → evaluate → backtest → examen de las últimas 4 semanas → predict (`data/fixtures/upcoming.csv` o el ejemplo). |

Tests: `.venv/bin/python -m pytest`

### Formato del CSV de próximos partidos

```csv
date,home,away,odds_h,odds_d,odds_a,odds_over,odds_under,league
2026-11-08,Manchester City,Liverpool,1.95,3.80,3.70,1.72,2.10,E0
2026-11-08,Hull City,Coventry City,,,,,,
```

- `date`, `home`, `away` son obligatorias. Las cuotas y la liga son opcionales (si falta la liga,
  se infiere por los equipos).
- **Pon las cuotas de tu casa de apuestas en el momento de predecir.** El modelo principal
  (`model.main: rlm`) las usa como información (quitando el margen) y calcula el valor esperado de
  apostar a esas cuotas. Sin cuotas, el partido se predice con el LightGBM (columna `modelo`).
- Los nombres pueden venir en otro formato ("Atlético Madrid", "Bayern München"): se mapean a los
  de football-data con diccionario + fuzzy matching. Los que no se emparejan se avisan en el log.
- Salida: `reports/predictions.csv` con el **pronóstico** (`pick` = resultado más probable, `p_pick` su
  probabilidad, `confianza` Alta/Media/Baja según `tracking.*` en `config.yaml`, `cuota_pick`; y
  `pick_ou`/`p_pick_ou` para Over/Under), `p_home/p_draw/p_away`, `p_over25`, cuotas justas
  (`fair_*` = 1/p), probabilidades Dixon-Coles, `edge_*` = p·cuota − 1, `value_bet` (1/X/2 si
  el mejor edge supera el umbral), `kelly_stake_frac` (fracción de bankroll con Kelly 25% y tope) y
  lo mismo para Over/Under (`value_bet_ou`, `kelly_stake_frac_ou`). También salen las probabilidades
  del mercado (`mkt_p_*`), del LightGBM y del Dixon-Coles para comparar.

## Estructura

```
config.yaml                 ligas, temporadas, fuentes, hiperparámetros, backtest
src/futbol/
  cli.py                    CLI (typer)
  config.py, logging_setup.py
  ingest/                   football_data.py, football_data_org.py, understat.py, clubelo.py, sofascore.py, team_names.py, http.py
  features/                 odds.py (quitar margen), engineering.py (features), build.py (dataset)
  models/                   dixon_coles.py, lgbm.py, reglog.py, training.py (walk-forward)
  deportes/                 espn.py (datos), modelo.py (Elo + forma + reg. logística), hoy.py (pronósticos del día)
  evaluation/               metrics.py, evaluate.py, backtest.py, exam.py, ledger.py, informe.py, predict.py, plots.py
tests/                      mapeo de nombres, no-leakage, odds, backtest, descarga offline, ledger, informe, reg. logística
data/raw/                   caché de descargas      data/processed/  dataset y predicciones
models/                     modelo final            reports/         métricas, gráficos, backtest
logs/                       un log por día
```

## Multideporte (`futbol hoy`)

```bash
.venv/bin/futbol hoy
```

Un solo comando que, para **fútbol (15 ligas y copas, incluida la colombiana), NBA, NHL, MLB, NFL y tenis
ATP/WTA**:

1. Descarga los resultados nuevos, el calendario y las cuotas actuales (la primera vez baja todo el
   historial desde 2016: unos 5 minutos; después, segundos).
2. Liquida las predicciones anteriores del ledger con los resultados reales.
3. Predice todos los partidos de hoy y de los próximos `deportes.dias_adelante` días y los ordena de más a
   menos probable (`reports/hoy.csv`), con su nivel de confianza y el % de acierto histórico de ese nivel.
4. Marca la **selección del día** (`reports/hoy.xlsx`, primera hoja): los pocos partidos pensados para
   apostar, siempre apuesta simple al ganador. Entra un partido solo si la probabilidad es ≥ 70%, ese nivel
   de confianza acertó ≥ 70% en esa liga en años no vistos (con ≥ 100 casos), no es empate y, si hay cuotas,
   el mercado tiene al mismo favorito; como mucho 5 por día (`seleccion.*` en `config.yaml`).
5. Registra cada predicción (válida solo si se hizo antes de que empezara el partido) y actualiza
   `reports/seguimiento.xlsx`, con las hojas **Mis apuestas**, **Selección**, **Modelo vs mercado** (modelo
   local solo, mercado y combinación sobre los mismos partidos) y **Por deporte**.

**Datos:** API pública de ESPN (`site.api.espn.com`, JSON, sin clave), con un cliente HTTP normal e
identificado, caché en disco y pausa entre peticiones. Da marcadores históricos, calendario y, para los
partidos próximos, cuotas (moneyline y total). **No da cuotas históricas**, así que la comparación con el
mercado en estos deportes solo se puede medir de aquí en adelante, con el ledger.

**Modelo** (`deportes/modelo.py`), el mismo para todos los deportes, entrenado liga por liga:

- **Elo** con K y peso del margen de victoria ajustados por liga, ventaja de campo estimada de los datos,
  regresión a la media entre temporadas y aprendizaje más rápido para equipos/jugadores nuevos.
- **Estado en el que llegan:** rendimiento y margen medio en los últimos 5/10 partidos, días de descanso,
  partidos en los últimos 7 días (cansancio, *back-to-back*), historial entre ambos y experiencia.
- Regresión logística sobre esas variables (multinomial si hay empate). Todo usa solo partidos anteriores
  (hay un test que lo comprueba) y los hiperparámetros se eligen sin mirar los años de validación.
- **Mercado:** cuando ESPN publica cuotas, la probabilidad final combina modelo y mercado
  (`deportes.peso_mercado`, 0.7 por defecto). El mercado ya incorpora lesiones, alineaciones y noticias
  que el modelo no ve; el peso es una decisión de diseño (apoyada en que, en el pipeline de fútbol, el
  mercado es claramente mejor que cualquier modelo de estadísticas), no un valor validado. El ledger guarda
  por separado `mod_p_*` (modelo) y `mkt_p_*` (mercado) para poder comprobarlo con el tiempo.

**Añadir una liga o deporte:** una línea en `deportes.ligas` de `config.yaml` con la ruta de ESPN
(`sport`/`league`, p. ej. `soccer`/`chi.1`, `basketball`/`wnba`, `football`/`college-football`) y después
`futbol hoy`. Sirve cualquier competición de dos contrincantes con marcador.

### Resultados de la validación multideporte

Walk-forward 2024-2026: cada año se predice con un modelo entrenado solo con años anteriores
(`futbol deportes-evaluar`; tabla completa en `reports/deportes_metricas.csv`).

| Liga | Deporte | Partidos de test | % acierto | Si siempre gana el local | Confianza alta: % acierto (partidos) |
|---|---|---|---|---|---|
| NBA | Baloncesto | 3.654 | 66.8% | 54.5% | 79.3% (1.346) |
| NFL | Fútbol americano | 737 | 66.3% | 53.3% | 77.8% (189) |
| WTA (individual femenino) | Tenis | 17.383 | 65.1% | 49.7% | 78.5% (5.128) |
| ATP (individual masculino) | Tenis | 11.787 | 63.2% | 50.8% | 76.7% (3.440) |
| Champions League | Fútbol | 479 | 59.7% | 49.5% | 74.6% (67) |
| NHL | Hockey | 3.927 | 57.2% | 54.1% | 73.7% (327) |
| MLB | Béisbol | 7.421 | 56.0% | 53.1% | 73.8% (149) |
| Primeira Liga | Fútbol | 881 | 55.0% | 42.0% | 79.8% (243) |
| La Liga | Fútbol | 1.079 | 53.8% | 45.9% | 85.0% (120) |
| Ligue 1 | Fútbol | 854 | 53.5% | 44.6% | 75.0% (68) |
| Serie A | Fútbol | 1.070 | 53.1% | 39.8% | 70.8% (130) |
| Eredivisie | Fútbol | 891 | 52.4% | 43.2% | 75.1% (177) |
| Bundesliga | Fútbol | 856 | 52.3% | 41.8% | 75.9% (83) |
| Premier League | Fútbol | 1.044 | 51.0% | 42.1% | 74.3% (113) |
| Liga MX | Fútbol | 990 | 50.4% | 45.6% | 75.9% (58) |
| Brasileirão | Fútbol | 1.099 | 49.9% | 48.8% | 72.5% (40) |
| Copa Libertadores | Fútbol | 463 | 49.0% | 46.2% | 65.2% (46) |
| Liga colombiana | Fútbol | 1.267 | 48.3% | 46.2% | 84.9% (33) |
| MLS | Fútbol | 1.575 | 46.5% | 45.1% | — (3) |
| Championship | Fútbol | 1.551 | 45.6% | 43.5% | — (21) |
| Liga argentina | Fútbol | 1.383 | 43.7% | 42.4% | — (3) |

- **Los deportes más predecibles** son baloncesto, fútbol americano y tenis (~63–67% de acierto, sin empate
  posible). Hockey y béisbol están cerca de la moneda al aire (~56–57%); en fútbol el empate baja el acierto
  al 44–55% según la liga.
- **La confianza alta acierta 71–85%** en las ligas con más de 50 casos, y las probabilidades están bien
  calibradas: sumando todas las ligas, los pronósticos de confianza alta decían 77.6% de media y acertaron
  el 77.6% (11.784 partidos). Se puede confiar en el número, no en que sea rentable.
- En tenis "siempre gana el local" no aplica (no hay local): es ~50% por construcción.
- Estas cifras son del modelo solo (sin cuotas). **No hay comparación con el mercado** porque ESPN no da
  cuotas históricas; por la experiencia del pipeline de fútbol, lo esperable es que el mercado sea algo
  mejor. El ledger lo irá midiendo con datos reales.

## Fuentes de datos y qué pasó con cada una

Estado de la última ejecución en `reports/sources_status.json`.

| Fuente | Estado | Detalle |
|---|---|---|
| **football-data.org** (respaldo) | 🔑 Token gratis | API oficial con plan gratuito. Si football-data.co.uk está bloqueado, añade los resultados recientes (sin tiros ni cuotas) y genera `upcoming.csv` con los próximos partidos. Regístrate en <https://www.football-data.org/client/register> y crea un archivo `.env` en la raíz con `FOOTBALL_DATA_ORG_TOKEN=tu_token` (está en `.gitignore`). |
| **football-data.co.uk** | ✅ Funciona (⚠️ bloqueado en algunas redes de Colombia) | 8 temporadas (2019/20 – 2026/27 en curso) × 5 ligas = 12.911 partidos. Resultados, tiros, córners y cuotas de apertura y cierre (1X2 y O/U 2.5). |
| **Understat** (xG) | ❌ Caído | `understat.com` responde con una página de mantenimiento ("We will come back soon") y `soccerdata` falla con `JSONDecodeError`. El módulo intenta `soccerdata` → endpoint JSON de la web → formato antiguo embebido en HTML; si todo falla registra un warning y sigue. **Alternativa usada:** football-data añadió columnas de xG (`HxG`, `AxG`) en 2026/27, que se usan cuando no hay Understat. Para temporadas anteriores no hay xG; como sustituto se usan tiros y tiros a puerta (muy correlacionados con xG). Cuando Understat vuelva, `futbol download` + `build` incorporan el xG sin tocar código. |
| **ClubElo** | ❌ Caído | `api.clubelo.com` devuelve 502 / no responde. **Alternativa usada:** un Elo propio calculado desde los resultados de football-data (`features/engineering.py::compute_elo`), siempre activo. Si ClubElo vuelve, se añaden `f_clubelo_*` como features extra. |
| **SofaScore** (opcional) | ⚠️ 403 | Desactivado por defecto (`sources.sofascore: false`). Probado una vez: la API interna responde **403**. El cliente tiene caché en disco, pausa de 2 s y ante 403 registra un warning y continúa. Nada depende de él. |

### Si football-data está bloqueado

En Colombia algunos proveedores de internet bloquean `football-data.co.uk` por orden de
Coljuegos (el sitio publica cuotas). Se nota porque `http://www.football-data.co.uk` redirige a
`coljuegos.gov.co` y por HTTPS la conexión no responde. En ese caso `futbol download` avisa
enseguida y sigue con la caché. Para actualizar los datos:

1. Descarga los CSV desde otra red (datos móviles, la red de la universidad...):
   `https://www.football-data.co.uk/mmz4281/2627/E0.csv` (y `SP1`, `I1`, `D1`, `F1`), y
   `https://www.football-data.co.uk/fixtures.csv`.
2. Cópialos en `data/raw/football_data/` con el nombre `{liga}_{temporada}.csv`
   (p. ej. `E0_2627.csv`) y `fixtures.csv`.
3. Ejecuta `futbol run-all` (sin `--refresh`): usa los archivos copiados.

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

- **Dixon-Coles:** **regresión de Poisson** con ataques/defensas por equipo, ventaja local,
  decaimiento temporal (vida media ~1 año) y corrección ρ para marcadores bajos. Reentrenado cada semana.
- **Regresión logística (stats)** (`rl`): multinomial (1X2) y binaria (O/U), con las mismas features
  pre-partido que el LightGBM. Coeficientes en `reports/reglog_stats_coeficientes.csv`.
- **Regresión logística + mercado** (`rlm`, **modelo principal**): log-probabilidades de apertura sin
  margen + Elo + Dixon-Coles. Coeficientes en `reports/reglog_mercado_coeficientes.csv`.
- **LightGBM** multiclase y binario (gradient boosting) con calibración Platt (`sigmoid`). La
  isotónica se descartó: con una sola temporada de calibración daba probabilidades de 0% o 100%.
- **Benchmarks de mercado:** probabilidades implícitas de las cuotas de apertura y cierre sin margen.
- **Walk-forward por temporada:** para predecir la temporada S se entrena con las temporadas < S
  (el LightGBM usa S−1 para early stopping y calibración) y se predice S, que el modelo nunca vio.
  Test: 2022/23 – 2026/27 (7.534 partidos).
- **Por qué no se añaden más estadísticas al modelo con cuotas:** en los experimentos, añadir las ~75
  features de forma a la regresión con cuotas *empeoró* el log loss (0.9758 vs 0.9683). El mercado
  ya incorpora esa información; añadirla solo añade ruido.

## Resultados de la última ejecución

Datos hasta el 02/11/2026. 7.534 partidos fuera de muestra (2022/23 – 2026/27).

### 1X2 (menor log loss / Brier = mejor)

| Modelo | Log loss | Brier | Accuracy | Δ log loss vs cierre (IC 95%) |
|---|---|---|---|---|
| LightGBM (Platt) | 0.9852 | 0.5870 | 52.6% | +0.0194 (+0.015, +0.024) |
| Dixon-Coles | 0.9873 | 0.5884 | 52.7% | +0.0215 (+0.018, +0.025) |
| Regresión logística (stats) | 0.9875 | 0.5883 | 52.4% | +0.0216 (+0.017, +0.026) |
| **Regresión logística + mercado** | **0.9683** | **0.5757** | **53.9%** | +0.0025 (+0.001, +0.004) |
| Mercado apertura | 0.9680 | 0.5755 | 54.0% | +0.0021 (+0.001, +0.004) |
| Mercado cierre | 0.9658 | 0.5742 | 54.5% | — |

### Over/Under 2.5

| Modelo | Log loss | Brier | Accuracy |
|---|---|---|---|
| LightGBM (Platt) | 0.6775 | 0.2423 | 57.4% |
| Dixon-Coles | 0.6791 | 0.2430 | 57.5% |
| Regresión logística (stats) | 0.6792 | 0.2431 | 56.6% |
| **Regresión logística + mercado** | **0.6692** | **0.2383** | **59.2%** |
| Mercado apertura | 0.6692 | 0.2383 | 59.4% |
| Mercado cierre | 0.6666 | 0.2370 | 59.4% |

**Lectura:** con solo estadísticas, los tres métodos (Poisson, regresión logística, gradient boosting)
llegan al mismo techo (~0.987) y quedan claramente por detrás del mercado. Anclado a las cuotas, el
modelo empata con el mercado de apertura, pero no lo supera, y el mercado de cierre es todavía mejor.

### Backtest (umbral 5%, stake fijo, temporadas no vistas)

| Mercado | Modelo | Precio | Apuestas | Beneficio (u) | Yield |
|---|---|---|---|---|---|
| 1X2 | LightGBM | apertura | 5.733 | −418.5 | −7.3% |
| 1X2 | Dixon-Coles | apertura | 5.162 | −670.4 | −13.0% |
| 1X2 | Reg. logística (stats) | apertura | 5.782 | −427.0 | −7.4% |
| 1X2 | Reg. logística + mercado | apertura | 572 | −107.0 | −18.7% |
| O/U 2.5 | LightGBM | apertura | 3.469 | −219.9 | −6.3% |
| O/U 2.5 | Dixon-Coles | apertura | 3.581 | −206.9 | −5.8% |

Ninguna combinación gana dinero (tabla completa en `reports/backtest_summary.csv`). Las pocas
"value bets" del modelo con mercado son las peores: cuando discrepa del mercado, se equivoca él.

### Examen de octubre 2026 (`futbol exam 2026-10-01 2026-11-02`)

202 partidos, modelo entrenado solo con datos hasta el 30/09/2026: el modelo principal acierta
**121/202 (59.9%)**, el mercado de apertura 60.4%. Log loss 0.9280 vs 0.9277 del mercado
(diferencia no significativa). 0 apuestas con valor > 5%.

### ¿Y si apuesto solo a lo más seguro?

Pronóstico = resultado más probable según el modelo principal, apostando 1 unidad a la cuota de
apertura. 7.534 partidos fuera de muestra (2022/23 – 2026/27):

| Probabilidad del pronóstico | Partidos | % acierto | Cuota media | Yield |
|---|---|---|---|---|
| < 40% | 1.473 | 38.2% | 2.54 | −3.2% |
| 40–50% | 2.306 | 46.7% | 2.15 | +0.1% |
| 50–60% | 1.724 | 55.8% | 1.76 | −1.9% |
| 60–70% | 1.109 | 66.8% | 1.50 | −0.2% |
| 70–80% | 666 | 74.8% | 1.31 | −2.0% |
| > 80% | 256 | 86.7% | 1.16 | +0.7% |
| **Todos** | **7.534** | **53.9%** | | **−1.2%** |

El modelo está **bien calibrado**: cuando dice 75%, acierta ~75%. Pero el yield ronda el 0 o es
negativo en todos los tramos, porque la cuota ya descuenta esa probabilidad: un favorito al 80% paga
~1.16 y un solo fallo se come cinco aciertos. Estas cuotas son en su mayoría de Pinnacle (margen ~3%);
con el margen de una casa local (5-8%) el yield esperado baja a −3% … −6%. En octubre de 2026 la misma
estrategia dio +7.6% ± 12.9% en 202 partidos: dentro del margen de error, es decir, suerte.

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

## Guía para empezar hoy

**Lo más simple (multideporte):** ejecuta `futbol hoy` una vez al día, antes de los partidos. Liquida lo de
ayer, predice lo de hoy y mañana, y mantiene al día `reports/seguimiento.xlsx`. Si apuestas, anota
`apuesta_real`, `cuota_real` y `stake_real` en `reports/ledger.csv`.

**Pipeline de fútbol con cuotas históricas** (5 grandes ligas, backtest contra el mercado):

Rutina de cada semana (el documento que vas a mirar es **`reports/seguimiento.xlsx`**):

1. **Antes de la jornada** — lista de partidos con cuotas en `data/fixtures/upcoming.csv`:
   - con football-data.co.uk accesible, `futbol actualizar` la crea sola (con cuotas promedio del mercado);
   - con el token de football-data.org, la crea con los partidos pero **sin cuotas**: ábrela y escribe las
     de tu casa (se conservan aunque vuelvas a descargar);
   - sin ninguna de las dos, créala a mano con el formato de arriba.
2. **`futbol actualizar`** — descarga datos, reentrena, liquida las predicciones anteriores, predice los
   partidos de `upcoming.csv` (de más a menos seguro) y actualiza el ledger y el Excel.
   Las predicciones hechas el mismo día del partido (o después) se guardan pero **no cuentan** en las métricas.
3. **Si apuestas**, anota en `reports/ledger.csv` las columnas `apuesta_real` (1/X/2/Over/Under),
   `cuota_real` y `stake_real` del partido.
4. **Después de la jornada** — otra vez `futbol actualizar` (o `futbol settle`): rellena marcadores, marca
   cada predicción como Acertó/Falló y recalcula el análisis. Si no hay datos nuevos (sitio bloqueado y sin
   token), escribe los goles en `fthg` y `ftag` del ledger y ejecuta `futbol settle`.

`reports/seguimiento.xlsx` trae: **Resumen** (% de acierto con intervalo de confianza, comparación con el
mercado, beneficio de apostar 1 u a cada pronóstico, apuestas reales), **Predicciones** (una fila por
partido, verde/rojo/amarillo), **Por confianza**, **Por liga**, **Por pronóstico**, **Por semana** y
**Over-Under 2.5**. El mismo documento se genera para los exámenes sobre partidos pasados
(`futbol exam 2026-10-01` → `reports/examen_*.xlsx`), útil para clase.

### Si vas a apostar dinero real

- La evidencia de este proyecto es que **el valor esperado es negativo**: ningún modelo bate al
  mercado y el margen de la casa (5-8% en casas locales) hace que apostar a ciegas pierda dinero.
- Haz primero **varias semanas de paper trading** (pasos 3 y 5 sin apostar). Con <200 partidos el
  azar domina cualquier resultado; con 20 apuestas, ganar o perder no demuestra nada.
- Si aun así apuestas: solo dinero que puedas perder entero, stake fijo pequeño (p. ej. 1% del bankroll),
  nunca subir el stake para recuperar pérdidas, y solo en casas autorizadas por Coljuegos.
- El indicador que importa es el **CLV**: si las cuotas que tomas son sistemáticamente mejores que
  las de cierre, puede haber ventaja; si no, no la hay, aunque una racha sea positiva.

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
| `hoy.xlsx`, `hoy.csv` | Selección del día (primera hoja) y todos los pronósticos de `futbol hoy`, de más a menos probable |
| `deportes_metricas.csv`, `deportes_calibracion.csv` | Validación del modelo multideporte por liga y por nivel de confianza |
| `ledger.csv`, `ledger_resumen.json` | Registro de predicciones y su liquidación (`settle`) |
| `seguimiento.xlsx` | Documento de seguimiento: predicciones, aciertos/fallos y análisis (se actualiza solo) |
| `examen_<desde>_<hasta>.csv/.json/.xlsx` | Examen sobre partidos pasados (el `.xlsx` con el mismo análisis) |
| `reglog_*_coeficientes.csv` | Coeficientes de las regresiones logísticas |

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
- **Selección con el test:** el cambio de calibración (isotónica → Platt) y la elección del modelo
  principal se hicieron mirando los resultados walk-forward; las cifras de esos modelos pueden ser
  algo optimistas. El examen y el ledger (datos nuevos) son la medida limpia.
- **Partidos de football-data.org** (respaldo) no traen tiros ni cuotas: actualizan Elo, forma y
  Dixon-Coles, pero el modelo con mercado necesita que pongas las cuotas en el CSV.
- Desarrollado y ejecutado en macOS con Python 3.12; el código usa rutas `pathlib` y UTF-8
  explícito, pero no se ha ejecutado en una máquina Windows.

### Limitaciones del módulo multideporte

- **Sin cuotas históricas** (ESPN solo da las actuales): no hay backtest de rentabilidad ni comparación con
  el mercado hasta que el ledger acumule partidos. El peso modelo/mercado (`peso_mercado`) no está validado.
- **El modelo no ve lesiones, alineaciones, sanciones ni noticias.** Solo entran indirectamente, a través de
  las cuotas, en los partidos que las tienen. Sin cuotas (p. ej. tenis), el pronóstico es solo Elo + forma.
- **Tenis sin superficie** (ESPN no la publica) y con fases previas mezcladas con el cuadro principal;
  **béisbol sin lanzador abridor**; **fútbol** con los resultados de prórroga/penaltis contados como empate.
- **Cuotas de una sola casa** (DraftKings, vía ESPN): tu casa puede pagar distinto. Para medir tus apuestas,
  anota siempre tu `cuota_real`.
- La API de ESPN es pública pero no está documentada: puede cambiar de formato sin aviso (los tests de
  lectura y el aviso "Sin datos nuevos de..." lo harían visible).

## Siguientes mejoras recomendadas

1. Recuperar xG (Understat cuando vuelva, o un proveedor con licencia) y añadir segundas divisiones.
2. Modelar la **diferencia con el mercado** en vez del resultado: usar la probabilidad de apertura
   como feature/offset y predecir el movimiento hacia el cierre (CLV) en lugar del 1X2.
3. Elegir calibración e hiperparámetros con validación anidada dentro del entrenamiento (nunca con el test).
4. Información pre-partido real: alineaciones confirmadas, lesiones, competiciones europeas entre semana.
5. Apostar al mejor precio disponible (`Max*`) y estudiar mercados menos eficientes (ligas menores, hándicap asiático).
