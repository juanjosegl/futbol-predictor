# Guía de uso diario

Todo se hace desde la carpeta del proyecto. Abre la Terminal y entra una vez:

```bash
cd ~/Documents/University/futbol-predictor
```

## 1. Qué hace el sistema solo

Todos los días a las **07:00** y a las **17:00** tu Mac ejecuta `futbol hoy` sin que hagas nada:

1. Descarga los resultados nuevos, el calendario y las cuotas de las 21 ligas (fútbol, NBA, NHL, MLB, NFL, tenis).
2. Marca como acertadas o falladas las predicciones anteriores y liquida las apuestas que hayas anotado.
3. Predice los partidos de hoy y de mañana.
4. Arma la **selección del día** y actualiza los dos documentos de abajo.

Si el Mac está dormido a esa hora, lo hace al despertar. Si está apagado, ese día no corre: lo lanzas tú con
`.venv/bin/futbol hoy` (tarda unos 20 segundos).

## 2. Dónde ver todo

```bash
.venv/bin/futbol ver
```

Abre los dos documentos (también puedes abrirlos a mano desde la carpeta `reports/`):

| Documento | Qué tiene |
|---|---|
| `reports/hoy.xlsx` → hoja **Selección del día** | Los pocos partidos pensados para apostar hoy y mañana. **Es la única hoja que necesitas para decidir.** |
| `reports/hoy.xlsx` → hoja **Todos los partidos** | Todos los pronósticos, de más a menos probable (solo para consultar). |
| `reports/seguimiento.xlsx` → **Resumen** | Aciertos, rendimiento de la selección y de tus apuestas. |
| `reports/seguimiento.xlsx` → **Mis apuestas** | Cada apuesta que anotaste: Pendiente / Ganada / Perdida y cuánto dio. |
| `reports/seguimiento.xlsx` → **Selección** | ¿La selección acierta lo que promete? Selección frente al resto. |
| `reports/seguimiento.xlsx` → **Modelo vs mercado** | ¿Funciona el modelo local? Modelo solo, cuotas y combinación sobre los mismos partidos. |
| `reports/seguimiento.xlsx` → **Predicciones** | Todas las predicciones con su resultado (verde acertó, rojo falló, amarillo pendiente). |

Cierra los Excel cuando termines de mirarlos: si están abiertos a las 07:00 o a las 17:00, no se pueden actualizar.

## 3. Cómo se elige la selección del día

Siempre es **apuesta simple al ganador de un partido**. Nunca combinadas, nunca empates, nunca marcadores.
Un partido entra solo si cumple todo esto:

- La probabilidad del pronóstico es **70% o más**.
- Ese nivel de confianza **acertó 70% o más en esa liga** en años que el modelo no había visto, con al menos 100 casos.
- Si hay cuotas, **el mercado tiene al mismo favorito** que el modelo.
- Como mucho **5 partidos por día**, los de mayor probabilidad.

Si la hoja sale vacía, ese día no hay nada que cumpla: no se apuesta. Habrá días así.

Columnas que importan:

| Columna | Significado |
|---|---|
| `apostar_a` | A quién apostar (ganador del partido). |
| `probabilidad` | Probabilidad de que gane, según el sistema. |
| `cuota_minima` | 1 ÷ probabilidad. **Si tu casa paga menos que esto, esa apuesta pierde a la larga aunque se acierte seguido.** |
| `cuota_espn` | Cuota de referencia (DraftKings). La de tu casa será distinta: mira la tuya. |
| `acierto_historico` | Cuánto acertó ese nivel de confianza en esa liga en la validación. |
| `nota` | Avisa si no hay cuotas (el pronóstico es solo del modelo, sin el respaldo del mercado). |

## 4. Rutina diaria (5 minutos)

1. Abre `reports/hoy.xlsx`, hoja **Selección del día**.
2. Busca cada partido en tu casa de apuestas y mira qué cuota paga por `apostar_a`.
3. Anota la apuesta con un comando (no edites los archivos a mano). Basta una parte del nombre del equipo o jugador:

   **De papel** (sin dinero; sirve para medir antes de arriesgar):

   ```bash
   .venv/bin/futbol apuesta "Pistons" --cuota 1.22 --papel
   ```

   **Real** (con el monto que apostaste):

   ```bash
   .venv/bin/futbol apuesta "Pistons" --cuota 1.22 --monto 10000
   ```

   El comando te dice si esa cuota supera la `cuota_minima` y te avisa si el partido no está en la selección.
4. No hay paso 4: el resultado se anota solo en la siguiente ejecución automática.

Anota la apuesta **antes de que empiece el partido** y con **la cuota real de tu casa**; es lo único que hace
que la medición valga.

## 5. Cómo medir si funciona

Mira `reports/seguimiento.xlsx`, hoja **Resumen**. Tres líneas:

| Línea | Qué responde |
|---|---|
| `SELECCIÓN DEL DÍA` | ¿Acierta la selección lo que promete (alrededor de 75–80%)? |
| `MIS APUESTAS REALES` / `Mis apuestas de papel` | ¿Ganas o pierdes con las cuotas de tu casa? Mira el **yield** (beneficio ÷ apostado). |
| Hoja **Modelo vs mercado** | ¿El modelo local es tan bueno como las cuotas? Menor `log_loss` = mejor. |

Cómo leer los números:

- **Acertar no es ganar.** A cuota 1.25 necesitas acertar el 80% solo para quedar en cero; a 1.30, el 77%; a
  1.40, el 71%. La cuenta es 1 ÷ cuota.
- **El `±` es el margen de error.** Si dice `yield 3.0 ± 9.0%`, el resultado real puede estar entre −6% y +12%:
  todavía no dice nada.
- **Cuántas apuestas hacen falta:** con menos de 50, nada. Entre 100 y 200 ya se ve si la selección acierta lo
  que promete. Para saber si es rentable hacen falta varios cientos.
- **Señal de alarma:** si tras 100 o más apuestas la selección acierta claramente menos que su `probabilidad`
  media (hoja **Selección**, columnas `pct_acierto` y `prob_media_modelo`), el modelo está fallando.

Qué esperar, sin adornos: la selección debería acertar cerca del 75–80%, pero esos favoritos pagan alrededor de
1.20–1.35. Con el margen de la casa, lo normal es quedar cerca de cero o algo por debajo. El sistema te dice
con qué probabilidad gana cada uno; no garantiza ganancia.

## 6. Reglas para que la medición sea limpia

- Solo partidos de la **selección del día**, solo **apuesta simple al ganador**.
- El **mismo monto** en todas las apuestas (el que tú decidas), para que el yield sea comparable. No subirlo
  para recuperar una pérdida.
- Empieza con apuestas **de papel** unas semanas: miden exactamente lo mismo sin arriesgar dinero.
- Si apuestas dinero, que sea dinero que puedas perder entero y en casas autorizadas por Coljuegos.

Para ser más exigente, edita `seleccion` en `config.yaml` (por ejemplo `prob_min: 0.75` o `max_por_dia: 3`).

## 7. La ejecución automática

```bash
.venv/bin/futbol programar --estado
```

Dice si está activa, a qué horas y cómo terminó la última vez. Otros usos:

| Comando | Efecto |
|---|---|
| `.venv/bin/futbol programar --hora 06:30 --hora 16:00` | Cambiar las horas. |
| `.venv/bin/futbol programar --probar` | Lanzarla ahora en segundo plano para comprobar que funciona. |
| `.venv/bin/futbol programar --quitar` | Desactivarla. |

La salida de cada ejecución queda en `logs/programado.log`.

## 8. Si algo falla

| Síntoma | Qué hacer |
|---|---|
| La selección sale vacía | Normal algunos días (pocos partidos o ninguno claro). No se apuesta. |
| "No se pudo escribir ... ¿está abierto en Excel?" | Cierra el Excel y ejecuta `.venv/bin/futbol hoy`. |
| "Sin datos nuevos de: ..." | ESPN no respondió para esas ligas; se usó lo guardado. Reintenta más tarde. |
| "Hay 2 partidos pendientes con ..." al anotar | Escribe más del nombre o añade `--fecha AAAA-MM-DD`. |
| Una apuesta sigue Pendiente varios días | El partido se aplazó o canceló. |
| Quieres añadir otra liga | Una línea en `deportes.ligas` de `config.yaml` (ver README). |
