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

Abre en tu navegador (Chrome o Safari) los dos documentos. **No hace falta Excel ni ningún otro programa.**
Arriba de cada página hay botones para saltar a cada sección:

| Documento | Qué tiene |
|---|---|
| `reports/hoy.html` → **Selección del día** | Los pocos partidos pensados para apostar hoy y mañana. **Es la única hoja que necesitas para decidir.** |
| `reports/hoy.html` → **Todos los partidos** | Todos los pronósticos, de más a menos probable (solo para consultar). |
| `reports/hoy.html` → **Tipos de apuesta (fútbol)** | Para cada partido de fútbol, todas las apuestas posibles con su probabilidad y cuál se recomienda. |
| `reports/seguimiento.html` → **Resumen** | Aciertos, rendimiento de la selección y de tus apuestas. |
| `reports/seguimiento.html` → **Mis apuestas** | Cada apuesta que anotaste: Pendiente / Ganada / Perdida y cuánto dio. |
| `reports/seguimiento.html` → **Selección** | ¿La selección acierta lo que promete? Selección frente al resto. |
| `reports/seguimiento.html` → **Por tipo de apuesta** | Aciertos de la apuesta recomendada según su tipo: ganador, total de goles, doble oportunidad, ambos marcan. |
| `reports/seguimiento.html` → **Tipos de apuesta (fútbol)** | Cómo le va a cada tipo de apuesta en todos los partidos de fútbol, se recomendara o no. |
| `reports/seguimiento.html` → **Modelo vs mercado** | ¿Funciona el modelo local? Modelo solo, cuotas y combinación sobre los mismos partidos. |
| `reports/seguimiento.html` → **Predicciones** | Todas las predicciones con su resultado (verde acertó, rojo falló, amarillo pendiente). |

Las páginas no se actualizan solas mientras las tienes abiertas: después de cada ejecución, recarga la pestaña
(Cmd + R) o vuelve a ejecutar `futbol ver`.

La misma información se guarda también como `hoy.xlsx` y `seguimiento.xlsx`, por si algún día instalas Numbers
(gratis en el App Store) o Excel, o para entregarla en la universidad. Se abren con `.venv/bin/futbol ver --excel`.

## 3. La apuesta recomendada y la selección del día

Siempre es **una apuesta simple por partido**. Nunca combinadas, nunca empates, nunca marcadores exactos.

### Qué tipo de apuesta se recomienda en cada partido

En fútbol el sistema ya no mira solo quién gana: calcula la probabilidad de varios tipos de apuesta y se queda
con uno por partido (columna `apostar_a`; el tipo está en `tipo_apuesta`):

| Tipo | Apuestas | Se gana si... |
|---|---|---|
| Ganador | Gana el local / gana el visitante | ese equipo gana |
| Doble oportunidad | «X gana o empata» | ese equipo no pierde |
| Total de goles | Más de 1.5 · Más de 2.5 · Menos de 2.5 · Menos de 3.5 | hay 2 o más · 3 o más · 2 o menos · 3 o menos goles |
| Ambos marcan | Sí / No | los dos equipos marcan / al menos uno se queda en cero |

Cómo elige: parte del **ganador más probable** y lo cambia por otro tipo de apuesta solo si ese otro es **más
probable**, **todavía paga algo** (probabilidad de 80% o menos; por encima la cuota justa baja de 1.25) y **ha
acertado lo que prometía** en la validación con años no vistos. En los demás deportes (NBA, NHL, tenis...) la
apuesta recomendada sigue siendo el ganador.

Los goles salen de un modelo aparte: cada equipo tiene un nivel de ataque y otro de defensa que se actualizan
tras cada partido; con eso se calculan los goles esperados de cada uno (columna `goles_esperados`) y, de ahí,
la probabilidad de cada marcador.

### Qué entra en la selección del día

La apuesta recomendada de un partido entra solo si cumple todo esto:

- Su probabilidad es **70% o más**.
- Ese tipo de apuesta, con esa probabilidad, **acertó 70% o más** en años que el sistema no había visto, con al
  menos 100 casos (en esa liga; si ahí hay pocos casos, en todas las ligas de fútbol juntas).
- Si es una apuesta al ganador y hay cuotas, **el mercado tiene al mismo favorito**.
- Como mucho **5 partidos por día**, los de mayor probabilidad.

Si la sección sale vacía, ese día no hay nada que cumpla: no se apuesta. Habrá días así.

Columnas que importan:

| Columna | Significado |
|---|---|
| `tipo_apuesta` / `apostar_a` | Qué apostar: el tipo de apuesta y la apuesta exacta. |
| `probabilidad` | Probabilidad de que esa apuesta se cumpla, según el sistema. |
| `cuota_minima` | 1 ÷ probabilidad. **Si tu casa paga menos que esto, esa apuesta pierde a la larga aunque se acierte seguido.** |
| `cuota_espn` | Cuota de referencia (DraftKings) cuando ESPN la publica (ganador y línea de goles). La de tu casa será distinta: mira la tuya. |
| `acierto_historico` | Cuánto acertó ese tipo de apuesta con esa probabilidad en la validación. |
| `ganador_probable` / `prob_ganador` | Quién es más probable que gane y con qué probabilidad (aunque la apuesta recomendada sea otra). |
| `goles_esperados` | Goles que el sistema espera de cada equipo (local - visitante). Solo fútbol. |

Lo que hay que saber de estos tipos de apuesta: las más probables (más de 1.5 goles, doble oportunidad) son
también las que **menos pagan**, alrededor de 1.20–1.35. Que sean más fáciles de acertar no las hace rentables;
la regla sigue siendo la misma: solo si tu casa paga al menos la `cuota_minima`.

Para cambiar qué tipos de apuesta se comparan, edita `mercados` en `config.yaml` (por ejemplo, quita `"1X"` y
`"X2"` de `activos` si no quieres doble oportunidad, o baja `prob_max` a `0.75` para exigir cuotas más altas).

## 4. Rutina diaria (5 minutos)

1. Ejecuta `.venv/bin/futbol ver` y mira la sección **Selección del día**.
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

   El comando anota la **apuesta recomendada** de ese partido, te dice si la cuota supera la `cuota_minima` y te
   avisa si el partido no está en la selección. Si apostaste a otra cosa, dilo con `--a`:

   ```bash
   .venv/bin/futbol apuesta "Millonarios" --cuota 1.30 --papel --a O1.5
   ```

   Códigos de `--a`: el nombre de un equipo (ganador), `1X` / `X2` (local / visitante gana o empata), `O1.5`
   `O2.5` `O3.5` (más de N goles), `U2.5` `U3.5` (menos de N goles), `BTTS` (ambos marcan) y `NOBTTS` (no marcan ambos).
4. No hay paso 4: el resultado se anota solo en la siguiente ejecución automática.

Anota la apuesta **antes de que empiece el partido** y con **la cuota real de tu casa**; es lo único que hace
que la medición valga.

## 5. Cómo medir si funciona

Mira la página de seguimiento, sección **Resumen**. Tres líneas:

| Línea | Qué responde |
|---|---|
| `SELECCIÓN DEL DÍA` | ¿Acierta la selección lo que promete (alrededor de 75–80%)? |
| `APUESTA RECOMENDADA` y sección **Por tipo de apuesta** | ¿Acierta la apuesta recomendada? ¿Qué tipo de apuesta acierta más? |
| `MIS APUESTAS REALES` / `Mis apuestas de papel` | ¿Ganas o pierdes con las cuotas de tu casa? Mira el **yield** (beneficio ÷ apostado). |
| Sección **Modelo vs mercado** | ¿El modelo local es tan bueno como las cuotas? Menor `log_loss` = mejor. |

Cómo leer los números:

- **Acertar no es ganar.** A cuota 1.25 necesitas acertar el 80% solo para quedar en cero; a 1.30, el 77%; a
  1.40, el 71%. La cuenta es 1 ÷ cuota.
- **El `±` es el margen de error.** Si dice `yield 3.0 ± 9.0%`, el resultado real puede estar entre −6% y +12%:
  todavía no dice nada.
- **Cuántas apuestas hacen falta:** con menos de 50, nada. Entre 100 y 200 ya se ve si la selección acierta lo
  que promete. Para saber si es rentable hacen falta varios cientos.
- **Señal de alarma:** si tras 100 o más apuestas la selección acierta claramente menos que su `probabilidad`
  media (sección **Selección**, columnas `pct_acierto` y `prob_media_modelo`), el modelo está fallando.

Qué esperar, sin adornos: la selección debería acertar cerca del 75–80%, pero esos favoritos pagan alrededor de
1.20–1.35. Con el margen de la casa, lo normal es quedar cerca de cero o algo por debajo. El sistema te dice
con qué probabilidad gana cada uno; no garantiza ganancia.

## 6. Reglas para que la medición sea limpia

- Solo partidos de la **selección del día**, solo **la apuesta simple que recomienda** (una por partido).
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
| "No application knows how to open ... .xlsx" | No tienes Excel ni Numbers. Usa `.venv/bin/futbol ver` (versión web). |
| La página muestra datos viejos | Recarga la pestaña (Cmd + R). |
| "Sin datos nuevos de: ..." | ESPN no respondió para esas ligas; se usó lo guardado. Reintenta más tarde. |
| "Hay 2 partidos pendientes con ..." al anotar | Escribe más del nombre o añade `--fecha AAAA-MM-DD`. |
| Una apuesta sigue Pendiente varios días | El partido se aplazó o canceló. |
| Quieres añadir otra liga | Una línea en `deportes.ligas` de `config.yaml` (ver README). |
