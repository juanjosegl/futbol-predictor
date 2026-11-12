"""Ejecución automática diaria de `futbol hoy` en macOS (launchd).

Crea ~/Library/LaunchAgents/com.futbol-predictor.hoy.plist, que lanza `futbol hoy` a las horas indicadas.
Si el Mac está dormido a esa hora, launchd lo ejecuta al despertar; si está apagado, ese día no corre.
La salida de cada ejecución queda en logs/programado.log.

    futbol programar                 # 07:00 y 17:00 (por defecto)
    futbol programar --hora 06:30    # otra(s) hora(s); repetible
    futbol programar --estado        # ¿está activo? ¿cuándo corrió por última vez?
    futbol programar --quitar        # desactivar
"""
from __future__ import annotations

import logging
import os
import plistlib
import subprocess
import sys
from pathlib import Path

from .config import get_path

log = logging.getLogger(__name__)

LABEL = "com.futbol-predictor.hoy"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def build_plist(cfg: dict, hours: list[str]) -> dict:
    root = Path(cfg["_root"])
    exe = Path(sys.executable).parent / "futbol"
    out = get_path(cfg, "logs") / "programado.log"
    times = []
    for h in hours:
        hh, mm = (int(x) for x in h.split(":"))
        if not (0 <= hh < 24 and 0 <= mm < 60):
            raise ValueError(f"Hora no válida: {h} (usa HH:MM)")
        times.append({"Hour": hh, "Minute": mm})
    return {"Label": LABEL, "ProgramArguments": [str(exe), "hoy"], "WorkingDirectory": str(root),
            "StartCalendarInterval": times, "StandardOutPath": str(out), "StandardErrorPath": str(out),
            "EnvironmentVariables": {"FUTBOL_HOME": str(root), "PYTHONIOENCODING": "utf-8",
                                     "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin"},
            "ProcessType": "Background"}


def install(cfg: dict, hours: list[str]) -> Path:
    if sys.platform != "darwin":
        raise RuntimeError("`futbol programar` solo funciona en macOS. En Windows usa el Programador de tareas "
                           "con el comando `.venv\\Scripts\\futbol.exe hoy`.")
    p = plist_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    _launchctl("bootout", f"{_domain()}/{LABEL}")  # si ya existía, se reemplaza
    p.write_bytes(plistlib.dumps(build_plist(cfg, hours)))
    r = _launchctl("bootstrap", _domain(), str(p))
    if r.returncode != 0:
        raise RuntimeError(f"launchctl no pudo activar la tarea: {r.stderr.strip() or r.stdout.strip()}")
    log.info("Programado: `futbol hoy` cada día a las %s. Salida en %s", ", ".join(hours),
             get_path(cfg, "logs") / "programado.log")
    return p


def remove() -> bool:
    p = plist_path()
    _launchctl("bootout", f"{_domain()}/{LABEL}")
    if p.exists():
        p.unlink()
        log.info("Tarea automática desactivada (%s eliminado).", p)
        return True
    log.info("No había ninguna tarea automática instalada.")
    return False


def run_now() -> None:
    """Lanza la tarea ahora mismo a través de launchd (para comprobar que funciona en segundo plano)."""
    r = _launchctl("kickstart", f"{_domain()}/{LABEL}")
    if r.returncode != 0:
        raise RuntimeError(f"No se pudo lanzar: {r.stderr.strip() or r.stdout.strip()}")


def status(cfg: dict) -> dict:
    p = plist_path()
    info = {"instalada": p.exists(), "activa": False, "horas": [], "ultima_salida": None}
    if p.exists():
        d = plistlib.loads(p.read_bytes())
        info["horas"] = [f"{t['Hour']:02d}:{t['Minute']:02d}" for t in d.get("StartCalendarInterval", [])]
        r = _launchctl("print", f"{_domain()}/{LABEL}")
        info["activa"] = r.returncode == 0
        for line in r.stdout.splitlines():
            if "last exit code" in line:
                info["ultimo_codigo_salida"] = line.split("=")[-1].strip()
    out = get_path(cfg, "logs") / "programado.log"
    if out.exists():
        lines = out.read_text(encoding="utf-8", errors="replace").splitlines()
        info["ultima_salida"] = lines[-1][:200] if lines else None
    log.info("Tarea automática: %s", info)
    return info
