"""
main.py — entry point for the Raspberry Pi agent.

Run with:  python main.py
On boot:   managed by systemd (see scripts/workspace-agent.service)

Startup sequence:
  1. Check if a saved anomaly baseline exists
     - Yes → load it and skip calibration
     - No  → run 5-min calibration, save baseline.pkl
  2. Spawn four daemon threads: camera, audio, fusion, cloud publisher
  3. Keep main thread alive

The monitoring threads (camera, audio, fusion) can be stopped / started
remotely via the MQTT command channel ("START_MONITORING" / "STOP_MONITORING").
The cloud publisher thread always stays alive so commands can be received.
"""

import threading
import logging
import os
import sys

from config import ANOMALY_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Monitoring lifecycle controls ─────────────────────────────────────────────
# When this event is *set*  → monitoring threads are alive.
# When this event is *clear*→ monitoring threads should gracefully exit.
monitoring_active = threading.Event()

# Keep references so we can join() old threads before spawning new ones.
monitoring_threads: list[threading.Thread] = []
_monitor_lock = threading.Lock()


def _ensure_calibrated():
    """
    Try to load an existing baseline model.
    If none exists, log a warning and continue — the system will run
    with vision + VAD only (anomaly score returns 0). The user can
    calibrate later from the Streamlit dashboard.
    """
    from anomaly.anomly_model import load_model

    if os.path.exists(ANOMALY_MODEL_PATH):
        log.info(f"Baseline model found at {ANOMALY_MODEL_PATH} — loading.")
        try:
            load_model()
        except Exception as e:
            log.error(f"Failed to load baseline model: {e}")
            log.warning("Continuing without anomaly detection.")
    else:
        log.warning(
            "⚠️  No calibrated anomaly model found! "
            "System will run with vision + VAD only (anomaly disabled). "
            "Run calibration from the Streamlit dashboard to enable full detection."
        )


def start_monitoring():
    """Spawn camera, audio, and fusion threads. Safe to call repeatedly."""
    global monitoring_threads

    with _monitor_lock:
        if monitoring_active.is_set():
            log.warning("start_monitoring() called but monitoring is already active.")
            return

        from vision.camera_loop    import camera_loop
        from audio.audio_loop      import audio_loop
        from fusion.fusion         import fusion_loop

        monitoring_active.set()

        monitoring_threads = [
            threading.Thread(target=camera_loop,  name="CameraLoop",  daemon=True),
            threading.Thread(target=audio_loop,   name="AudioLoop",   daemon=True),
            threading.Thread(target=fusion_loop,  name="FusionLoop",  daemon=True),
        ]

        for t in monitoring_threads:
            t.start()
            log.info(f"Started thread: {t.name}")

        log.info("Monitoring started — all sensor threads running.")


def stop_monitoring():
    """Signal monitoring threads to stop and wait for them to exit."""
    global monitoring_threads

    with _monitor_lock:
        if not monitoring_active.is_set():
            log.warning("stop_monitoring() called but monitoring is already stopped.")
            return

        log.info("Stopping monitoring — signalling threads to exit...")
        monitoring_active.clear()

        # Threads check monitoring_active each iteration, so they exit quickly
        for t in monitoring_threads:
            t.join(timeout=3)
            if t.is_alive():
                log.warning(f"Thread {t.name} did not exit within timeout (daemon — will be cleaned up).")
            else:
                log.info(f"Thread {t.name} exited cleanly.")

        monitoring_threads = []
        log.info("Monitoring stopped — all sensor threads shut down.")


def is_monitoring_active() -> bool:
    """Thread-safe check of current monitoring state."""
    return monitoring_active.is_set()


def main():
    log.info("Workspace Agent starting up...")

    # ── Step 1: Calibration ───────────────────────────────────────────────────
    _ensure_calibrated()

    # ── Step 2: Start cloud publisher (always-on) ─────────────────────────────
    from cloud.cloud_publisher import cloud_publisher

    cloud_thread = threading.Thread(target=cloud_publisher, name="CloudPublisher", daemon=True)
    cloud_thread.start()
    log.info(f"Started thread: {cloud_thread.name}")

    # ── Step 3: Start monitoring by default ───────────────────────────────────
    start_monitoring()

    log.info("All threads running. Agent is live.")

    # ── Step 4: Keep main thread alive ────────────────────────────────────────
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        log.info("Shutdown requested. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
