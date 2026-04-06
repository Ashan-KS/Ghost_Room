"""
cloud/cloud_publisher.py — ASHAN
===================================
Publishes room state to AWS MQTT broker and subscribes to maintenance commands.
On receiving a "LOCK" command, triggers the GPIO red LED.

Install: pip install paho-mqtt
"""

import json
import logging
from datetime import datetime, timezone

import config

log = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazy-init MQTT client."""
    global _client
    if _client is not None:
        return _client

    try:
        import paho.mqtt.client as mqtt

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                log.info(f"MQTT connected to {config.MQTT_BROKER_HOST}")
                client.subscribe(config.MQTT_TOPIC_CMD)
            else:
                log.error(f"MQTT connection failed with code {rc}")

        def on_message(client, userdata, msg):
            payload = msg.payload.decode()
            log.info(f"MQTT command received: {payload}")
            _handle_command(payload)

        _client = mqtt.Client()
        _client.on_connect = on_connect
        _client.on_message = on_message
        _client.connect(config.MQTT_BROKER_HOST, config.MQTT_BROKER_PORT, keepalive=60)
        _client.loop_start()

    except Exception as e:
        log.error(f"MQTT init failed: {e}. Running without cloud.")

    return _client


def publish_state(state: str):
    """
    Publish a room state change to AWS.

    Args:
        state: "IN_USE" or "EMPTY"
    """
    payload = json.dumps({
        "room":      config.ROOM_ID,
        "status":    state,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    client = _get_client()
    if client:
        client.publish(config.MQTT_TOPIC_STATUS, payload)
        log.info(f"Published: {payload}")
    else:
        log.warning(f"No MQTT client — state not published: {state}")

    # Also flip GPIO
    from gpio.actuator import set_room_state
    set_room_state(state)


def _handle_command(payload: str):
    """Handle incoming command from the cloud dashboard."""
    try:
        data = json.loads(payload)
        cmd  = data.get("command", "")

        from gpio.actuator import maintenance_lock, release_lock

        if cmd == "LOCK":
            log.info("Maintenance lock activated.")
            maintenance_lock()
        elif cmd == "UNLOCK":
            log.info("Maintenance lock released.")
            release_lock()
        elif cmd == "CALIBRATE":
            duration = data.get("duration", config.CALIBRATION_DURATION_S)
            log.info(f"Calibration command received — duration={duration}s")
            _start_calibration(duration)
        else:
            log.warning(f"Unknown command: {cmd}")

    except Exception as e:
        log.error(f"Command handling error: {e}")


def _publish_calibration_progress(progress: int, message: str):
    """Callback passed to run_calibration(); publishes progress via MQTT."""
    payload = json.dumps({
        "room":     config.ROOM_ID,
        "progress": progress,
        "message":  message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    client = _get_client()
    if client:
        client.publish(config.MQTT_TOPIC_CALIB_PROGRESS, payload)
        log.info(f"Calibration progress published: {progress}% — {message}")


def _start_calibration(duration: int):
    """Run calibration in a background thread so the MQTT loop stays alive."""
    import threading
    from anomaly.calibration import run_calibration

    def _run():
        try:
            _publish_calibration_progress(0, "Calibration starting...")
            run_calibration(
                duration_s=duration,
                progress_callback=_publish_calibration_progress,
            )
        except Exception as e:
            log.error(f"Calibration thread error: {e}")
            _publish_calibration_progress(-1, f"ERROR: {e}")

    t = threading.Thread(target=_run, name="CalibrationThread", daemon=True)
    t.start()
    log.info("Calibration thread spawned.")


def cloud_publisher():
    """Thread target — keeps MQTT loop alive."""
    log.info("Cloud publisher started.")
    _get_client()
    import time
    while True:
        time.sleep(5)

