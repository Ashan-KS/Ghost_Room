import paho.mqtt.client as mqtt
import logging
import json
import threading
import sys
import os
from datetime import datetime, timezone
import boto3

import re

import database

# Ensure config can be loaded if run from the project root or app boundary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    import config
    MQTT_BROKER_HOST = config.MQTT_BROKER_HOST
    MQTT_BROKER_PORT = config.MQTT_BROKER_PORT
    MQTT_TOPIC_STATUS = config.MQTT_TOPIC_STATUS
    MQTT_TOPIC_CMD = config.MQTT_TOPIC_CMD
    MQTT_TOPIC_CALIB_PROGRESS = config.MQTT_TOPIC_CALIB_PROGRESS
    MQTT_TOPIC_MONITOR_STATUS = config.MQTT_TOPIC_MONITOR_STATUS
except ImportError:
    # Fallbacks if config.py is not available directly
    MQTT_BROKER_HOST = "localhost"
    MQTT_BROKER_PORT = 1883
    MQTT_TOPIC_STATUS = "room/A/status"
    MQTT_TOPIC_CMD = "room/A/command"
    MQTT_TOPIC_CALIB_PROGRESS = "room/A/calibration/progress"
    MQTT_TOPIC_MONITOR_STATUS = "room/A/monitoring/status"
    MQTT_TOPIC_HEARTBEAT      = "room/A/heartbeat"

import time
_client = None
_lock = threading.Lock()
_last_heartbeat_time = time.time()

# Global state to share across Streamlit sessions
room_state = {
    "status": "UNKNOWN",
    "last_updated": "Never",
    "last_logged_status": None # Track transitions to avoid redundant S3 writes
}

# Calibration progress state (updated by MQTT messages from the Pi)
calibration_state = {
    "progress": 0,
    "message": "Idle",
    "status": "idle",   # idle | running | done | error
    "last_updated": "Never",
}

# Monitoring run-state (updated by MQTT retained messages from the Pi)
monitoring_state = {
    "running": None,        # None = unknown, True/False after first MQTT msg
    "message": "",
    "last_updated": "Never",
    "is_calibrated": False, # Updated live via heartbeat 
}

# Internal tracking for calibration history logging
_calib_start_time = None
_calib_samples = 0
_calib_duration = 0

def on_connect(client, userdata, flags, rc):
    """Callback fired when connected to MQTT Broker."""
    if rc == 0:
        logging.info(f"Connected to MQTT broker at {MQTT_BROKER_HOST}")
        client.subscribe(MQTT_TOPIC_STATUS)
        client.subscribe(MQTT_TOPIC_CALIB_PROGRESS)
        client.subscribe(MQTT_TOPIC_MONITOR_STATUS)
        client.subscribe(MQTT_TOPIC_HEARTBEAT)
    else:
        logging.error(f"Failed to connect to MQTT broker, return code {rc}")

def _log_event_to_s3(room_id, status, timestamp):
    """Logs state transitions and ghost detections to S3."""
    if not getattr(config, 'ENABLE_S3_LOGGING', True):
        return
        
    try:
        now = datetime.now()
        bookings = database.get_bookings()
        ghost_meeting = None
        
        # 1. Check if this 'EMPTY' state is actually a Ghost Booking
        if status == "EMPTY":
            for b in bookings:
                start = datetime.fromisoformat(b['start_time'].replace(' ', 'T'))
                end = datetime.fromisoformat(b['end_time'].replace(' ', 'T'))
                if start <= now <= end:
                    ghost_meeting = b
                    break

        # 2. Prepare payload
        payload = {
            "event_type": "GHOST_DETECTION" if ghost_meeting else "STATE_CHANGE",
            "room": room_id,
            "status": status,
            "sensor_timestamp": timestamp,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "booked_by": ghost_meeting['booked_by'] if ghost_meeting else "N/A",
            "title": ghost_meeting['title'] if ghost_meeting else "N/A",
            "duration_min": 30 # Default block size for utilization charts
        }

        # 3. Upload to S3
        s3 = boto3.client('s3')
        bucket = getattr(config, 'S3_BUCKET_NAME', 'workspace-agent-logs')
        prefix = getattr(config, 'S3_LOG_PREFIX', 'events/')
        
        utc_now = datetime.now(timezone.utc)
        suffix = "_ghost" if ghost_meeting else "_state"
        key = f"{prefix}{room_id}/{utc_now.strftime('%Y/%m/%d/%H%M%S')}{suffix}.json"
        
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload),
            ContentType='application/json'
        )
        logging.info(f"S3 Log successful -> {payload['event_type']} ({status})")

    except Exception as e:
        logging.error(f"S3 Logging failed: {e}")


def on_message(client, userdata, msg):
    """Callback fired when a message arrives on a subscribed topic."""
    global room_state, calibration_state
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)

        # ── Monitoring status messages ──
        if msg.topic == MQTT_TOPIC_MONITOR_STATUS:
            with _lock:
                monitoring_state["running"] = data.get("running")
                monitoring_state["message"] = data.get("message", "")
                monitoring_state["last_updated"] = data.get("timestamp", "Never")
            logging.info(f"Monitoring state updated -> running={monitoring_state['running']}")
            return

        # ── Heartbeat ──
        if msg.topic == MQTT_TOPIC_HEARTBEAT:
            import time
            global _last_heartbeat_time
            with _lock:
                _last_heartbeat_time = time.time()
                monitoring_state["is_calibrated"] = data.get("is_calibrated", False)
            return
            
        # ── Calibration progress messages ──
        if msg.topic == MQTT_TOPIC_CALIB_PROGRESS:
            with _lock:
                global _calib_start_time, _calib_samples, _calib_duration
                progress = data.get("progress", 0)
                message = data.get("message", "")
                ts = data.get("timestamp", "Never")

                calibration_state["progress"] = progress
                calibration_state["message"] = message
                calibration_state["last_updated"] = ts

                # Extract sample count from progress messages like "20% — 200 samples collected"
                sample_match = re.search(r'(\d+)\s*samples', message)
                if sample_match:
                    _calib_samples = int(sample_match.group(1))

                # Extract duration from the starting message
                dur_match = re.search(r'for\s+(\d+)s', message)
                if dur_match:
                    _calib_duration = int(dur_match.group(1))

                if progress <= 0 and "start" in message.lower():
                    # Calibration just started
                    _calib_start_time = datetime.now(timezone.utc)
                    calibration_state["status"] = "running" if progress == 0 else "error"
                elif progress < 0:
                    calibration_state["status"] = "error"
                    # Log failed run
                    if _calib_start_time:
                        try:
                            database.add_calibration_run(
                                started_at=_calib_start_time.isoformat(),
                                completed_at=datetime.now(timezone.utc).isoformat(),
                                duration_s=_calib_duration,
                                samples_collected=_calib_samples,
                                status="error"
                            )
                        except Exception as db_err:
                            logging.error(f"Failed to log calibration error to DB: {db_err}")
                elif progress >= 100:
                    calibration_state["status"] = "done"
                    # Log successful run to database
                    if _calib_start_time:
                        try:
                            database.add_calibration_run(
                                started_at=_calib_start_time.isoformat(),
                                completed_at=datetime.now(timezone.utc).isoformat(),
                                duration_s=_calib_duration,
                                samples_collected=_calib_samples,
                                status="success"
                            )
                            logging.info(f"Calibration run logged to DB: {_calib_samples} samples, {_calib_duration}s")
                        except Exception as db_err:
                            logging.error(f"Failed to log calibration to DB: {db_err}")
                else:
                    calibration_state["status"] = "running"

            logging.info(f"Calibration progress -> {progress}%: {message}")
            return

        # ── Room status messages ──
        status = data.get("status", "UNKNOWN")
        ts = data.get("timestamp", "Never")
        room_id = data.get("room", "A")

        with _lock:
            # Only LOG to S3 if the status has actually changed
            if status != room_state["last_logged_status"]:
                room_state["last_logged_status"] = status
                
                # Trigger log in background
                threading.Thread(
                    target=_log_event_to_s3, 
                    args=(room_id, status, ts), 
                    daemon=True
                ).start()

            # Update live state for UI
            room_state["status"] = status
            room_state["last_updated"] = ts
            
        logging.info(f"Received MQTT update -> {data}")

    except Exception as e:
        logging.error(f"Error parsing MQTT message: {e}")

def get_current_state():
    """Returns a thread-safe copy of the current room state."""
    import time
    with _lock:
        state = dict(room_state)
        # If backend is dead, override retained state
        if time.time() - _last_heartbeat_time > 15:
            state["status"] = "UNKNOWN"
        return state

def get_calibration_state():
    """Returns a thread-safe copy of the current calibration progress."""
    with _lock:
        return dict(calibration_state)

def get_monitoring_state():
    """Returns a thread-safe copy of the current monitoring run-state."""
    import time
    with _lock:
        state = dict(monitoring_state)
        # If backend hasn't pinged in 15 seconds, assume it crashed/offline
        if time.time() - _last_heartbeat_time > 15:
            state["running"] = False
            state["message"] = "Backend Offline (No connection to Edge Agent)"
        return state

def reset_calibration_state():
    """Reset calibration state back to idle (call before starting a new run)."""
    global _calib_start_time, _calib_samples, _calib_duration
    with _lock:
        calibration_state["progress"] = 0
        calibration_state["message"] = "Idle"
        calibration_state["status"] = "idle"
        calibration_state["last_updated"] = "Never"
        _calib_start_time = None
        _calib_samples = 0
        _calib_duration = 0

def publish_command(command_payload: dict):
    """
    Publish a command to the Pi agent via MQTT.
    e.g. publish_command({"command": "CALIBRATE", "duration": 60})
    """
    global _client
    if _client is None:
        logging.error("MQTT client not initialised — cannot publish command.")
        return False
    try:
        _client.publish(MQTT_TOPIC_CMD, json.dumps(command_payload))
        logging.info(f"Published command: {command_payload}")
        return True
    except Exception as e:
        logging.error(f"Failed to publish command: {e}")
        return False

def start_mqtt():
    """Initialises and starts the MQTT client loop on a background thread."""
    global _client
    if _client is not None:
        return _client
    
    _client = mqtt.Client()
    _client.on_connect = on_connect
    _client.on_message = on_message
    
    try:
        # connect_async is non-blocking, so the app won't crash if the broker isn't up
        _client.connect_async(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
        _client.loop_start()  # Runs the network loop in a background thread
    except Exception as e:
        logging.error(f"MQTT init failed: {e}")
        
    return _client
