import paho.mqtt.client as mqtt
import logging
import json
import threading
import sys
import os
from datetime import datetime, timezone
import boto3

import database

# Ensure app_config can be loaded if run from the project root or app boundary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    import app_config
    MQTT_BROKER_HOST = app_config.MQTT_BROKER_HOST
    MQTT_BROKER_PORT = app_config.MQTT_BROKER_PORT
    MQTT_TOPIC_STATUS = app_config.MQTT_TOPIC_STATUS
except ImportError:
    # Fallbacks if app_config.py is not available directly
    MQTT_BROKER_HOST = "localhost"
    MQTT_BROKER_PORT = 1883
    MQTT_TOPIC_STATUS = "room/A/status"

_client = None
_lock = threading.Lock()

# Global state to share across Streamlit sessions
room_state = {
    "status": "UNKNOWN",
    "last_updated": "Never",
    "last_logged_status": None # Track transitions to avoid redundant S3 writes
}

def on_connect(client, userdata, flags, rc):
    """Callback fired when connected to MQTT Broker."""
    if rc == 0:
        logging.info(f"Connected to MQTT broker at {MQTT_BROKER_HOST}")
        client.subscribe(MQTT_TOPIC_STATUS)
    else:
        logging.error(f"Failed to connect to MQTT broker, return code {rc}")

def _log_event_to_s3(room_id, status, timestamp):
    """Logs state transitions and ghost detections to S3."""
    if not getattr(app_config, 'ENABLE_S3_LOGGING', True):
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
        bucket = getattr(app_config, 'S3_BUCKET_NAME', 'workspace-agent-logs')
        prefix = getattr(app_config, 'S3_LOG_PREFIX', 'events/')
        
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
    global room_state
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        
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
    with _lock:
        return dict(room_state)

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
