import paho.mqtt.client as mqtt
import logging
import json
import threading
import sys
import os

# Ensure config can be loaded if run from the project root or app boundary
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    import config
    MQTT_BROKER_HOST = config.MQTT_BROKER_HOST
    MQTT_BROKER_PORT = config.MQTT_BROKER_PORT
    MQTT_TOPIC_STATUS = config.MQTT_TOPIC_STATUS
except ImportError:
    # Fallbacks if config.py is not available directly
    MQTT_BROKER_HOST = "localhost"
    MQTT_BROKER_PORT = 1883
    MQTT_TOPIC_STATUS = "room/A/status"

_client = None
_lock = threading.Lock()

# Global state to share across Streamlit sessions
room_state = {
    "status": "UNKNOWN",
    "last_updated": "Never"
}

def on_connect(client, userdata, flags, rc):
    """Callback fired when connected to MQTT Broker."""
    if rc == 0:
        logging.info(f"Connected to MQTT broker at {MQTT_BROKER_HOST}")
        client.subscribe(MQTT_TOPIC_STATUS)
    else:
        logging.error(f"Failed to connect to MQTT broker, return code {rc}")

def on_message(client, userdata, msg):
    """Callback fired when a message arrives on a subscribed topic."""
    global room_state
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        with _lock:
            room_state["status"] = data.get("status", "UNKNOWN")
            room_state["last_updated"] = data.get("timestamp", "Never")
            
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
