"""
cloud/dashboard/app.py — ASHAN
================================
Streamlit web dashboard — runs on AWS EC2.

Run with:  streamlit run cloud/dashboard/app.py --server.port 8501

Features:
  - Live room status (IN_USE / EMPTY)
  - Occupancy timeline chart from S3 logs
  - Maintenance Lock / Unlock button → publishes MQTT command to Pi

Install: pip install streamlit paho-mqtt boto3 pandas altair
"""

import json
import time
import threading
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import config

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Workspace Agent Dashboard",
    page_icon="🏢",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "room_status"    not in st.session_state: st.session_state.room_status    = "UNKNOWN"
if "last_updated"   not in st.session_state: st.session_state.last_updated   = "—"
if "locked"         not in st.session_state: st.session_state.locked         = False
if "history"        not in st.session_state: st.session_state.history        = []


# ── MQTT listener (background thread) ────────────────────────────────────────
def start_mqtt_listener():
    try:
        import paho.mqtt.client as mqtt

        def on_message(client, userdata, msg):
            data = json.loads(msg.payload.decode())
            st.session_state.room_status  = data.get("status", "UNKNOWN")
            st.session_state.last_updated = data.get("timestamp", "")
            st.session_state.history.append({
                "time":   data.get("timestamp", ""),
                "status": data.get("status", "UNKNOWN"),
            })

        client = mqtt.Client()
        client.on_message = on_message
        client.connect(config.MQTT_BROKER_HOST, config.MQTT_BROKER_PORT)
        client.subscribe(config.MQTT_TOPIC_STATUS)
        client.loop_forever()

    except Exception as e:
        st.session_state.room_status = f"MQTT error: {e}"


# Start listener once
if "mqtt_started" not in st.session_state:
    st.session_state.mqtt_started = True
    t = threading.Thread(target=start_mqtt_listener, daemon=True)
    t.start()


# ── MQTT command sender ───────────────────────────────────────────────────────
def send_command(command: str):
    try:
        import paho.mqtt.client as mqtt
        client = mqtt.Client()
        client.connect(config.MQTT_BROKER_HOST, config.MQTT_BROKER_PORT)
        payload = json.dumps({"command": command, "timestamp": datetime.now(timezone.utc).isoformat()})
        client.publish(config.MQTT_TOPIC_CMD, payload)
        client.disconnect()
    except Exception as e:
        st.error(f"Failed to send command: {e}")


# ── Layout ────────────────────────────────────────────────────────────────────
st.title("Workspace Agent")
st.caption(f"Room {config.ROOM_ID}  ·  Refreshes every 2 seconds")

col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    status = st.session_state.room_status
    if status == "IN_USE":
        st.error("🔴  Room IN USE")
    elif status == "EMPTY":
        st.success("🟢  Room AVAILABLE")
    else:
        st.info(f"⚪  {status}")
    st.caption(f"Last updated: {st.session_state.last_updated}")

with col2:
    st.metric("Events today", len(st.session_state.history))

with col3:
    if st.session_state.locked:
        if st.button("🔓 Release maintenance lock", use_container_width=True):
            send_command("UNLOCK")
            st.session_state.locked = False
            st.rerun()
    else:
        if st.button("🔒 Maintenance lock", use_container_width=True):
            send_command("LOCK")
            st.session_state.locked = True
            st.rerun()

st.divider()

# ── History chart ─────────────────────────────────────────────────────────────
st.subheader("Occupancy timeline")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    df["time"]  = pd.to_datetime(df["time"])
    df["value"] = df["status"].map({"IN_USE": 1, "EMPTY": 0})
    st.line_chart(df.set_index("time")["value"])
else:
    st.caption("No events yet — waiting for data from the Pi.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(2)
st.rerun()
