import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys
import json
import boto3
import altair as alt
from streamlit_autorefresh import st_autorefresh

# Ensure imports work regardless of where the script is run from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from database import init_db, add_booking, get_bookings, delete_booking
from mqtt_subscriber import start_mqtt, get_current_state, get_calibration_state, reset_calibration_state, publish_command, get_monitoring_state
import config

# ==========================================================
# Initialize Background Systems (Run Once)
# ==========================================================
# Initialize DB
init_db()
# Start background MQTT subscriber thread
start_mqtt()

# ==========================================================
# Streamlit UI Configuration
# ==========================================================
st.set_page_config(
    page_title="Ghost Room Dashboard",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Sidebar Navigation
# ==========================================================
st.sidebar.title("👻 Ghost Room")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Dashboard", "Manage Bookings", "History", "Calibration"])
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh (Live Sync)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Developer Tools")
local_test_mode = st.sidebar.toggle("Enable Local Test Mode", help="Use manual data instead of MQTT for testing.")

s3_demo_mode = False
if local_test_mode:
    s3_demo_mode = st.sidebar.checkbox("S3 Demo Mode (Mock Data)", value=False, help="Show sample charts without S3 connection.")
    mock_status = st.sidebar.radio("Mock Room Status", ["EMPTY", "IN_USE", "UNKNOWN"], index=0)
    mock_last_updated = datetime.now().strftime("%Y-%m-%d %I:%M %p")



# ==========================================================
# Page: Dashboard
# ==========================================================
if page == "Dashboard":
    st.title("🖥️ Meeting Room Status Dashboard")
    st.markdown("Real-time monitoring and analytics for **Room A**.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Room Realtime Status ──
    if local_test_mode:
        status = mock_status
        last_updated = mock_last_updated
    else:
        state = get_current_state()
        status = state.get("status", "UNKNOWN")
        last_updated = state.get("last_updated", "Never")
        
        # Format timestamp if possible
        try:
            if last_updated != "Never":
                # parse iso format
                dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
                last_updated = dt.strftime("%Y-%m-%d %I:%M %p")
        except Exception:
            pass # fallback to original string
        
    # ── Current monitoring state from MQTT ──
    mon = get_monitoring_state()
    mon_running = mon["running"]
    mon_message = mon["message"]
    mon_updated = mon["last_updated"]

    try:
        if mon_updated and mon_updated != "Never":
            dt_mon = datetime.fromisoformat(mon_updated.replace("Z", "+00:00"))
            mon_updated = dt_mon.strftime("%Y-%m-%d %I:%M %p")
    except Exception:
        pass

    # ── Row 1: Edge Agent ──
    with st.container(border=True):
        st.subheader("🛡️ Edge Agent Monitoring", help="Controls the occupancy detection sensors (camera, audio, anomaly).")
        e1, e2 = st.columns([3, 1])
        
        with e1:
            if mon_running is True:
                st.success("🟢 Status: **RUNNING**", icon="🟢")
            elif mon_running is False:
                st.error("🔴 Status: **STOPPED**", icon="🔴")
            else:
                st.warning("🟡 Status: **UNKNOWN**", icon="🟡")
                
            if mon_message:
                st.caption(f"Info: {mon_message}")
            else:
                st.caption(f"Last Updated: {mon_updated}")
                
        with e2:
            st.write("") # spacing push down to align
            if mon_running is True:
                if st.button("⏹️ Stop Sensors", type="secondary", use_container_width=True):
                    if publish_command({"command": "STOP_MONITORING"}):
                        time.sleep(1)
                        st.rerun()
            elif mon_running is False:
                if st.button("▶️ Start Sensors", type="primary", use_container_width=True):
                    if publish_command({"command": "START_MONITORING"}):
                        time.sleep(1)
                        st.rerun()
            else:
                bc1, bc2 = st.columns(2)
                with bc1:
                    if st.button("▶️ Start", use_container_width=True):
                        if publish_command({"command": "START_MONITORING"}):
                            time.sleep(1)
                            st.rerun()
                with bc2:
                    if st.button("⏹️ Stop", use_container_width=True):
                        if publish_command({"command": "STOP_MONITORING"}):
                            time.sleep(1)
                            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Occupancy & Booking Overview ──
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.container(border=True):
            st.subheader("Current Occupancy")
            if status == "IN_USE":
                st.error("🔴 IN USE", icon="🔴")
                st.caption(f"Last Updated: {last_updated}")
            elif status == "EMPTY":
                st.success("🟢 AVAILABLE", icon="🟢")
                st.caption(f"Last Updated: {last_updated}")
            else:
                st.warning(f"🟡 {status}", icon="🟡")
                st.caption(f"Last Updated: {last_updated}")
            
    with col2:
        with st.container(border=True):
            st.subheader("Booking Overview")
            bookings = get_bookings()
            today = datetime.now().date()
            today_df = pd.DataFrame()
            
            if bookings:
                df = pd.DataFrame(bookings)
                df['start_time'] = pd.to_datetime(df['start_time'])
                today_df = df[df['start_time'].dt.date == today]
                
            today_count = len(today_df)
            st.metric("Meetings Scheduled Today", f"📅 {today_count}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Ghost Booking Detection ──
    if not today_df.empty:
        now = pd.to_datetime(datetime.now())
        # Add timezone-naive datetime comparison if needed, but since sqlite is naive, pandas should be naive
        df['end_time'] = pd.to_datetime(df['end_time'])
        today_df = df[df['start_time'].dt.date == today].copy()
        
        current_bookings = today_df[(today_df['start_time'] <= now) & (today_df['end_time'] >= now)]
        
        if not current_bookings.empty and status == "EMPTY":
            ghost = current_bookings.iloc[0]
            st.warning(f"👻 **Ghost Booking Detected!** The room is currently empty but is booked by **{ghost['booked_by']}** for '{ghost['title']}'.", icon="👻")
            
            with st.expander("🚀 Claim Room Now!", expanded=False):
                with st.form("claim_form"):
                    st.write("Take over the room and add your own booking right now.")
                    claim_name = st.text_input("Your Name", placeholder="e.g. Alice")
                    claim_duration = st.number_input("Duration (minutes)", min_value=15, max_value=120, value=30, step=15)
                    claim_submit = st.form_submit_button("Take Over Room")
                    
                    if claim_submit:
                        if not claim_name.strip():
                            st.error("Please enter your name to claim the room.")
                        else:
                            end_dt = now + timedelta(minutes=claim_duration)
                            add_booking(
                                f"Takeover: {claim_name}", 
                                claim_name, 
                                now.strftime('%Y-%m-%d %H:%M:%S'), 
                                end_dt.strftime('%Y-%m-%d %H:%M:%S')
                            )
                            st.success("Room successfully claimed!")
                            time.sleep(1.5)
                            st.rerun()

    st.subheader("Today's Agenda")
    
    # ── Today's Bookings ──
    if today_df.empty:
        st.info("No meetings scheduled for today! Room is completely free.")
    else:
        df['end_time'] = pd.to_datetime(df['end_time'])
        today_df = df[df['start_time'].dt.date == today].copy()
        today_df = today_df.sort_values(by='start_time')
        
        now = pd.to_datetime(datetime.now())
        
        def get_row_status(row):
            is_current = row['start_time'] <= now <= row['end_time']
            if row['end_time'] < now:
                return "✅ Finished"
            elif row['start_time'] > now:
                return "🕒 Upcoming"
            elif is_current:
                if status == "EMPTY":
                    return "👻 Ghost Booking"
                elif status == "IN_USE":
                    # Check if there is a takeover in current bookings
                    current_subset = today_df[(today_df['start_time'] <= now) & (today_df['end_time'] >= now)]
                    has_takeover = any(current_subset['title'].str.startswith("Takeover:"))
                    
                    if has_takeover:
                        if row['title'].startswith("Takeover:"):
                            return "🔴 Ongoing (Takeover)"
                        else:
                            return "👻 Ghost Booking (Replaced)"
                    else:
                        return "🔴 Ongoing"
                else:
                    return "🟡 Scheduled"
            return "〰️ Unknown"
            
        today_df['Status'] = today_df.apply(get_row_status, axis=1)
        today_df['Time'] = today_df['start_time'].dt.strftime('%I:%M %p') + " - " + today_df['end_time'].dt.strftime('%I:%M %p')
        today_df['Meeting'] = today_df['title']
        today_df['Organizer'] = today_df['booked_by']
        
        st.dataframe(
            today_df[['Time', 'Meeting', 'Organizer', 'Status']], 
            width='stretch', 
            hide_index=True
        )

    # ── Auto Reset Logic ──
    # If auto-refresh is active, we trigger a rerun cleanly using JS (avoids duplicate table ghosting)
    if auto_refresh:
        st_autorefresh(interval=3000, key="dashboard_autorefresh")
        
# ==========================================================
# Page: History (Analytics)
# ==========================================================
elif page == "History":
    col_header, col_refresh = st.columns([5, 1])
    with col_header:
        st.title("📜 Room History & Analytics")
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True) # Align with title
        if st.button("🔄 Refresh", width='stretch'):
            st.cache_data.clear()
            st.rerun()
            
    st.markdown("Auditing sensor events and ghost booking detections stored in S3.")

    @st.cache_data(ttl=300)
    def fetch_s3_history(force_demo=False):
        """Fetches and parses JSON logs from the S3 bucket or returns mock data."""
        # Only use mock data if the user explicitly enabled S3 Demo Mode
        use_mock = force_demo
        
        if use_mock:
            # Generate 30 mock events for a "full day" visualization
            mock_data = []
            now = datetime.now().replace(hour=17, minute=0, second=0, microsecond=0)
            organizers = ["Alice", "Bob", "Charlie", "Diana"]
            
            for i in range(30):
                event_time = now - timedelta(minutes=i*30)
                # Cycle through types
                if i % 4 == 0:
                    etype, status = "GHOST_DETECTION", "EMPTY"
                elif i % 2 == 0:
                    etype, status = "STATE_CHANGE", "IN_USE"
                else:
                    etype, status = "STATE_CHANGE", "EMPTY"

                mock_data.append({
                    "event_type": etype,
                    "room": "A",
                    "status": status,
                    "logged_at": event_time.isoformat(),
                    "booked_by": organizers[i % len(organizers)],
                    "title": f"Meeting {i}",
                    "duration_min": 30 # For utilization calc
                })
            return pd.DataFrame(mock_data)

        # If S3 logging is disabled, don't attempt to connect — just return empty
        if not getattr(config, 'ENABLE_S3_LOGGING', True):
            return pd.DataFrame()

        try:
            s3 = boto3.client('s3')
            # Extract bucket/prefix from config
            bucket = config.S3_BUCKET_NAME
            prefix = config.S3_LOG_PREFIX
            
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' not in response:
                return pd.DataFrame()
            
            all_events = []
            # We only fetch the last 100 objects to keep the dashboard snappy
            sorted_contents = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)[:100]
            
            for obj in sorted_contents:
                try:
                    data_obj = s3.get_object(Bucket=bucket, Key=obj['Key'])
                    payload = json.loads(data_obj['Body'].read().decode('utf-8'))
                    all_events.append(payload)
                except Exception:
                    continue
            
            return pd.DataFrame(all_events)
        except Exception as e:
            st.error(f"Could not reach S3. Check IAM permissions. Error: {e}")
            return pd.DataFrame()

    with st.spinner("Fetching logs..."):
        history_df = fetch_s3_history(force_demo=s3_demo_mode)

    if history_df.empty:
        st.info("No logs found in S3 yet. History will appear once events are recorded.")
    else:
        # ── KPI Metrics Card ──
        with st.container(border=True):
            st.subheader("📊 Key Performance Indicators")
            total_events = len(history_df)
            ghost_events = len(history_df[history_df['event_type'] == 'GHOST_DETECTION']) if 'event_type' in history_df.columns else 0
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Events Logged", total_events)
            m2.metric("Ghost Bookings Detected", ghost_events, delta=f"{ghost_events} alerts", delta_color="inverse")
            
            if ghost_events > 0 and 'booked_by' in history_df.columns:
                ghost_df = history_df[history_df['event_type'] == 'GHOST_DETECTION']
                if not ghost_df.empty:
                    top_offender = ghost_df['booked_by'].mode()[0]
                    m3.metric("Top 'Ghost' Organizer", top_offender)
                else:
                    m3.metric("Top 'Ghost' Organizer", "N/A")
            else:
                m3.metric("Top 'Ghost' Organizer", "N/A")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── Visualization & Audit Explorer ──
        with st.container(border=True):
            tab1, tab2 = st.tabs(["📉 Analytics Dashboard", "📅 Detailed Event Audit"])
            
            with tab1:
                # 1. Room Utilization by Organizer (Stacked Bar)
                st.subheader("Room Utilization by Organizer")
                if not history_df.empty:
                    util_df = history_df.copy()
                    def categorize(row):
                        if row['event_type'] == 'GHOST_DETECTION': return 'GHOST_BOOKING'
                        if row['status'] == 'IN_USE': return 'IN_USE'
                        return 'NOT_IN_USE'
                    
                    util_df['category'] = util_df.apply(categorize, axis=1)
                    
                    util_chart = alt.Chart(util_df).mark_bar().encode(
                        x=alt.X('sum(duration_min):Q' if 'duration_min' in util_df.columns else 'count():Q', title='Total Minutes / Events'),
                        y=alt.Y('booked_by:N', title='Organizer', sort='-x'),
                        color=alt.Color('category:N', scale=alt.Scale(
                            domain=['IN_USE', 'NOT_IN_USE', 'GHOST_BOOKING'],
                            range=['#4CAF50', '#9E9E9E', '#FF4B4B']
                        ), title='Room State'),
                        tooltip=['booked_by', 'category', 'count()' if 'duration_min' not in util_df.columns else 'sum(duration_min)']
                    ).properties(height=300)
                    
                    st.altair_chart(util_chart, width='stretch')
                else:
                    st.info("No logs available yet.")

                st.divider()
                
                # 2. Distribution Donut Chart & Efficiency Stats
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.subheader("Total Time Distribution")
                    if not history_df.empty:
                        dist_df = util_df['category'].value_counts().reset_index()
                        dist_df.columns = ['status', 'count']
                        
                        pie = alt.Chart(dist_df).mark_arc(innerRadius=60).encode(
                            theta=alt.Theta(field="count", type="quantitative"),
                            color=alt.Color(field="status", type="nominal", scale=alt.Scale(
                                domain=['IN_USE', 'NOT_IN_USE', 'GHOST_BOOKING'],
                                range=['#4CAF50', '#9E9E9E', '#FF4B4B']
                            )),
                            tooltip=['status', 'count']
                        ).properties(height=300)
                        
                        st.altair_chart(pie, width='stretch')

                with col_b:
                    st.subheader("Quick Stats")
                    total_min = util_df['duration_min'].sum() if 'duration_min' in util_df.columns else len(util_df)
                    ghost_min = util_df[util_df['category'] == 'GHOST_BOOKING']['duration_min'].sum() if 'duration_min' in util_df.columns else len(util_df[util_df['category'] == 'GHOST_BOOKING'])
                    efficiency = (1 - (ghost_min / total_min)) * 100 if total_min > 0 else 0
                    
                    st.metric("Room Efficiency Score", f"{efficiency:.1f}%")
                    st.write("Efficiency is calculated based on the ratio of actual usage vs ghosted slots.")
                    st.markdown(f"**Total Tracked Time:** {total_min} mins" if 'duration_min' in util_df.columns else f"**Total Events:** {total_min}")

            with tab2:
                st.subheader("Detailed Event Audit")
                if not history_df.empty:
                    display_df = history_df.copy()
                    if 'logged_at' in display_df.columns:
                        display_df['time'] = pd.to_datetime(display_df['logged_at']).dt.strftime('%b %d, %H:%M:%S')
                    
                    cols = ['time', 'event_type', 'status', 'booked_by', 'title']
                    existing_cols = [c for c in cols if c in display_df.columns]
                    other_cols = [c for c in display_df.columns if c not in existing_cols]
                    
                    st.dataframe(
                        display_df[existing_cols + other_cols], 
                        width='stretch', 
                        hide_index=True
                    )
                else:
                    st.write("No data available to display in the audit log.")

# ==========================================================
# Page: Manage Bookings
# ==========================================================
elif page == "Manage Bookings":
    st.title("🗓️ Manage Room Bookings")
    
    # ── Add New Booking Card ──
    with st.container(border=True):
        st.subheader("➕ Add a New Booking")
        with st.form("new_booking_form"):
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Meeting Title", placeholder="e.g. Daily Standup")
                booked_by = st.text_input("Organizer Name", placeholder="e.g. Alice")
            with col2:
                date = st.date_input("Date")
                start_time = st.time_input("Start Time", value=datetime.now().time())
                duration = st.number_input("Duration (minutes)", min_value=15, max_value=480, value=60, step=15)
                
            submit = st.form_submit_button("Book Room", type="primary", use_container_width=True)
            
            if submit:
                if not title.strip() or not booked_by.strip():
                    st.error("Please fill in both the Meeting Title and Organizer fields.")
                else:
                    start_dt = datetime.combine(date, start_time)
                    end_dt = start_dt + timedelta(minutes=duration)
                    
                    add_booking(
                        title, 
                        booked_by, 
                        start_dt.strftime('%Y-%m-%d %H:%M:%S'), 
                        end_dt.strftime('%Y-%m-%d %H:%M:%S')
                    )
                    st.success(f"Successfully booked '{title}' on {start_dt.strftime('%b %d')} at {start_dt.strftime('%I:%M %p')}")
                    time.sleep(1.5)
                    st.rerun()
                    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Upcoming Schedule Feed ──
    with st.container(border=True):
        col_title, col_refresh = st.columns([5, 1])
        with col_title:
            st.subheader("📅 Upcoming Schedule")
        with col_refresh:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
                
        bookings = get_bookings()
        if bookings:
            df = pd.DataFrame(bookings)
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])
            
            today = datetime.now().date()
            future_df = df[df['start_time'].dt.date >= today].copy()
            future_df = future_df.sort_values(by='start_time')
            
            if future_df.empty:
                st.info("No upcoming bookings found.")
            else:
                for index, row in future_df.iterrows():
                    dt_str = row['start_time'].strftime('%A, %B %d, %Y')
                    time_str = f"{row['start_time'].strftime('%I:%M %p')} - {row['end_time'].strftime('%I:%M %p')}"
                    
                    # Individual Booking Card
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([5, 3, 2])
                        with c1:
                            st.markdown(f"**{row['title']}**")
                            st.caption(f"Organizer: {row['booked_by']}")
                        with c2:
                            st.markdown(f"*{dt_str}*")
                            st.markdown(f"**{time_str}**")
                        with c3:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button("Cancel", key=f"del_{row['id']}", type="secondary", use_container_width=True, help="Remove this booking"):
                                delete_booking(row['id'])
                                st.rerun()
        else:
            st.info("No bookings found in the database.")

# ==========================================================
# Page: Calibration
# ==========================================================
elif page == "Calibration":
    st.title("🎛️ Anomaly Model Calibration")
    st.markdown("Run environmental audio calibration to establish a baseline for anomaly detection on the edge device.")

    # Initialize session state for calibration tracking
    if "calib_triggered" not in st.session_state:
        st.session_state.calib_triggered = False
    if "calib_done_shown" not in st.session_state:
        st.session_state.calib_done_shown = False

    # ── Container 1: Trigger Calibration ──
    with st.container(border=True):
        st.subheader("🚀 Run New Calibration")
        st.warning("⚠️ Please ensure the room is completely empty and quiet during calibration.")
        
        c1, c2 = st.columns([2, 1])
        with c1:
             calib_duration = st.number_input("Calibration Duration (seconds)", min_value=10, max_value=300, value=60, step=10)
        with c2:
             st.markdown("<br>", unsafe_allow_html=True)
             if st.button("Start Calibration", type="primary", use_container_width=True):
                 reset_calibration_state()
                 st.session_state.calib_done_shown = False
                 success = publish_command({"command": "CALIBRATE", "duration": calib_duration})
                 if success:
                     st.session_state.calib_triggered = True
                     st.rerun()
                 else:
                     st.error("❌ Failed to send calibration command. Check MQTT broker connection.")

    # ── Container 2: Live Progress ──
    with st.container(border=True):
        st.subheader("📡 Live Progress")
        calib = get_calibration_state()
        calib_status = calib["status"]
        progress = calib["progress"]
        message = calib["message"]

        # Once the backend responds, clear the "waiting" flag
        if calib_status in ("running", "done", "error"):
            st.session_state.calib_triggered = False

        if calib_status == "running":
            st.session_state.calib_done_shown = False
            st.progress(min(max(progress, 0), 100))
            st.markdown(f"**Status:** {message}")
            st_autorefresh(interval=1000, key="calibration_autorefresh")
        elif calib_status == "done" and not st.session_state.calib_done_shown:
            st.success("✅ Calibration complete! Baseline model updated successfully.")
            st.session_state.calib_done_shown = True
            st_autorefresh(interval=3000, limit=1, key="calibration_done_dismiss")
        elif calib_status == "error":
            st.error(f"❌ Calibration error: {message}")
        elif st.session_state.calib_triggered:
            st.info("📡 Command sent. Waiting for the edge device to respond...")
            st_autorefresh(interval=1000, key="calibration_waiting_autorefresh")
        else:
            st.info("💤 No calibration in progress. Use the controls above to start one.")

    # ── Container 3: Calibration History ──
    with st.container(border=True):
        st.subheader("📋 Calibration History")

        from database import get_calibration_runs
        runs = get_calibration_runs()

        if runs:
            history_df = pd.DataFrame(runs)
            history_df['started_at'] = pd.to_datetime(history_df['started_at'])
            history_df['completed_at'] = pd.to_datetime(history_df['completed_at'])
            history_df['Date'] = history_df['completed_at'].dt.strftime('%b %d, %Y')
            history_df['Time'] = history_df['started_at'].dt.strftime('%I:%M %p') + ' \u2192 ' + history_df['completed_at'].dt.strftime('%I:%M %p')
            history_df['Duration'] = history_df['duration_s'].apply(lambda s: f"{s}s")
            history_df['Samples'] = history_df['samples_collected'].apply(lambda s: f"{s:,}")
            history_df['Status'] = history_df['status'].apply(lambda s: "\u2705 Success" if s == "success" else "\u274c Failed")

            st.dataframe(
                history_df[['Date', 'Time', 'Duration', 'Samples', 'Status']],
                width='stretch',
                hide_index=True
            )
        else:
            st.caption("No calibration runs recorded yet. Run your first calibration above!")
