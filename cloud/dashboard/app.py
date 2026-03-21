import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys
from streamlit_autorefresh import st_autorefresh

# Ensure imports work if run from inside app/ folder
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database import init_db, add_booking, get_bookings, delete_booking
from mqtt_subscriber import start_mqtt, get_current_state

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
page = st.sidebar.radio("Navigation", ["Dashboard", "Manage Bookings"])
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh (Live Sync)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Developer Tools")
local_test_mode = st.sidebar.toggle("Enable Local Test Mode", help="Use manual data instead of MQTT for testing.")

if local_test_mode:
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
        
    col1, col2 = st.columns([1, 1])
    
    with col1:
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
            use_container_width=True, 
            hide_index=True
        )

    # ── Auto Reset Logic ──
    # If auto-refresh is active, we trigger a rerun cleanly using JS (avoids duplicate table ghosting)
    if auto_refresh:
        st_autorefresh(interval=3000, key="dashboard_autorefresh")
        
# ==========================================================
# Page: Manage Bookings
# ==========================================================
elif page == "Manage Bookings":
    st.title("🗓️ Manage Room Bookings")
    
    # ── Add New Booking ──
    with st.expander("➕ **Add a New Booking**", expanded=False):
        with st.form("new_booking_form"):
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Meeting Title", placeholder="e.g. Daily Standup")
                booked_by = st.text_input("Organizer Name", placeholder="e.g. Alice")
            with col2:
                date = st.date_input("Date")
                start_time = st.time_input("Start Time", value=datetime.now().time())
                duration = st.number_input("Duration (minutes)", min_value=15, max_value=480, value=60, step=15)
                
            submit = st.form_submit_button("Book Room", use_container_width=True)
            
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
                    
                    # Pause briefly so user can read message before rerunning list
                    time.sleep(1.5)
                    st.rerun()
                    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Upcoming Schedule")
    
    # ── Upcoming Bookings List ──
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
                # Format visually
                dt_str = row['start_time'].strftime('%A, %B %d, %Y')
                time_str = f"{row['start_time'].strftime('%I:%M %p')} - {row['end_time'].strftime('%I:%M %p')}"
                
                with st.container():
                    c1, c2, c3 = st.columns([5, 2, 1])
                    with c1:
                        st.markdown(f"**{row['title']}**")
                        st.caption(f"Organizer: {row['booked_by']}")
                    with c2:
                        st.markdown(f"*{dt_str}*")
                        st.markdown(f"**{time_str}**")
                    with c3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("Cancel", key=f"del_{row['id']}", help="Remove this booking"):
                            delete_booking(row['id'])
                            st.rerun()
                st.divider()
    else:
        st.info("No bookings found in the database.")
