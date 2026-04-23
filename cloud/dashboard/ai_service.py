import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import openai
import sys
import os

# Ensure config can be loaded
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

# Use the official v1.x client initialization
from openai import OpenAI
client = OpenAI(api_key=config.OPENAI_API_KEY)


def _ensure_client():
    """Re-initialize client if necessary to ensure it picks up the loaded API key."""
    if not client.api_key:
        client.api_key = config.OPENAI_API_KEY


def _generate_organizer_email(organizer, title, start_time, end_time):
    """Generate a professional email addressed to the meeting organizer."""
    prompt = f"""
    You are an automated admin assistant for a corporate meeting room booking system.
    You need to write a professional email to a meeting organizer named '{organizer}'.
    
    The system detected a "Ghost Booking" for their meeting titled '{title}'.
    The meeting was scheduled from {start_time} to {end_time}.
    A "Ghost Booking" means that no one showed up to the room during this time, leaving it empty and wasting resources.
    
    Write a short, professional, but firm email notifying them of this detection.
    Ask them to please remember to cancel their bookings in advance if they no longer need the room.
    Do not use placeholders like [Your Name] or [Company Name], sign off as "Meeting Room Auto-Admin".
    """
    
    _ensure_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional corporate administrative assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()


def _generate_admin_email(organizer, organizer_email, title, start_time, end_time):
    """Generate an internal report email addressed to the system administrator."""
    prompt = f"""
    You are an automated admin assistant for a corporate meeting room booking system.
    You need to write an internal report email to the System Administrator.

    The system detected a "Ghost Booking":
    - Organizer: {organizer} ({organizer_email or 'no email on file'})
    - Meeting Title: '{title}'
    - Scheduled Time: {start_time} to {end_time}

    A "Ghost Booking" means the room was booked but no one showed up during the scheduled time.
    
    Write a short, professional internal notification summarizing this detection for the admin's records.
    Include the organizer's name and email so the admin knows who was notified.
    Mention that the organizer has already been notified separately.
    Do not use placeholders like [Your Name] or [Company Name], sign off as "Meeting Room Auto-Admin".
    """
    
    _ensure_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional corporate administrative assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()


def _send_email(to_email, subject, body):
    """Send a single email via Gmail SMTP."""
    msg = MIMEMultipart()
    msg['From'] = config.SMTP_GMAIL_USER
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(config.SMTP_GMAIL_USER, config.SMTP_GMAIL_APP_PASSWORD)
    server.send_message(msg)
    server.quit()


def generate_and_send_ghost_booking_email(organizer, title, start_time, end_time, to_email=None):
    """
    Generates and sends TWO separate ghost booking emails:
      1. A notification to the meeting organizer (using their booking email).
      2. An internal report to the system admin (using DEFAULT_ADMIN_EMAIL from .env).
    
    Returns a list of result dicts for each email sent, so the caller can log them individually.
    """
    admin_email = getattr(config, 'DEFAULT_ADMIN_EMAIL', None)
    results = []  # Each entry: {"to_email": ..., "subject": ..., "body": ...}
    errors = []

    # ── 1. Email to the Organizer ──
    if to_email:
        try:
            organizer_body = _generate_organizer_email(organizer, title, start_time, end_time)
            organizer_subject = f"Notice: Unattended Room Booking Detection - {title}"
            _send_email(to_email, organizer_subject, organizer_body)
            results.append({"to_email": to_email, "subject": organizer_subject, "body": organizer_body, "recipient_type": "organizer"})
        except Exception as e:
            print(f"Error sending organizer email: {e}")
            errors.append(f"Organizer email failed: {e}")

    # ── 2. Email to the System Admin ──
    if admin_email and admin_email != to_email:
        try:
            admin_body = _generate_admin_email(organizer, to_email, title, start_time, end_time)
            admin_subject = f"[Admin Report] Ghost Booking Detected - {title}"
            _send_email(admin_email, admin_subject, admin_body)
            results.append({"to_email": admin_email, "subject": admin_subject, "body": admin_body, "recipient_type": "admin"})
        except Exception as e:
            print(f"Error sending admin email: {e}")
            errors.append(f"Admin email failed: {e}")
    elif admin_email and admin_email == to_email:
        # Admin IS the organizer — send a single combined email (already sent above)
        pass

    if not results and errors:
        return False, "; ".join(errors)
    elif not results:
        return False, "No destination email addresses found."
    
    return True, results
