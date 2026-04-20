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

def generate_and_send_ghost_booking_email(organizer, title, start_time, end_time, to_email=None):
    """
    Generates a professional email using OpenAI and sends it via Gmail SMTP.
    If to_email is not provided, it falls back to the DEFAULT_ADMIN_EMAIL from config.
    """
    to_email = to_email or config.DEFAULT_ADMIN_EMAIL
    
    # 1. Generate Email Content using OpenAI
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
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano", # Or gpt-4 depending on the user's preference
            messages=[
                {"role": "system", "content": "You are a professional corporate administrative assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=250
        )
        email_body = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating AI email: {e}")
        return False, f"Failed to generate email content via OpenAI. Error: {e}"

    # 2. Send the Email via SMTP
    subject = f"Notice: Unattended Room Booking Detection - {title}"
    
    try:
        msg = MIMEMultipart()
        msg['From'] = config.SMTP_GMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(email_body, 'plain'))
        
        # Connect to Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        
        # Login
        server.login(config.SMTP_GMAIL_USER, config.SMTP_GMAIL_APP_PASSWORD)
        
        # Send
        server.send_message(msg)
        server.quit()
        
        return True, {"subject": subject, "body": email_body, "to_email": to_email}
    except Exception as e:
        print(f"Error sending email via SMTP: {e}")
        return False, f"Failed to send email via SMTP: {e}"
