from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow

import os
import base64
from datetime import datetime
from email.message import EmailMessage

current_dir = "appFiles"
files_dir = os.path.join(current_dir,"important_files")
TOKEN_PATH = os.path.join(files_dir, 'token.json')
SCOPES = ['https://www.googleapis.com/auth/calendar',"https://www.googleapis.com/auth/gmail.send"]

def get_calender_credentials():
    cred = None
    if os.path.exists(TOKEN_PATH):
        cred = Credentials.from_authorized_user_file(TOKEN_PATH)
    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            cred_file = os.path.join(files_dir,"credentials.json")
            flow = InstalledAppFlow.from_client_secrets_file(cred_file, SCOPES)
            cred = flow.run_local_server(port=0)
        with open(TOKEN_PATH,"w")as Token:
            Token.write(cred.to_json())
    return cred 

def fetch_calendar_events():
    try:
        creds = get_calender_credentials()
        service = build('calendar', 'v3', credentials=creds)
        now = datetime.utcnow().isoformat() + "Z"
        events_result = service.events().list(
            calendarId='primary',
            timeMin=now,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        return events_result.get('items', [])
    except HttpError as error:
        raise f"An HTTP error occurred: {error}"
    except Exception as e:
        raise e

def format_booked_slots():
    events= fetch_calendar_events()
    booked_slots = []
    for event in events:
        try:
            if "dateTime" in event["start"]:
                start = datetime.fromisoformat(event["start"]["dateTime"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(event["end"]["dateTime"].replace("Z", "+00:00"))
                slot = f"{start.strftime('%A, %b %d, %Y, %I:%M %p')} to {end.strftime('%I:%M %p')}"
            elif "date" in event["start"]:
                start = datetime.fromisoformat(event["start"]["date"])
                end = datetime.fromisoformat(event["end"]["date"])
                slot = f"{start.strftime('%A, %b %d, %Y')} (All Day)"
            else:
                continue
            booked_slots.append(slot)
        except Exception as e:
            print(f"Skipping event due to error: {event}, error: {e}")
            continue
    return "\n".join(booked_slots)

def send_appointment_confirmation_email(creds: Credentials,user_email: str, user_fullname: str, appointment_time: str, duration_minutes: int):
    """
    Sends a confirmation email to both the user and the host after the appointment is successfully booked.
    Uses Gmail API with OAuth credentials for email sending.

    Args:
        user_email (str): The email of the user.
        user_fullname (str): The name of the user.
        appointment_time (str): The appointment start time in a readable format (e.g., 'Monday, Feb 20, 2025, 10:30 AM').
        duration_minutes (int): Duration of the appointment in minutes.

    Returns:
        None
    """

    try:
        # Build the Gmail API service
        service = build('gmail', 'v1', credentials=creds)

        # Create the email message
        message = EmailMessage()
        message.set_content(f"""
        Hello {user_fullname},

        Your appointment has been successfully scheduled.

        Appointment Details:
        - Date & Time: {appointment_time}
        - Duration: {duration_minutes} minutes

        Looking forward to meeting you!

        Best regards,
            Team Wattlesol
        """)
        

        # Set email headers
        message["To"] = f"awaisjutt2512@gmail.com, {user_email}"  # Send to both user and host
        message["From"] = "awaisjutt2512@gmail.com"  # Use your email as sender
        message["Subject"] = "Appointment Confirmation"

        # Encode the message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        # Create the message object for Gmail API
        create_message = {"raw": encoded_message}

        # Send the email using Gmail API
        send_message = service.users().messages().send(userId="me", body=create_message).execute()

        print(f"✅ Confirmation email sent successfully! Message Id: {send_message['id']}")

    except HttpError as error:
        print(f"⚠️ Error occurred while sending email: {error}")
        send_message = None

    return send_message
