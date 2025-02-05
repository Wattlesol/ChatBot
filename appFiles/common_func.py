from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from datetime import datetime
import os

current_dir = "appFiles"
files_dir = os.path.join(current_dir,"important_files")
TOKEN_PATH = os.path.join(files_dir, 'token.json')
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

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