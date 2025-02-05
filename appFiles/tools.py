from langchain_core.tools import tool
from datetime import datetime, timezone ,timedelta
from googleapiclient.discovery import build

from appFiles.common_func import get_calender_credentials

@tool(parse_docstring=True)
def book_appointment(user_email: str, appointment_time: str, duration_minutes: int = 30):
    """
    Books an appointment in Google Calendar only after user confirmation.

    Process:
    1. Automatically extracts and decides the appointment date and time based on the user's message.
        - If the user doesn't specify the year, the function assumes the current year.
        - If the user doesn't specify the exact day, it defaults to the next available day.
        - If the time provided is already in the past for the current day, it automatically shifts to the next available time slot.
    2. Asks the user to confirm these details (including date, time, year, email ) before booking.
    3. If the user requests changes, updates details and re-confirms.
    4. Once confirmed, proceeds to book the appointment.

    Args:
        user_email (str): The email of the attendee.
        appointment_time (str): The desired appointment start time in ISO format ('YYYY-MM-DDTHH:MM:SS').
        duration_minutes (int, optional): Duration of the appointment in minutes. Defaults to 30.

    Returns:
        dict: 
            - If confirmation is needed: Returns a message asking the user to confirm or modify details.
            - If confirmed: Books the appointment and returns confirmation details.
            - If time slot is unavailable: Returns an error message.
    """

    try:
        print(f"📅 Booking appointment for {user_email} on {appointment_time} for {duration_minutes} minutes.")

        # Convert provided time to datetime object
        start_time = datetime.fromisoformat(appointment_time)
        end_time = start_time + timedelta(minutes=duration_minutes)

        # Authenticate with Google Calendar API
        creds = get_calender_credentials()
        service = build('calendar', 'v3', credentials=creds)

        # ✅ Step 1: Fetch existing booked slots
        events_result = service.events().list(
            calendarId='primary',
            timeMin=start_time.isoformat() + "Z",
            timeMax=end_time.isoformat() + "Z",
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])

        # ✅ Step 2: Check for conflicts
        if events:
            print("❌ Time slot is already booked. Cancelling request.")
            return {"error": "The selected time slot is already booked. Please choose another time."}

        # ✅ Step 3: Create the appointment
        event = {
            "summary": "Appointment with Wattlesol Representative",
            "location": "Virtual or Office",
            "description": f"Meeting scheduled by {user_email}.",
            "start": {"dateTime": start_time.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": end_time.isoformat(), "timeZone": "UTC"},
            "attendees": [{"email": user_email}],
            "reminders": {"useDefault": True},
        }

        created_event = service.events().insert(calendarId='primary', body=event).execute()

        print(f"✅ Appointment successfully created! Event ID: {created_event['id']}")
        print(f"📌 View event: {created_event['htmlLink']}")

        return {
            "message": "Appointment booked successfully.",
            "event_id": created_event["id"],
            "event_link": created_event["htmlLink"],
            "start_time": start_time.strftime('%A, %b %d, %Y, %I:%M %p'),
            "end_time": end_time.strftime('%I:%M %p')
        }

    except Exception as e:
        print(f"⚠️ Error occurred while booking appointment: {e}")
        return {"error": str(e)}
