from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from datetime import datetime, timedelta, timezone

from datetime import datetime

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Path to the Google OAuth2 credentials token
TOKEN_PATH = "important_files/token.json"

@tool(parse_docstring=True)
def book_appointment(user_email: str, appointment_time: str, duration_minutes: int = 30):
    """
    Books an appointment in Google Calendar.

    Args:
        user_email (str): The email of the attendee.
        appointment_time (str): The desired appointment start time in ISO format ('YYYY-MM-DDTHH:MM:SS').
        duration_minutes (int, optional): Duration of the appointment in minutes. Defaults to 30.

    Returns:
        dict: Confirmation details or an error message.
    """
    try:
        print(f"📅 Booking appointment for {user_email} on {appointment_time} for {duration_minutes} minutes.")

        # Convert provided time to datetime object
        start_time = datetime.fromisoformat(appointment_time)
        end_time = start_time + timedelta(minutes=duration_minutes)

        # Authenticate with Google Calendar API
        creds = Credentials.from_authorized_user_file(TOKEN_PATH)
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


tools_list = {
    "book_appointment": book_appointment,
}
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools(list(tools_list.values()))


prompt = input("Enter your prompt: ")
if prompt:
    messages = []

    messages.append(HumanMessage(prompt))
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    if not ai_response.tool_calls:
        response_text = ai_response.content
    else:
        for tool_call in ai_response.tool_calls:
            selected_tool = tools_list.get(tool_call["name"].lower())
            tool_response = selected_tool.invoke(tool_call["args"])
            messages.append(ToolMessage(tool_response, tool_call_id=tool_call["id"]))

        final_response = llm_with_tools.stream(messages)

        # Collect all chunks
        response_text = "".join(chunk.content for chunk in final_response) 



print(response_text)  # Prints the entire response as a single string
