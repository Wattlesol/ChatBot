import datetime
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

# "installed": {"client_id": "338720042367-a8uvjkh1qvr223qkfq37t0fs0mk5odsn.apps.googleusercontent.com", "project_id": "wattlesol-porject", "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token", "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs", "client_secret": "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", "redirect_uris": ["http://localhost"


def main():
  """Shows basic usage of the Google Calendar API.
  Prints the start and name of the next 10 events on the user's calendar.
  """
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

  try:
    service = build("calendar", "v3", credentials=creds)

    # Call the Calendar API
    now = datetime.datetime.utcnow().isoformat() + "Z"  # 'Z' indicates UTC time
    print("Getting the upcoming 10 events")
    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=10,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])

    if not events:
      print("No upcoming events found.")
      return

    # Prints the start and name of the next 10 events
    for event in events:
      start = event["start"].get("dateTime", event["start"].get("date"))
      print(start, event["summary"])

  except HttpError as error:
    print(f"An error occurred: {error}")


if __name__ == "__main__":
  main()


# import requests

# url = "https://hook.eu2.make.com/j0obx8toyvvvf998n731mmxyh9wycgrt"

# # Correct the key formatting in the dictionary
# params = {
#     "Event name": "Test Webhook",
#     "Start Time": "2025-02-15T14:00:00",
#     "end time": "2025-02-15T15:00:00"  # Ensure key names match what Make.com expects
# }

# # Send POST Request
# headers = {"Content-Type": "application/json"}
# response = requests.post(url, params=params, headers=headers)  # Use `json=params` instead of `params=params`

# # Check Response
# if response.status_code == 200:
#     print("Event created successfully!")
#     print("Raw Response:", response.text)  # Print raw response

#     try:
#         json_response = response.json()  # Try parsing JSON
#         print("JSON Response:", json_response)
#     except requests.exceptions.JSONDecodeError:
#         print("Response is not in JSON format.")

# else:
#     print(f"Failed to create event. Status Code: {response.status_code}")
#     print(response.text)
