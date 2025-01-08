from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import os , json
import pickle
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import warnings
from datetime import datetime, timedelta ,timezone
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from pytz import utc  # Add this for timezone handling
import requests
from xml.etree import ElementTree
import re

warnings.filterwarnings("ignore")


# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "chatbot_with_langchain"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)


sitemap_url = "https://wattlesol.com/sitemap.xml"

# scraping the data from all the given urls and save them into faiss db in faiss
def scrape_urls_and_create_vector_store(urls):
    all_documents = []

    # Load content from each URL
    for url in urls:
        try:
            print(f"Loading content from: {url}")
            loader = WebBaseLoader(url)
            documents = loader.load()
            all_documents.extend(documents)
        except Exception as e:
            print(f"Failed to load content from {url}: {e}")

    # Split content into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_documents = text_splitter.split_documents(all_documents)

    # Create embeddings and build FAISS vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(split_documents, embeddings)

    # Save the vector store locally
    vector_store.save_local(vector_store_path)
    print(f"Vector store saved at {vector_store_path}")
    return vector_store

# loading the already scraped or new scrap and load
def load_vector_store():
    # Check if both files exist
    if os.path.exists(f"{vector_store_path}/index.faiss") and os.path.exists(f"{vector_store_path}/index.pkl"):
        try:
            # Allow dangerous deserialization to load .pkl files safely
            vector_store = FAISS.load_local(vector_store_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            print("Vector store loaded successfully.")
            return vector_store
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise e
    else:
        sitemap_file = os.path.join(files_dir,"sitemap_urls.json")
        with open(sitemap_file,'r')as f:
            sitemap_data = json.load(f)
            urls = sitemap_data['urls']        
        return scrape_urls_and_create_vector_store(urls)

# Session history management
def get_session_history(session_id: str):
    """
    Retrieve session history from disk if it exists.
    If not, fetch calendar data, initialize a new session, and save it.
    """
    filepath = os.path.join(history_dir, f"{session_id}.pkl")
    if os.path.exists(filepath):
        # Load existing session history
        with open(filepath, "rb") as f:
            return pickle.load(f)

    # If no session file exists, fetch calendar data and initialize a new session
    print(f"No session file found for {session_id}. Initializing a new session.")

    events = fetch_calendar_events()
    booked_slots = format_booked_slots(events)

    # Initialize session history
    history = InMemoryChatMessageHistory()

    # Save the new session history to a file
    session_data = {"history": history, "booked_slots": booked_slots}
    save_session_history(session_id, session_data)

    return session_data

# Saving the session History
def save_session_history(session_id: str, session_data: dict):
    """
    Save session history and associated data to disk.
    """
    filepath = os.path.join(history_dir, f"{session_id}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(session_data, f)

# Trim and get the last 10 chat messages history
def format_history(history, max_messages=10):
    """
    Format the conversation history into a readable string for the LLM,
    trimming to the most recent `max_messages`.
    """
    # Take the most recent `max_messages`
    recent_messages = history.messages[-max_messages:]
    return "\n".join(
        [
            f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Bot: {msg.content}"
            for msg in recent_messages
        ]
    )

# Getting All Google calender Booking Events
def fetch_calendar_events():
    """
    Fetch all booked slots from the current date and time onward.
    """
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        service = build('calendar', 'v3', credentials=creds)

        # Fetch events from the current time onward
        now = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()  # Current UTC time

        events_result = service.events().list(
            calendarId='primary',
            timeMin=now,  # Start from now
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        return events
    except Exception as e:
        print(f"Error fetching calendar events: {e}")
        raise e

# Formating the The Calender Booking events
def format_booked_slots(events):
    """
    Format booked slots into a readable string for the LLM.
    """
    booked_slots = []
    for event in events:
        # Parse start and end times
        start = datetime.fromisoformat(event["start"]["dateTime"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(event["end"]["dateTime"].replace("Z", "+00:00"))
        # Format as: Day, Date, Time Range
        slot = f"{start.strftime('%A, %b %d, %Y, %I:%M %p')} to {end.strftime('%I:%M %p')}"
        booked_slots.append(slot)
    return "\n".join(booked_slots)

# Detect intent for booking appointments or regular query
def detect_intent(message):
    keywords = ["book", "appointment", "schedule", "meeting"]
    for keyword in keywords:
        if keyword in message.lower():
            return "appointment_booking"
    return "regular_query"

# generate a llm prompt for chat either for appointment or regular query
# def customize_prompt(message:str, booked_slots):
#     """
#     Generate a customized prompt based on user intent, focusing only on booked slots.
#     """
#     intent = detect_intent(message)
#     if intent == "appointment_booking":
#         # Format booked slots for context
#         booked_slots_text = booked_slots if booked_slots else "No booked slots available."

#         return f"""
#         You are a professional representative for Wattlesol (Not The CEO), a leading solutions provider.
#         The user has expressed interest in booking an appointment. Here is their message:
#         "{message}"

#         Current Context:
#         - Office working hours: 09:00 AM to 05:00 PM.
#         - Working days: Monday to Friday.
#         - Booked slots:
#         {booked_slots_text}

#         Task:
#         Based on the user's message and the context provided:
#         - Suggest the best available times for booking an appointment within office hours.
#         - Avoid conflicts with the booked slots provided.
#         - Ensure the response is polite, professional, and formatted with exact date, day, and time in AM/PM format.
#         """

#     # Default response for other intents
#     return f"""
#     You are a professional representative for Wattlesol (Not The CEO), a leading solutions provider. Respond politely and concisely, focusing on key points directly related to Wattlesol's expertise. Limit responses to 2-3 sentences while ensuring clarity and professionalism.

#     User query: {message}
#     """

# customize a llm prompt for chat either for appointment or regular query
def customize_prompt(message: str, booked_slots: str, base_prompts:dict):
    """
    Customize a base prompt based on the user's intent and additional context.

    Args:
        message (str): The user's message.
        booked_slots (str): A string of booked slots for appointment booking.
        base_prompts (dict): The path to the JSON file containing the base prompts.

    Returns:
        str: A customized prompt based on the user's intent.
    """
    try:
        # Detect user intent
        intent = detect_intent(message)

        if intent == "appointment_booking":
            # Format the "booking_prompt" dynamically with booked slots and the user's message
            base_prompt = base_prompts.get("booking_prompt", "")
            booked_slots_text = booked_slots if booked_slots else "No booked slots available."
            customized_prompt = base_prompt.format(message=message, booked_slots_text=booked_slots_text)
        else:
            # Format the "regular_prompt" dynamically with the user's message
            base_prompt = base_prompts.get("regular_prompt", "")
            customized_prompt = base_prompt.format(message=message)

        return customized_prompt

    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"

def generate_prompts_logic(vector_store, model, files_dir):
    """
    Generate prompts based on the vector store context, invoke the LLM, and save results to a file.

    Args:
        vector_store: The FAISS vector store object.
        model: The LLM model object.
        files_dir: Directory to save generated prompts.

    Returns:
        dict: A dictionary containing the generated prompts and the file path.
    """
    try:
        # Step 1: Get context from the vector store
        top_docs = vector_store.similarity_search("Provide context for generating prompts", k=5)
        context = "\n\n".join([doc.page_content for doc in top_docs])

        # Step 2: Define the example dictionary
        example_dict = {
            "booking_prompt": """You are a professional representative for Wattlesol (Not The CEO), a leading solutions provider.
            The user has expressed interest in booking an appointment. Here is their message:
            "{message}"

            Current Context:
            - Office working hours: 09:00 AM to 05:00 PM.
            - Working days: Monday to Friday.
            - Booked slots:
            {booked_slots_text}

            Task:
            Based on the user's message and the context provided:
            - Suggest the best available times for booking an appointment within office hours.
            - Avoid conflicts with the booked slots provided.
            - Ensure the response is polite, professional, and formatted with exact date, day, and time in AM/PM format.
            """,
            "regular_prompt": """You are a professional representative for Wattlesol (Not The CEO), a leading solutions provider.
            Respond politely and concisely, focusing on key points directly related to Wattlesol's expertise. Limit responses to 2-3 sentences while ensuring clarity and professionalism.

            User query: {message}
            """
        }

        # Step 3: Generate the LLM prompt
        llm_prompt = f"""
        Below is an example of a dictionary containing two prompts: one for booking appointments and one for regular queries. 
        The examples are specific to Wattlesol. Use the following example to generate similar prompts tailored to the website 
        represented by the FAISS index context.

        Example Dictionary:
        {json.dumps(example_dict, indent=4)}

        Context:
        {context}

        Task:
        Based on the above context:
        - Generate a JSON object with two keys: "booking_prompt" and "regular_prompt".
        - Ensure that the generated prompts align with the context provided by the FAISS index.
        - The style and tone of the prompts must follow the examples provided above.
        - Ensure that the "booking_prompt" helps users book an appointment professionally, avoiding scheduling conflicts.
        - Ensure that the "regular_prompt" provides concise, polite answers related to the expertise of the company.

        Output Format:
        {{
            "booking_prompt": "...",
            "regular_prompt": "..."
        }}
        """

        # Step 4: Invoke LLM and extract response
        llm_response = model.invoke(llm_prompt, config={"max_tokens": 1000})
        generated_text = llm_response.content

        # Use regex to extract the dictionary portion
        match = re.search(r"\{.*\}", generated_text, re.DOTALL)
        if match:
            json_text = match.group(0)
            try:
                generated_dict = json.loads(json_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {str(e)}\nRaw Output: {json_text}")
        else:
            raise ValueError(f"Failed to extract JSON from LLM output. Raw Output: {generated_text}")
        # update the Global base_prompts
        global base_prompts
        base_prompts = generated_dict
        # Step 5: Save the generated prompts to a file
        output_file = os.path.join(files_dir, "generated_prompts.json")
        with open(output_file, "w") as f:
            json.dump(generated_dict, f, indent=4)

        return {"prompts": generated_dict, "saved_to": output_file}
    except Exception as e:
        raise RuntimeError(f"Error occurred while generating prompts: {str(e)}")


# Initialize model and parser
model = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()

# Defining the important directories
history_dir = "chat_histories"
files_dir = "important_files"
# Ensure history directory exists
if not os.path.exists(history_dir):
    os.makedirs(history_dir)
if not os.path.exists(files_dir):
    os.makedirs(files_dir)
# VectorStore path
vector_store_path = "faiss_index"
# Google Calendar API settings
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
TOKEN_PATH = os.path.join(files_dir,'token.json')
# Initialize vector store
vector_store = load_vector_store()
# Load base prompts from the JSON file
prompt_file = os.path.join(files_dir,"generated_prompts.json")
with open(prompt_file, "r") as file:
    base_prompts = json.load(file)


# Retrieval chain for refining answers
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
retrieval_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    return_source_documents=True
)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        session_id = data.get("session_id")
        message = data.get("message")

        if not session_id or not message:
            return jsonify({"error": "Session ID and message are required"}), 400

        # Get or initialize session data
        session_data = get_session_history(session_id)
        history = session_data["history"]
        booked_slots = session_data["booked_slots"]

        # Format history for context
        formatted_history = format_history(history)
        # Generate the customized prompt
        message_prompt = customize_prompt(message,booked_slots,base_prompts)

        # Combine formatted history with the customized prompt
        query_with_history = f"{formatted_history}\n\n{message_prompt}"

        # Pass the combined query to the retrieval chain
        response = retrieval_chain.invoke({"query": query_with_history}, config={"max_tokens": 300})
        result = response.get("result", "No response generated.")
        source_documents = response.get("source_documents", [])

        # Add user and bot messages to the history
        history.add_user_message(HumanMessage(content=message))
        history.add_ai_message(result)
        session_data["history"] = history
        save_session_history(session_id, session_data)

        return jsonify({"response": result, "sources": [doc.page_content for doc in source_documents]})
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/rescrape', methods=['POST'])
def rescrape_sitemap():
    try:
        data = request.json
        sitemap_url = data.get("sitemap_url")
        print('Rescraping...:', sitemap_url)
        
        if not sitemap_url:
            return jsonify({"error": "Sitemap URL is required"}), 400
        
        # Fetch and parse the sitemap
        response = requests.get(sitemap_url)
        if response.status_code != 200:
            return jsonify({"error": f"Failed to fetch sitemap. HTTP Status: {response.status_code}"}), 500
        
        sitemap_content = response.content
        root = ElementTree.fromstring(sitemap_content)
        namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Extract all URLs from the sitemap
        urls = [url_elem.text for url_elem in root.findall('.//ns:loc', namespaces)]
        if not urls:
            return jsonify({"error": "No URLs found in the sitemap"}), 400
        
        print(f"Extracted {len(urls)} URLs from sitemap.")
        
        # Save the sitemap URL and extracted URLs to a JSON file
        output_data = {
            "sitemap_url": sitemap_url,
            "extracted_urls": urls
        }
        sitemap_file = os.path.join(files_dir,"sitemap_urls.json")
        with open(sitemap_file, "w") as file:
            json.dump(output_data, file, indent=4)
        
        print(f"Sitemap URL and extracted URLs saved to {sitemap_file}.")
        
        # Scrape and process the data
        new_vector_store = scrape_urls_and_create_vector_store(urls)
        
        # Replace the old FAISS index
        global vector_store
        vector_store = new_vector_store
        
        return jsonify({
            "message": "Rescraping completed and FAISS index updated successfully.",
            "url_count": len(urls),
            "saved_to": sitemap_file
        })
    except Exception as e:
        print(f"Error occurred during rescraping: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/generate-ai-prompts', methods=['POST'])
def generate_ai_prompts():
    try:
        # Ensure the vector store is loaded
        global vector_store
        if not vector_store:
            vector_store = load_vector_store()

        # Call the merged function
        result = generate_prompts_logic(vector_store, model, files_dir)

        return jsonify({"message": "Prompts generated successfully.", **result})
    except Exception as e:
        print(f"Error occurred while generating prompts: {e}")
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
