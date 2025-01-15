from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os, json, pickle
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
from datetime import datetime, timezone
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from xml.etree import ElementTree
import requests
import re
import mysql.connector

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "chatbot_with_langchain"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class DatabaseManager:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DB"),
            pool_name="chatbot_pool",
            pool_size=10,
        )
        self.initialize_database()

    def get_connection(self):
        if not self.conn.is_connected():
            self.conn.reconnect(attempts=3, delay=5)
        return self.conn

    def close_connection(self):
        if self.conn.is_connected():
            self.conn.close()

    def initialize_database(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_history (
                session_id VARCHAR(255) PRIMARY KEY,
                history BLOB NOT NULL,
                booked_slots TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS base_prompts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                booking_prompt TEXT NOT NULL,
                regular_prompt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    def save_base_prompts(self, booking_prompt, regular_prompt):
        conn = self.get_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO base_prompts (booking_prompt, regular_prompt)
            VALUES (%s, %s)
        """
        cursor.execute(query, (booking_prompt, regular_prompt))
        conn.commit()

    def get_latest_base_prompts(self):
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT booking_prompt, regular_prompt
            FROM base_prompts
            ORDER BY created_at DESC
            LIMIT 1
        """
        cursor.execute(query)
        return cursor.fetchone()

class WattlesolChatBot:
    def __init__(self):
        self.history_dir = "chat_histories"
        self.files_dir = "important_files"
        self.vector_store_path = "faiss_index"
        self.SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
        self.TOKEN_PATH = os.path.join(self.files_dir, 'token.json')
        self.db_manager = DatabaseManager()

        self.model = ChatGroq(model="llama-3.1-8b-instant")
        self.parser = StrOutputParser()

        self.base_prompts = self.load_base_prompts()
        self.vector_store = self.load_vector_store()
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.model,
            retriever=self.retriever,
            return_source_documents=True
        )

        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.files_dir, exist_ok=True)

    def load_base_prompts(self):
        prompts = self.db_manager.get_latest_base_prompts()
        if prompts:
            return {
                "booking_prompt": prompts["booking_prompt"],
                "regular_prompt": prompts["regular_prompt"]
            }
        return {}

    def load_vector_store(self):
        if os.path.exists(f"{self.vector_store_path}/index.faiss") and os.path.exists(f"{self.vector_store_path}/index.pkl"):
            try:
                return FAISS.load_local(self.vector_store_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            except Exception as e:
                raise RuntimeError(f"Error loading vector store: {e}")
        else:
            sitemap_file = os.path.join(self.files_dir, "sitemap_urls.json")
            with open(sitemap_file, 'r') as f:
                urls = json.load(f)['extracted_urls']
            return self.scrape_urls_and_create_vector_store(urls)

    def scrape_urls_and_create_vector_store(self, urls):
        all_documents = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                print(f"Failed to load content from {url}: {e}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_documents = text_splitter.split_documents(all_documents)
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(split_documents, embeddings)
        vector_store.save_local(self.vector_store_path)
        return vector_store

    def get_session_history(self, session_id):
        conn = self.db_manager.get_connection()
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM session_history WHERE session_id = %s"
        cursor.execute(query, (session_id,))
        result = cursor.fetchone()

        if result:
            history = InMemoryChatMessageHistory()
            history.messages = pickle.loads(result['history'])
            booked_slots = result['booked_slots']
            return {"history": history, "booked_slots": booked_slots}

        events = self.fetch_calendar_events()
        booked_slots = self.format_booked_slots(events)

        history = InMemoryChatMessageHistory()
        session_data = {"history": history, "booked_slots": booked_slots}
        self.save_session_history(session_id, session_data)
        return session_data

    def save_session_history(self, session_id, session_data):
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        query = "REPLACE INTO session_history (session_id, history, booked_slots) VALUES (%s, %s, %s)"
        cursor.execute(query, (session_id, pickle.dumps(session_data["history"].messages), session_data["booked_slots"]))
        conn.commit()

    def fetch_calendar_events(self):
        creds = Credentials.from_authorized_user_file(self.TOKEN_PATH, self.SCOPES)
        service = build('calendar', 'v3', credentials=creds)
        now = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        events_result = service.events().list(
            calendarId='primary',
            timeMin=now,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        return events_result.get('items', [])

    def format_booked_slots(self, events):
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

    def detect_intent(self, message):
        keywords = ["book", "appointment", "schedule", "meeting"]
        return "appointment_booking" if any(keyword in message.lower() for keyword in keywords) else "regular_query"

    def customize_prompt(self, message, booked_slots):
        intent = self.detect_intent(message)
        if intent == "appointment_booking":
            base_prompt = self.base_prompts.get("booking_prompt", "")
            booked_slots_text = booked_slots if booked_slots else "No booked slots available."
            return base_prompt.format(message=message, booked_slots_text=booked_slots_text)
        else:
            base_prompt = self.base_prompts.get("regular_prompt", "")
            return base_prompt.format(message=message)

    def generate_ai_response(self, session_id, message):
        session_data = self.get_session_history(session_id)
        history = session_data["history"]
        booked_slots = session_data["booked_slots"]
        formatted_history = self.format_history(history)  # Correct usage here
        message_prompt = self.customize_prompt(message, booked_slots)
        query_with_history = f"{formatted_history}\n\n{message_prompt}"

        response = self.retrieval_chain.invoke({"query": query_with_history}, config={"max_tokens": 300})
        result = response.get("result", "No response generated.")
        source_documents = response.get("source_documents", [])

        history.add_user_message(HumanMessage(content=message))
        history.add_ai_message(result)
        session_data["history"] = history
        self.save_session_history(session_id, session_data)

        return result, source_documents

    def format_history(self, history, max_messages=10):
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

    def rescrape_sitemap(self, sitemap_url):
        response = requests.get(sitemap_url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch sitemap. HTTP Status: {response.status_code}")

        sitemap_content = response.content
        root = ElementTree.fromstring(sitemap_content)
        namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        urls = [url_elem.text for url_elem in root.findall('.//ns:loc', namespaces)]
        sitemap_file = os.path.join(self.files_dir, "sitemap_urls.json")
        with open(sitemap_file, "w") as file:
            json.dump({"sitemap_url": sitemap_url, "extracted_urls": urls}, file, indent=4)

        self.vector_store = self.scrape_urls_and_create_vector_store(urls)
        return len(urls)

    def generate_prompts_logic(self):
        try:
            top_docs = self.vector_store.similarity_search("Provide context for generating prompts", k=5)
            context = "\n\n".join([doc.page_content for doc in top_docs])

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

            llm_response = self.model.invoke(llm_prompt, config={"max_tokens": 1000})
            generated_text = llm_response.content

            match = re.search(r"\{.*\}", generated_text, re.DOTALL)
            if match:
                json_text = match.group(0)
                try:
                    generated_dict = json.loads(json_text)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON: {str(e)}\nRaw Output: {json_text}")
            else:
                raise ValueError(f"Failed to extract JSON from LLM output. Raw Output: {generated_text}")

            self.base_prompts = generated_dict

            self.db_manager.save_base_prompts(
                booking_prompt=generated_dict["booking_prompt"],
                regular_prompt=generated_dict["regular_prompt"]
            )

            return {"prompts": generated_dict}
        except Exception as e:
            raise RuntimeError(f"Error occurred while generating prompts: {str(e)}")
# Flask app
app = Flask(__name__)
CORS(app)
chatbot = WattlesolChatBot()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    session_id = data.get("session_id")
    message = data.get("message")
    if not session_id or not message:
        return jsonify({"error": "Session ID and message are required"}), 400
    try:
        result, sources = chatbot.generate_ai_response(session_id, message)
        return jsonify({"response": result, "sources": [doc.page_content for doc in sources]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/rescrape', methods=['POST'])
def rescrape():
    data = request.json
    sitemap_url = data.get("sitemap_url")
    if not sitemap_url:
        return jsonify({"error": "Sitemap URL is required"}), 400
    try:
        url_count = chatbot.rescrape_sitemap(sitemap_url)
        return jsonify({"message": "Rescraping completed successfully.", "url_count": url_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-ai-prompts', methods=['POST'])
def generate_ai_prompts():
    try:
        result = chatbot.generate_prompts_logic()
        return jsonify({"message": "Prompts generated successfully.", **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
