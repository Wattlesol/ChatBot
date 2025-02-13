from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import create_openai_functions_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

from xml.etree import ElementTree
import os, json, pickle
import warnings
import requests
import re

from appFiles.db_manager import DatabaseManager
from appFiles.tools import book_appointment , datetime_tool
from appFiles.common_func import format_booked_slots
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "chatbot_with_langchain"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class WattlesolChatBot:
    def __init__(self):
        current_dir = "appFiles"
        self.files_dir = os.path.join(current_dir,"important_files")
        self.history_dir = os.path.join(current_dir,"chat_histories")
        self.vector_store_path = os.path.join(current_dir,"faiss_index") 
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.files_dir, exist_ok=True)
        
        self.TOKEN_PATH = os.path.join(self.files_dir, 'token.json')

        self.db_manager = DatabaseManager()
        self.SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

        self.llm = ChatOpenAI(temperature=0.5)
        self.parser = StrOutputParser()

        self.vector_store = self.load_vector_store()
        self.system_prompt = self.load_base_prompts()
        self.retrieval_tool = create_retriever_tool(
            self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
            name="company_general_chat",
            description="""Useful when you need to answer general queries related to the company's services, policies, and FAQs.
            Not useful for handling appointment bookings or any user-specific account-related inquiries.""",
        )

        self.tools = [self.retrieval_tool, book_appointment, datetime_tool]
        self.agent_executor = self.get_coversational_agent_with_history(self.tools)

    def get_coversational_agent_with_history(self, tools):
        # Load the base system prompt
        system_prompt_dic = self.load_base_prompts()
        if system_prompt_dic:
            system_prompt = system_prompt_dic["system_prompt"]
        else:
            raise "NO system Prompt found"
        
        # Define the system message prompt with a placeholder for chat history
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(
                    prompt=PromptTemplate(input_variables=[], template=system_prompt)
                ),
                MessagesPlaceholder(variable_name='chat_history', optional=True),
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(input_variables=['input'], template='{input}')
                ),
                MessagesPlaceholder(variable_name='agent_scratchpad'),
            ]
        )

        # Create the agent with the updated prompt
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        return agent_executor

    def generate_ai_response(self, session_id, message):
        # Retrieve session data and history
        session_data = self.get_session_history(session_id)
        history = session_data["history"]
        booked_slots = session_data["booked_slots"]  # Get already booked slots

        # Pass the history (which now includes booked_slots) as part of the context for the agent
        conversational_agent_executor = RunnableWithMessageHistory(
            self.agent_executor,
            lambda session_id: history,
            input_messages_key="input",
            output_messages_key="output",
            history_messages_key="chat_history",
        )
        # 🛠 Invoke the agent executor with the updated context (booked_slots is now part of history)
        agent_response = conversational_agent_executor.invoke(
            {
                "input": message
            },
            config={"configurable": {"session_id": session_id}}
        )
        # Get the result from the agent response
        result = agent_response.get("output", "No response generated.")
        print(result)
        # Add user and AI messages to the history
        history.add_user_message(HumanMessage(content=message))
        history.add_ai_message(result)
        session_data["history"] = history
        # Save the updated session history
        self.save_session_history(session_id, session_data)
        return result
    
    def load_base_prompts(self):
        system_prompt = self.db_manager.get_latest_base_prompts()
        if system_prompt:
            return system_prompt
        return {}

    def load_vector_store(self):
        # Check if the vector store exists on disk
        if os.path.exists(f"{self.vector_store_path}/index.faiss") and os.path.exists(f"{self.vector_store_path}/index.pkl"):
            try:
                return FAISS.load_local(self.vector_store_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            except Exception as e:
                raise RuntimeError(f"Error loading vector store: {e}")
        else:
            # Retrieve the latest sitemap URLs from the database
            conn = self.db_manager.get_connection()
            cursor = conn.cursor(dictionary=True)

            # Get the latest sitemap and its URLs
            query = "SELECT DISTINCT sitemap_url, extracted_url FROM sitemaps ORDER BY created_at DESC"
            cursor.execute(query)
            rows = cursor.fetchall()

            if not rows:
                raise RuntimeError("No sitemap data found in the database.")

            # Extract the sitemap URLs from the database rows
            urls = [row['extracted_url'] for row in rows]

            # If no URLs are found, raise an error
            if not urls:
                raise RuntimeError("No URLs found in the latest sitemap.")

            # Scrape the URLs and create a vector store
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

        booked_slots = format_booked_slots()

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
 
    def rescrape_sitemap(self, sitemap_url):
        """
        Rescrape a sitemap and update the database with its URLs.
        """
        response = requests.get(sitemap_url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch sitemap. HTTP Status: {response.status_code}")

        sitemap_content = response.content
        root = ElementTree.fromstring(sitemap_content)
        namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        # Extract URLs from the sitemap
        urls = [url_elem.text for url_elem in root.findall('.//ns:loc', namespaces)]
        if not urls:
            raise RuntimeError("No URLs found in the sitemap.")

        # Save sitemap and URLs to the database
        self.db_manager.save_sitemap_to_db(sitemap_url, urls)

        # Update the vector store
        self.vector_store = self.scrape_urls_and_create_vector_store(urls)
        return len(urls)

    def generate_prompts_logic(self, company: str, company_details: str):
        try:
            query = f"Provide context for generating system prompts for {company} with details {company_details}."
            top_docs = self.vector_store.similarity_search(query, k=5)
            context = "\n\n".join([doc.page_content for doc in top_docs])
            example = {
                "system_prompt":'''
            You are a professional representative for Wattlesol, a leading solutions provider in AI-powered automation tools for businesses. Your responses should always be polite, professional, and concise, focusing on key points related to Wattlesol's expertise. For general inquiries, respond clearly in 2-3 sentences.

            If a user expresses interest in booking an appointment, follow this process:

            1. Collect user details – Ask for their full name, email, preferred appointment date and time, and duration (default: 30 minutes).
            2.Handle missing details – If any information is missing, prompt the user to provide it. If the date is unspecified, default to the next available day. If the time is not valid, suggest an alternative.
            3. Confirm appointment details – Summarize the provided details and ask the user to confirm with all details before proceeding.
            4. Book the appointment – If the user confirms, use the book_appointment tool to finalize the booking.
            5. Handle scheduling conflicts – If the chosen time slot is unavailable, notify the user and suggest an alternative.

            Maintain professionalism throughout and ensure clarity in the process.
            '''
            }

            # Step 3: Generate the LLM prompt
            llm_prompt = f"""
            You are an expert in designing system prompts for AI chatbots.

            **Objective:** 
            Create a system prompt tailored specifically for {company}. 
            The goal is to ensure that the AI assistant provides **polite, professional, and concise** responses. 
            It should handle **both normal user queries and function-calling queries** effectively.
            
            **The company details:**
            {company_details}

            **Example System_Prompt:**  
            {json.dumps(example,indent=4)}

            **Context (from the FAISS index):**  
            {context}

            **Task:**  
            - Based on the given context and company details, generate a professional **system-level prompt**.  
            - The AI should maintain a **polite, professional, and concise tone** while answering.  
            - It should handle **general inquiries and function-calling queries** effectively.  

            **Output Format:**  
            {{
                "system_prompt": "..."
            }}
            """

            llm_response = self.llm.invoke(llm_prompt, config={"max_tokens": 1000})
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

            self.system_prompt = generated_dict
            self.db_manager.save_base_prompts(generated_dict["system_prompt"])

            return generated_dict

        except Exception as e:
            raise RuntimeError(f"Error occurred while generating prompts: {str(e)}")
   