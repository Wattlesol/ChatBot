# import eventlet
# eventlet.monkey_patch()
# import time
# start_time = time.time()
# from langchain_core.messages import HumanMessage
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from flask_socketio import SocketIO, emit 
# import logging
# import warnings

# from appFiles.chatbot import WattlesolChatBot

# warnings.filterwarnings("ignore", message="USER_AGENT environment variable not set, consider setting it to identify your requests.")

# # Flask app
# app = Flask(__name__)
# CORS(app, origins="*", methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Content-Type"])

# socketio = SocketIO(app, cors_allowed_origins="*")

# # Set up logging
# logging.basicConfig(level=logging.INFO)

# chatbot = WattlesolChatBot()

# # WebSocket connection event: Send a welcome message on connection
# @socketio.on('connect')
# def handle_connect():
#     print("connected")
#     emit("welcome", {"message": "Welcome to the Wattlesol Chat! How can I assist you today?"})

# # WebSocket message event: Handle incoming chat messages
# @socketio.on('message')
# def handle_message(data):
#     print("message recived")
#     session_id = data.get("session_id")
#     message = data.get("message")

#     if not session_id or not message:
#         emit("error", {"error": "Session ID and message are required"})
#         return

#     logging.info(f"Received message from session {session_id}: {message}")

#     try:
#         result = chatbot.generate_ai_response(session_id, message)
#         emit("response", {"response": result})
#     except Exception as e:
#         logging.error(f"Error generating response: {e}")
#         emit("error", {"error": str(e)})

# @app.route('/rescrape', methods=['POST'])
# def rescrape():
#     data = request.json
#     sitemap_url = data.get("sitemap_url")
#     if not sitemap_url:
#         return jsonify({"error": "Sitemap URL is required"}), 400
#     try:
#         url_count = chatbot.rescrape_sitemap(sitemap_url)
#         return jsonify({"message": "Rescraping completed successfully.", "url_count": url_count})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/generate-ai-prompts', methods=['POST'])
# def generate_ai_prompts():
#     try:
#         data = request.json
#         company = data.get("company")
#         company_details = data.get("company_details")
#         if not company or not company_details:
#             return jsonify({"error": "Session ID and message are required"}), 400
#         result = chatbot.generate_prompts_logic(company,company_details)
#         return jsonify({"message": "Prompts generated successfully.", **result})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/base-prompts', methods=['GET'])
# def get_latest_base_prompts():
#     try:
#         prompts = chatbot.db_manager.get_latest_base_prompts()
#         if not prompts:
#             return jsonify({"error": "No base prompts found."}), 404
#         return jsonify(prompts)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/chat-history', methods=['GET'])
# def get_chat_history():
#     try:
#         session_id = request.args.get('session_id')
#         # print(session_id)  # Use query param for GET
#         if not session_id:
#             return jsonify({"error": "session_id is required"}), 400
        
#         session_data = chatbot.get_session_history(session_id)
#         history = session_data["history"]
        
#         formatted_history = [
#             {
#                 "role": "User" if isinstance(msg, HumanMessage) else "Bot",
#                 "content": msg.content
#             }
#             for msg in history.messages
#         ]     
#         return jsonify({
#             "session_id": session_id,
#             "history": formatted_history,
#             "booked_slots": session_data["booked_slots"]
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     socketio.run(app, host="0.0.0.0", port=80, debug=True, allow_unsafe_werkzeug=True)

import eventlet
eventlet.monkey_patch()
import time
start_time = time.time()

from langchain_core.messages import HumanMessage
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging
import warnings
import json

from appFiles.chatbot import WattlesolChatBot

warnings.filterwarnings("ignore", message="USER_AGENT environment variable not set, consider setting it to identify your requests.")

# Flask app
app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Content-Type"])

socketio = SocketIO(app, cors_allowed_origins="*")

# Set up logging
logging.basicConfig(level=logging.INFO)

chatbot = WattlesolChatBot()

# Dictionary to track connected sessions
connected_sessions = {}

# WebSocket connection event: Send a welcome message on connection
@socketio.on('connect')
def handle_connect():
    session_id = request.args.get("session_id")  # Get session ID from query parameters
    sid = request.sid  # Unique Socket.IO session ID

    if session_id:
        connected_sessions[sid] = session_id  # Store session mapping

    print(f"Client connected: Session ID: {session_id}, Socket ID: {sid}")

    emit("welcome", {"message": "Welcome to the Wattlesol Chat! How can I assist you today?"})

# WebSocket message event: Handle incoming chat messages
@socketio.on('message')
def handle_message(data):
    print("Message received:", data)

    # Ensure data is a dictionary
    if isinstance(data, str):  # If data is a string, parse it as JSON
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            emit("error", {"error": "Invalid JSON format"})
            return

    session_id = data.get("session_id")
    message = data.get("message")

    if not session_id or not message:
        emit("error", {"error": "Session ID and message are required"})
        return

    logging.info(f"Received message from session {session_id}: {message}")

    try:
        result = chatbot.generate_ai_response(session_id, message)
        print(f"Sending response to session {session_id}: {result}")  # Debugging
        emit("response", {"response": result})
        print("Sent")
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        emit("error", {"error": str(e)})

# WebSocket disconnect event: Remove session tracking
@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid  # Get the socket session ID
    session_id = connected_sessions.pop(sid, None)  # Remove from dictionary

    print(f"Client disconnected: Session ID: {session_id}, Socket ID: {sid}")

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
        data = request.json
        company = data.get("company")
        company_details = data.get("company_details")
        if not company or not company_details:
            return jsonify({"error": "Session ID and message are required"}), 400
        result = chatbot.generate_prompts_logic(company, company_details)
        return jsonify({"message": "Prompts generated successfully.", **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/base-prompts', methods=['GET'])
def get_latest_base_prompts():
    try:
        prompts = chatbot.db_manager.get_latest_base_prompts()
        if not prompts:
            return jsonify({"error": "No base prompts found."}), 404
        return jsonify(prompts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat-history', methods=['GET'])
def get_chat_history():
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        
        session_data = chatbot.get_session_history(session_id)
        history = session_data["history"]
        
        formatted_history = [
            {
                "role": "User" if isinstance(msg, HumanMessage) else "Bot",
                "content": msg.content
            }
            for msg in history.messages
        ]     
        return jsonify({
            "session_id": session_id,
            "history": formatted_history,
            "booked_slots": session_data["booked_slots"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=80, debug=True, allow_unsafe_werkzeug=True)
