import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit 
import logging
import warnings
from appFiles.chatbot import WattlesolChatBot

# Suppress warnings
warnings.filterwarnings("ignore")

# Flask app setup
app = Flask(__name__)

# CORS setup to allow specific origin
CORS(app)

# Initialize SocketIO with allowed origins
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize the chatbot
chatbot = WattlesolChatBot()

# Set up logging
logging.basicConfig(level=logging.INFO)

# WebSocket connection event: Send a welcome message on connection
@socketio.on('connect')
def handle_connect():
    print("connected")
    emit("welcome", {"message": "Welcome to the Wattlesol Chat! How can I assist you today?"})

# WebSocket message event: Handle incoming chat messages
@socketio.on('message')
def handle_message(data):
    print("message recived")
    session_id = data.get("session_id")
    message = data.get("message")

    if not session_id or not message:
        emit("error", {"error": "Session ID and message are required"})
        return

    logging.info(f"Received message from session {session_id}: {message}")

    try:
        result = chatbot.generate_ai_response(session_id, message)
        emit("response", {"response": result})
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        emit("error", {"error": str(e)})

# HTTP POST endpoint to test if the server is responding
@app.route('/test', methods=['POST'])
def test():
    data = request.get_json()
    logging.info(f"Received POST request: {data}")
    return jsonify({"message": "Server is running!", "received_data": data})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=80, debug=True)
