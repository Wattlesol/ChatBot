
from langchain_core.messages import HumanMessage
from flask import Flask, request, jsonify
from flask_cors import CORS

import warnings

from appFiles.chatbot import WattlesolChatBot

warnings.filterwarnings("ignore")
   

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
        result = chatbot.generate_ai_response(session_id, message)
        return jsonify({"response": result})
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
        data = request.json
        company = data.get("company")
        company_details = data.get("company_details")
        if not company or not company_details:
            return jsonify({"error": "Session ID and message are required"}), 400
        result = chatbot.generate_prompts_logic(company,company_details)
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
        print(session_id)  # Use query param for GET
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        
        session_data = chatbot.get_session_history(session_id)
        print(session_data)
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
    app.run(host="0.0.0.0", port=8000, debug=True)