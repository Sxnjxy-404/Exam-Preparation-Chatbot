import os
import traceback
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

import rag_chain  # our Ollama-based RAG pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data')
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'md'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=BASE_DIR, template_folder=BASE_DIR)
CORS(app)

# ---------------- Database Setup ----------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'
db = SQLAlchemy(app)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String, nullable=False)
    message = db.Column(db.Text, nullable=False)
    sender = db.Column(db.String, nullable=False)

with app.app_context():
    db.create_all()
# -------------------------------------------------


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return send_file(os.path.join(BASE_DIR, 'index.html'))


@app.route('/health', methods=['GET'])
def health():
    try:
        ready = rag_chain.is_ready()
        return jsonify({"status": "ready" if ready else "not ready", "rag_ready": ready})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        payload = request.get_json(force=True)
        query = payload.get("query", "").strip()
        user_id = payload.get("user_id", "guest")

        if not query:
            return jsonify({"response": "Please provide a question."}), 400

        # Save user message
        db.session.add(ChatHistory(user_id=user_id, message=query, sender="user"))
        db.session.commit()

        response_text = rag_chain.invoke(query)

        # Save bot message
        db.session.add(ChatHistory(user_id=user_id, message=response_text, sender="bot"))
        db.session.commit()

        return jsonify({"response": response_text})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"response": f"Server error: {str(e)}"}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        rag_chain.ingest_file(save_path)
        return jsonify({"message": f"{filename} uploaded and indexed successfully."})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/history', methods=['GET'])
def history():
    try:
        user_id = request.args.get("user_id", "guest")
        chats = ChatHistory.query.filter_by(user_id=user_id).all()
        history_data = [{"sender": c.sender, "message": c.message} for c in chats]
        return jsonify(history_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
