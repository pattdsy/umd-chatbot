from flask import Flask, render_template, request, jsonify, send_from_directory
import os

app = Flask(__name__)

# In-memory session dict (per user you'd use a real session in production)
session = {}

# Load chatbot logic
from umd_bot import get_response  # assuming your chatbot logic is in umd_bot.py

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    response = get_response(user_message, session)
    return jsonify({'response': response})

# Serve video file from static folder
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
   
