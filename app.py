# from flask import Flask, request, jsonify, render_template, session
# from chatbot import get_response
# import os
# # request - object containing incoming data from the client/browser
# # jsonify - helper function that conveys Python dictionaries to JSON (web-friendly format)
# # render_template renders HTML pages in Flask
# # session stores data temporarily between different user interactions (stateful conversation)


# # Initialize Flask app and creates an instance of a Flask web application
# app = Flask(__name__)

# # Secret key for session management, sets a secret key for the flask application to securely manage sessions. 
# # os.urandom(24) generates a secure random 24-byte string
# app.secret_key = os.urandom(24)

# # home route 
# @app.route('/') # decorator that defines the url route of the home page (localhost:5000/)
# def home():  # defines the function executed when the home URL is accessed
#     return render_template('chat.html') # displays your HTML interface defined in templates/chat.html

# # chatbot route 
# @app.route('/chat', methods=['POST']) # handles POST requests (sending user input to the server)
# def chat():
#     user_message = request.json.get('message') # request.json retrives JSON data sent by the frontend
#     response = get_response(user_message, session) # extracts the user's message from the JSON data
#     return jsonify({'response': response}) # sends back chatbot's response as JSON to the frontend

# # running the flask app
# if __name__ == '__main__': #runs the flask application if executed directly from the file
#     app.run(debug=True) #enables debugging and auto-reload and is useful for development
    
    
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
   
