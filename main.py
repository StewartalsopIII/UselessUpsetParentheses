import os
from flask import Flask, request, jsonify, render_template
import groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Function to initialize Groq client
def initialize_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return groq.Groq(api_key=api_key)

# Initialize Groq client
try:
    client = initialize_groq_client()
except ValueError as e:
    print(f"Error initializing Groq client: {e}")
    client = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    if client is None:
        return jsonify({"error": "Groq client is not initialized. Please check your API key."}), 500
    try:
        user_input = request.json['user_input']
        # Create chat completion
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            model="mixtral-8x7b-32768",
        )
        # Get response content
        response = chat_completion.choices[0].message.content
        return jsonify({"response": response})
    except groq.GroqError as e:
        return jsonify({"error": f"Groq API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# Vercel serverless function handler
def handler(event, context):
    return app.wsgi_app(event, context)

if __name__ == '__main__':
    if client is None:
        print("Warning: Application is running without a valid Groq client. API calls will fail.")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))