from flask import Flask, request, jsonify
from supabase import create_client
from supabase import Client

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask Backend is running!"

if __name__ == "__main__":
    app.run(debug=True)
