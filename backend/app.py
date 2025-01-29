from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase_client import supabase

app = Flask(__name__)
CORS(app)



if __name__ == "__main__":
    app.run(debug=True)
