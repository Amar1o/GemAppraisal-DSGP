from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase_client import supabase
import joblib

app = Flask(__name__)
CORS(app)

@app.route('/test-connection', methods=['GET'])
def test_connection():
    return jsonify({'message': 'Backend connection successful!'})

if __name__ == "__main__":
    app.run(debug=True)
