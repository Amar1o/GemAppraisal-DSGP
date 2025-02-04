from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase_client import supabase
import joblib

app = Flask(__name__)
CORS(app)

@app.route('/price-calculator', methods=['GET'])
def test_connection():
    return jsonify({'message': 'Backend connection successful!'})

@app.route('/users')
def get_users():
    response = supabase.table("appraisals").select("*").execute()
    return jsonify(response.data)

if __name__ == "__main__":
    app.run(debug=True)
