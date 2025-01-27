from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase_client import supabase

app = Flask(__name__)
CORS(app)

@app.route('/register', methods=['POST'])
# Register a new user
def register_user():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    try:
        response = supabase.auth.sign_up({"email":email, "password":password})
        return jsonify({"message":"User registered successfully"})
    except Exception as e:
        return jsonify({"message":str(e)})

@app.route('/login', methods=['POST'])
# Login a user
def login_user():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return jsonify({"message":"User signed in successfully"})
    except Exception as e:
        return jsonify({"message":str(e)})


if __name__ == "__main__":
    app.run(debug=True)
