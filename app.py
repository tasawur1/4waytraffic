from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, this is the home page of trafficsystem-fyp!"

@app.route('/detect', methods=['POST'])
def detect():
    return "API is working! This is the /detect endpoint."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
