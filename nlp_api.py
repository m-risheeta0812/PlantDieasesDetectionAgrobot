from flask import Flask, request, jsonify
import joblib

model = joblib.load("plantvillage_nlp_model.joblib")

app = Flask(__name__)

@app.post("/nlp")
def nlp_predict():
    data = request.json["query"]
    result = model.predict([data])[0]  # modify based on your model
    return jsonify({"reply": result})

app.run(port=5000)
