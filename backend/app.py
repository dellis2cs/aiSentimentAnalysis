from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("sentiment_model.pkl", "rb") as model_file:
    svm = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review_text = data.get("review", "")
    
    if not review_text:
        return jsonify({"error": "No review text provided"}), 400

    # Vectorize and predict
    review_vectorized = vectorizer.transform([review_text])
    prediction = svm.predict(review_vectorized)

    return jsonify({"sentiment": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
