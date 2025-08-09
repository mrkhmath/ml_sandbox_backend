from flask import Flask, request, jsonify
from model.predict import run_inference
from utils.graph_json import get_graph_json
from flask_cors import CORS
import os

app = Flask(__name__)

allowed_origins = [
    "http://localhost:3000",
    "https://mathgraphexplorer.netlify.app"
]
CORS(app, origins=allowed_origins, supports_credentials=True)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "Backend is live"})

@app.route("/predict_readiness", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    student_id = data.get("student_id")
    target_ccss = data.get("target_ccss")
    normalized_dok = data.get("normalized_dok")

    if student_id is None or target_ccss is None or normalized_dok is None:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        prediction, probability = run_inference(student_id, target_ccss, normalized_dok)
        graph_data = get_graph_json(student_id, target_ccss)  # must return {'nodes': [...], 'links': [...]}

        return jsonify({
            "readiness": int(prediction),
            "probability": float(probability),
            "graph": graph_data
        })
    except Exception as e:
        print(f"Error in /predict_readiness: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
