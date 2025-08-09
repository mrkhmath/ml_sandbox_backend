from flask import Flask, request, jsonify
from model.predict import run_inference
from utils.graph_viz import render_graph_image
from flask_cors import CORS
from utils.graph_json import get_graph_json
import os

app = Flask(__name__)

# ✅ Allow both local and deployed frontend origins
allowed_origins = [
    "http://localhost:3000",
    "https://mathgraphexplorer.netlify.app"
]

CORS(app, resources={r"/predict_readiness": {"origins": allowed_origins}})

@app.route("/predict_readiness", methods=["POST", "OPTIONS"])
def predict():
    origin = request.headers.get("Origin")
    if request.method == "OPTIONS":
        # ✅ Handle preflight request
        response = jsonify({"status": "CORS preflight passed"})
        if origin in allowed_origins:
            response.headers.add("Access-Control-Allow-Origin", origin)
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    data = request.get_json()
    student_id = data.get("student_id")
    target_ccss = data.get("target_ccss")
    normalized_dok = data.get("normalized_dok")

    if not all([student_id, target_ccss, normalized_dok is not None]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        prediction, probability = run_inference(student_id, target_ccss, normalized_dok)
        graph_data = get_graph_json(student_id, target_ccss)

        response = jsonify({
            "readiness": int(prediction),
            "probability": float(probability),
            "graph": graph_data
        })
        if origin in allowed_origins:
            response.headers.add("Access-Control-Allow-Origin", origin)
        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # fallback for local dev
    app.run(host="0.0.0.0", port=port)
