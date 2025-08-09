from flask import Flask, request, jsonify
from model.predict import run_inference
from utils.graph_viz import render_graph_image
from utils.graph_json import get_graph_json
from flask_cors import CORS
import os

app = Flask(__name__)

# ✅ Define allowed frontend origins
allowed_origins = [
    "http://localhost:3000",
    "https://mathgraphexplorer.netlify.app/"
    "https://mathgraphexplorer.netlify.app/ml"
]

# ✅ Apply CORS globally (optional: restrict to specific routes)
CORS(app, supports_credentials=True)

@app.route("/", methods=["POST", "OPTIONS"])
def index():
    origin = request.headers.get("Origin")
    response = jsonify({"status": "Backend is live"})
    if origin in allowed_origins:
        response.headers.add("Access-Control-Allow-Origin", origin)
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, OPTIONS")
    return response

@app.route("/predict_readiness", methods=["POST", "OPTIONS"])
def predict():
    origin = request.headers.get("Origin")

    # ✅ Handle CORS preflight
    if request.method == "OPTIONS":
        response = jsonify({"status": "CORS preflight passed"})
        if origin in allowed_origins:
            response.headers.add("Access-Control-Allow-Origin", origin)
            response.headers.add("Access-Control-Allow-Headers", "Content-Type")
            response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    # ✅ Parse and validate input
    data = request.get_json()
    student_id = data.get("student_id")
    target_ccss = data.get("target_ccss")
    normalized_dok = data.get("normalized_dok")

    if not all([student_id, target_ccss, normalized_dok is not None]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # ✅ Run model inference
        prediction, probability = run_inference(student_id, target_ccss, normalized_dok)

        # ✅ Generate graph data
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
        print(f"Error in /predict_readiness: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ Use PORT from environment for deployment
    app.run(host="0.0.0.0", port=port)
