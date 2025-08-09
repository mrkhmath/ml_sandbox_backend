# app.py
import os
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from model.predict import run_inference
from utils.graph_json import get_graph_json

ALLOWED_ORIGINS = [
    "https://mathgraphexplorer.netlify.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app = Flask(__name__)

CORS(app, resources={r"/predict_readiness": {
    "origins": ALLOWED_ORIGINS,
    "methods": ["POST", "OPTIONS"],
    "allow_headers": ["Content-Type"],
}})

@app.after_request
def add_cors(resp):
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.route("/predict_readiness", methods=["OPTIONS"])
def preflight():
    return ("", 204)

# Inference
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
        graph_data = get_graph_json(student_id, target_ccss)  # should return {"nodes": [...], "links": [...]}
        return jsonify({
            "readiness": int(prediction),
            "probability": float(probability),
            "graph": graph_data
        }), 200

    except ValueError as ve:
        # Bad inputs / missing resources
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        # Minimal log on server; generic message to client
        print(f"[ERROR] /predict_readiness failed: {e}", flush=True)
        return jsonify({"error": "Internal server error"}), 500

# Consistent JSON errors
@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed"}), 405

# Gunicorn entrypoint:
# gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --threads 4 --access-logfile - --error-logfile -
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
