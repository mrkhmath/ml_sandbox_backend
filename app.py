# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from model.predict import run_inference
from utils.graph_json import get_graph_json

def create_app() -> Flask:
    app = Flask(__name__)

    # --- CORS configuration ---
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "https://mathgraphexplorer.netlify.app",
    ]

    # Let Flask-CORS handle preflights and responses
    CORS(
        app,
        resources={r"/*": {"origins": ALLOWED_ORIGINS}},
        supports_credentials=False,  # flip to True only if you use cookies/auth
    )

    # Safety net to ensure errors also include ACAO
    @app.after_request
    def add_cors_headers(resp):
        origin = request.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
        return resp

    # --- Health & diagnostics ---
    @app.route("/", methods=["GET"])
    def index():
        return jsonify({"status": "Backend is live"}), 200

    @app.route("/_version", methods=["GET"])
    def version():
        return jsonify({
            "app": "srpm-backend",
            "env": os.environ.get("RENDER_SERVICE_NAME", "local"),
            "commit": os.environ.get("RENDER_GIT_COMMIT", "unknown"),
        }), 200

    # --- Main inference endpoint ---
    @app.route("/predict_readiness", methods=["POST"])
    def predict():
        # Parse JSON safely
        data = request.get_json(silent=True) or {}
        student_id = data.get("student_id")
        target_ccss = data.get("target_ccss")
        normalized_dok = data.get("normalized_dok")

        # Basic validation
        if student_id is None or target_ccss is None or normalized_dok is None:
            return jsonify({"error": "Missing required fields"}), 400

        try:
            # Run model inference (predict.py handles caching/CPU)
            prediction, probability = run_inference(student_id, target_ccss, normalized_dok)

            # Build graph payload for the UI
            graph_data = get_graph_json(student_id, target_ccss)  # must return {"nodes": [...], "links": [...]}

            return jsonify({
                "readiness": int(prediction),
                "probability": float(probability),
                "graph": graph_data
            }), 200

        except ValueError as ve:
            # Invalid inputs or missing resources -> 400
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            # Log server-side for debugging; return generic message to client
            print(f"[ERROR] /predict_readiness failed: {e}")
            return jsonify({"error": "Internal server error"}), 500

    # Optional: JSON-only error handlers (keeps responses consistent)
    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(_):
        return jsonify({"error": "Method not allowed"}), 405

    return app


# Gunicorn entrypoint: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300`
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Flask dev server (for local dev only). On Render, use Gunicorn as above.
    app.run(host="0.0.0.0", port=port, debug=False)
