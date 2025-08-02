# app.py
from flask import Flask, request, jsonify
from model.loader import build_sequence_and_metadata
from model.gin_lstm_model import load_model
import torch
from torch.nn.functional import sigmoid
from flask_cors import CORS
app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",  # for local development
    "https://mathgraphexplorer.netlify.app"  # for deployed frontend
])


# Load model and device once at startup
model, device = load_model()

@app.route("/predict_readiness", methods=["POST"])
def predict_readiness():
    data = request.json
    student_id = data.get("student_id")
    target_ccss = data.get("target_ccss")
    dok = int(data.get("dok", 1))

    if not student_id or not target_ccss:
        return jsonify({"error": "Missing student_id or target_ccss"}), 400

    try:
        sequence, meta = build_sequence_and_metadata(student_id, target_ccss, dok, device)

        with torch.no_grad():
            output = model(sequence)
            prob = sigmoid(output[-1]).item()
            pred = int(prob > 0.5)

        return jsonify({
            "student_id": student_id,
            "target_ccss": target_ccss,
            "dok": dok,
            "readiness_score": round(prob, 4),
            "ready": bool(pred),
            "timeline_img": meta["timeline_img"],
            "graph_img": meta["graph_img"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
