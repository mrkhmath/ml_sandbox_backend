from flask import Flask, request, jsonify
from model.predict import run_inference
from utils.graph_viz import render_graph_image
from flask_cors import CORS
from utils.graph_json import get_graph_json


app = Flask(__name__)
CORS(app)  

@app.route("/predict_readiness", methods=["POST"])
def predict():
    data = request.get_json()
    student_id = data.get("student_id")
    target_ccss = data.get("target_ccss")
    normalized_dok = data.get("normalized_dok")

    if not all([student_id, target_ccss, normalized_dok is not None]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        prediction, probability = run_inference(student_id, target_ccss, normalized_dok)
        graph_data = get_graph_json(student_id, target_ccss)


        return jsonify({
            "readiness": int(prediction),
            "probability": float(probability),
            "graph": graph_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5050)

