# ML Sandbox Backend (Flask)

This is a standalone backend service to serve GIN-LSTM readiness predictions for student assessment data.

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

## 🔁 API Usage
### POST /predict_readiness
**Body:**
```json
{
  "student_id": "abc123",
  "target_ccss": "8.EE.2",
  "dok": 2
}
```

**Response:**
```json
{
  "student_id": "abc123",
  "target_ccss": "8.EE.2",
  "dok": 2,
  "readiness_score": 0.7821,
  "ready": true,
  "timeline_img": "data:image/png;base64,...",
  "graph_img": "data:image/png;base64,..."
}
```