from flask import Flask, request, jsonify, render_template
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from model.preprocess import prepare_image
from model.predict import predict

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model("model.h5", compile=False)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        file = request.files['file']

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # preprocess
        img = prepare_image(filepath)

        # prediction
        prediction = model.predict(img)

        print("PREDICTION:", prediction)

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        print("INDEX:", class_index)

        # load labels
        with open("classes.json") as f:
            labels = json.load(f)

        # ✅ FIX: label assign karo
        label = labels[class_index]
        label = str(label).strip().lower()

        print("FINAL LABEL:", label)

        # solutions
        solutions = {
            "rust": {
                "treatment": "Use fungicide like Propiconazole",
                "cause": "Fungal infection",
                "prevention": "Avoid moisture"
            },
            "blight": {
                "treatment": "Apply Mancozeb",
                "cause": "Fungal/Bacterial infection",
                "prevention": "Use resistant seeds"
            },
            "mildew": {
                "treatment": "Use sulfur-based fungicide",
                "cause": "Powdery mildew fungus",
                "prevention": "Improve air circulation"
            },
            "healthy": {
                "treatment": "No treatment needed",
                "cause": "Healthy crop",
                "prevention": "Maintain nutrients"
            }
        }

        return jsonify({
            "disease": label,
            "confidence": round(confidence, 2),
            **solutions.get(label, {
                "treatment": "General crop care",
                "cause": "Model prediction",
                "prevention": "Monitor plant"
            })
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🔥 Starting Flask server...")
    app.run(debug=True)