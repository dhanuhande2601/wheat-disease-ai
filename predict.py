import numpy as np
import json

# load class names once
with open("classes.json", "r") as f:
    labels = json.load(f)

def predict(model, img):
    prediction = model.predict(img)

    prediction = np.array(prediction)

    class_index = np.argmax(prediction)

    confidence = float(np.max(prediction)) * 100

    # ✅ FIX: dynamic labels from training
    label = labels[class_index]

    return {
        "label": label,
        "confidence": round(confidence, 2)
    }