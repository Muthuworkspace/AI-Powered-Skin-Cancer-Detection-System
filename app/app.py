"""
app.py — Skin Cancer Detection Web App
Runs locally on your PC. No Colab, no ngrok needed.
"""

import os
import json
import uuid
import numpy as np
import cv2
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tf_keras as keras
from flask import Flask, request, render_template, redirect, url_for

# ─────────────────────────────────────────────────────────────
# PATHS  (everything relative to this file — works on any PC)
# ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
import gdown

MODEL_PATH = os.path.join(BASE_DIR, "skin_cancer_model.h5")

# Download model automatically if missing
if not os.path.exists(MODEL_PATH):

    print("Model not found. Downloading from Google Drive...")

    file_id = "1CysZpmmVbOR5X29lvgnpZULmDY3lkUGs"

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(url, MODEL_PATH, quiet=False)

    print("Model downloaded successfully!")

CLASS_JSON   = os.path.join(BASE_DIR, "class_names.json")
UPLOAD_DIR   = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.secret_key = "sk_local_dev_2024"

# ─────────────────────────────────────────────────────────────
# LOAD MODEL  (tf_keras handles the old h5 format correctly)
# ─────────────────────────────────────────────────────────────
print("[*] Loading model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\n\n  Model not found at: {MODEL_PATH}\n"
        f"  Place skin_cancer_model.h5 in the same folder as app.py\n"
    )

model = keras.models.load_model(MODEL_PATH, compile=False)
print(f"[✓] Model loaded — Input: {model.input_shape}  |  Classes: {model.output_shape[-1]}")

# ─────────────────────────────────────────────────────────────
# CLASS NAMES
# ─────────────────────────────────────────────────────────────
if not os.path.exists(CLASS_JSON):
    raise FileNotFoundError(
        f"\n\n  class_names.json not found at: {CLASS_JSON}\n"
        f"  Copy it from your Google Drive to the same folder as app.py\n"
        f"  Format: [\"melanoma\", \"bcc\", \"scc\", ...] (9 items in training order)\n"
    )

with open(CLASS_JSON, "r") as f:
    CLASS_NAMES = json.load(f)

print(f"[✓] Class names loaded: {CLASS_NAMES}")

# ─────────────────────────────────────────────────────────────
# HUMAN-READABLE LABELS
# ─────────────────────────────────────────────────────────────
CLASS_MAP = {
    "melanoma":       "Melanoma",
    "bcc":            "Basal Cell Carcinoma",
    "scc":            "Squamous Cell Carcinoma",
    "a_keratosis":    "Actinic Keratosis",
    "pgk":            "Benign Keratosis",
    "dermatofibroma": "Dermatofibroma",
    "nevus":          "Melanocytic Nevi",
    "vascularLesion": "Vascular Lesion",
    "normal":         "Normal Skin",
}

IMG_SIZE   = 224
TTA_ROUNDS = 5      # test-time augmentation passes

# ─────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = keras.applications.efficientnet.preprocess_input(img.astype("float32"))
    return img

# ─────────────────────────────────────────────────────────────
# RISK LEVEL
# ─────────────────────────────────────────────────────────────
def get_risk(disease: str, confidence: float):
    if confidence < 60:
        return "UNCERTAIN", "Low confidence — please upload a clearer image.", "#7f8c8d"

    risk_map = {
        "Melanoma":               ("EMERGENCY", "High-risk cancer. See a dermatologist immediately.", "#c0392b"),
        "Basal Cell Carcinoma":   ("HIGH",      "Strong indication detected. Medical evaluation needed.", "#e67e22"),
        "Squamous Cell Carcinoma":("HIGH",      "Strong indication detected. Medical evaluation needed.", "#e67e22"),
        "Actinic Keratosis":      ("MODERATE",  "Potential pre-cancerous condition.", "#f39c12"),
        "Benign Keratosis":       ("LOW",       "Likely non-cancerous. Monitor for changes.", "#27ae60"),
        "Dermatofibroma":         ("LOW",       "Typically a benign skin condition.", "#27ae60"),
        "Melanocytic Nevi":       ("LOW",       "Monitor mole for shape/size/color changes.", "#27ae60"),
        "Vascular Lesion":        ("LOW",       "Usually a non-cancerous vascular condition.", "#27ae60"),
        "Normal Skin":            ("SAFE",      "No visible abnormality detected.", "#1abc9c"),
    }

    return risk_map.get(disease, ("LOW", "Consult a dermatologist for confirmation.", "#27ae60"))

# ─────────────────────────────────────────────────────────────
# SUGGESTION TEXT
# ─────────────────────────────────────────────────────────────
def get_suggestion(disease: str) -> str:
    suggestions = {
        "Melanoma":               "Immediate dermatologist consultation is strongly recommended.",
        "Basal Cell Carcinoma":   "Early medical treatment is advised. Avoid prolonged sun exposure.",
        "Squamous Cell Carcinoma":"Avoid UV exposure and consult a doctor soon.",
        "Actinic Keratosis":      "Use sunscreen daily and monitor skin condition regularly.",
        "Benign Keratosis":       "Usually harmless. Watch for rapid changes in size or color.",
        "Dermatofibroma":         "Typically benign. No immediate action needed.",
        "Melanocytic Nevi":       "Monitor using the ABCDE rule (Asymmetry, Border, Color, Diameter, Evolution).",
        "Vascular Lesion":        "Usually non-cancerous. Consult a doctor if it grows or bleeds.",
        "Normal Skin":            "Keep up your regular skincare routine.",
    }
    return suggestions.get(disease, "Consult a dermatologist for a professional opinion.")

# ─────────────────────────────────────────────────────────────
# PREDICTION  (with test-time augmentation)
# ─────────────────────────────────────────────────────────────
def predict(img_path: str) -> dict:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read image. Make sure it is a valid JPG/PNG file.")

    preds = []
    for _ in range(TTA_ROUNDS):
        aug = img.copy()
        if np.random.rand() > 0.5:
            aug = np.fliplr(aug)
        if np.random.rand() > 0.5:
            aug = np.flipud(aug)
        proc = preprocess(aug)
        pred = model.predict(np.expand_dims(proc, axis=0), verbose=0)[0]
        preds.append(pred)

    avg = np.mean(preds, axis=0)
    sorted_idx = np.argsort(avg)[::-1]

    results = []
    for idx in sorted_idx:
        raw_label = CLASS_NAMES[idx]
        label     = CLASS_MAP.get(raw_label, raw_label)
        confidence = round(float(avg[idx]) * 100, 2)
        results.append({"class": label, "confidence": confidence})

    top = results[0]
    risk, message, color = get_risk(top["class"], top["confidence"])
    suggestion = get_suggestion(top["class"])

    return {
        "top_class":  top["class"],
        "confidence": top["confidence"],
        "risk":       risk,
        "message":    message,
        "color":      color,
        "suggestion": suggestion,
        "all_results": results,
    }

# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    file = request.files.get("file")

    if not file or file.filename == "":
        return render_template("upload.html", error="Please select an image to upload.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        return render_template("upload.html", error="Unsupported file type. Use JPG or PNG.")

    try:
        filename  = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        data = predict(save_path)

        return render_template(
            "result.html",
            image_url   = f"/static/uploads/{filename}",
            top_class   = data["top_class"],
            confidence  = data["confidence"],
            risk        = data["risk"],
            message     = data["message"],
            color       = data["color"],
            suggestion  = data["suggestion"],
            all_results = data["all_results"],
        )

    except Exception as e:
        print(f"[ERROR] {e}")
        return render_template("upload.html", error=f"Prediction failed: {str(e)}")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/health")
def health():
    return {
        "status":       "running",
        "model_loaded": True,
        "num_classes":  len(CLASS_NAMES),
        "classes":      CLASS_NAMES,
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Skin Cancer Detector — running on:")
    print("  http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
