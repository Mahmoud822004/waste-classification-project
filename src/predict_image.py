# =====================================================
# Waste Classification Test Script
# CNN (MobileNetV2) → SVM & KNN
# =====================================================

import cv2
import numpy as np
import joblib
from IPython.display import Image, display
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
FEATURE_EXTRACTOR_PATH = "models/mobilenet_feature_extractor.h5"
SVM_MODEL_PATH = "models/svm_model.pkl"
KNN_MODEL_PATH = "models/knn_model.pkl"

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.5

CLASS_NAMES = [
    "glass",
    "paper",
    "cardboard",
    "plastic",
    "metal",
    "trash"
]

# -----------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------
print("[INFO] Loading models...")

feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)
svm_classifier = joblib.load(SVM_MODEL_PATH)
knn_classifier = joblib.load(KNN_MODEL_PATH)

print("[INFO] Models loaded successfully!\n")

# -----------------------------------------------------
# PREDICTION FUNCTION
# -----------------------------------------------------
def predict_image(image_path, show_image=True, confidence_threshold=CONFIDENCE_THRESHOLD):

    if show_image:
        display(Image(image_path, width=224))

    # -------------------------
    # Load & preprocess image
    # -------------------------
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(" Could not load image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img * 255.0)

    # -------------------------
    # CNN Feature Extraction
    # -------------------------
    features = feature_extractor.predict(img, verbose=0)

    # -------------------------
    # SVM Prediction
    # -------------------------
    svm_pred = svm_classifier.predict(features)[0]
    svm_class = CLASS_NAMES[svm_pred]

    svm_confidence = None
    if hasattr(svm_classifier, "predict_proba"):
        svm_proba = svm_classifier.predict_proba(features)[0]
        svm_confidence = np.max(svm_proba)

    # -------------------------
    # KNN Prediction
    # -------------------------
    knn_pred = knn_classifier.predict(features)[0]
    knn_class = CLASS_NAMES[knn_pred]

    knn_proba = knn_classifier.predict_proba(features)[0]
    knn_confidence = np.max(knn_proba)

    # -------------------------
    # UNKNOWN rejection
    # -------------------------
    if svm_confidence is not None and svm_confidence < confidence_threshold:
        svm_class = "UNKNOWN"

    if knn_confidence < confidence_threshold:
        knn_class = "UNKNOWN"

    # -------------------------
    # Print results
    # -------------------------
    print("\n" + "=" * 60)
    print("📊 PREDICTION RESULTS")
    print("=" * 60)

    print(f"\n🔹 SVM Prediction: {svm_class}")
    if svm_confidence is not None:
        print(f"   Confidence: {svm_confidence * 100:.2f}%")
        if svm_confidence < confidence_threshold:
            print(f"    Below threshold ({confidence_threshold*100:.0f}%)")

    print(f"\n🔹 KNN Prediction: {knn_class}")
    print(f"   Confidence: {knn_confidence * 100:.2f}%")
    if knn_confidence < confidence_threshold:
        print(f"    Below threshold ({confidence_threshold*100:.0f}%)")

    print("=" * 60 + "\n")

    return svm_class, knn_class

# -----------------------------------------------------
# TEST IMAGE
# -----------------------------------------------------
if __name__ == "__main__":

    test_image_path = "C:\\Users\\ahmed\\OneDrive\\Desktop\\waste-classification-project\\dataset\\images\\paper1.jpg"

    predict_image(
        test_image_path,
        confidence_threshold=0.5
    )
