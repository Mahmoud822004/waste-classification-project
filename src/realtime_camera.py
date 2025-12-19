# =====================================================
# Real-Time Waste Classification
# CNN (MobileNetV2) → SVM & KNN
# =====================================================

import cv2
import numpy as np
import joblib
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
def predict_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    img_pp = preprocess_input(img * 255.0)

    features = feature_extractor.predict(img_pp, verbose=0)

    # SVM
    svm_pred = svm_classifier.predict(features)[0]
    svm_class = CLASS_NAMES[svm_pred]
    svm_confidence = None
    if hasattr(svm_classifier, "predict_proba"):
        svm_confidence = np.max(svm_classifier.predict_proba(features)[0])
        if svm_confidence < CONFIDENCE_THRESHOLD:
            svm_class = "UNKNOWN"

    # KNN
    knn_pred = knn_classifier.predict(features)[0]
    knn_class = CLASS_NAMES[knn_pred]
    knn_confidence = np.max(knn_classifier.predict_proba(features)[0])
    if knn_confidence < CONFIDENCE_THRESHOLD:
        knn_class = "UNKNOWN"

    return svm_class, knn_class, svm_confidence, knn_confidence

# -----------------------------------------------------
# REAL-TIME CAMERA LOOP
# -----------------------------------------------------
def run_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Cannot open camera")
        return

    print("[INFO] Press 'q' to exit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to grab frame")
            break

        svm_class, knn_class, svm_conf, knn_conf = predict_frame(frame)

        # Overlay predictions
        cv2.putText(frame, f"SVM: {svm_class} ({svm_conf*100:.1f}%)" if svm_conf else f"SVM: {svm_class}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"KNN: {knn_class} ({knn_conf*100:.1f}%)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Waste Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    run_camera()
