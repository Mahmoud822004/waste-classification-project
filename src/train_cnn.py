# ============================================================
#  CNN (MobileNetV2) Feature Extraction + SVM / KNN Classifier
# ============================================================

import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ---------------------------
# CONFIG
# ---------------------------
DATA_DIR = "C:\\Users\\ahmed\\OneDrive\\Desktop\\waste-classification-project\\dataset"   
IMG_SIZE = (224, 224)
CLASS_NAMES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
RANDOM_STATE = 42


# ---------------------------
# SAFE DATA LOADER
# ---------------------------
def load_dataset(data_dir):
    X, y = [], []

    for label, cls in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, cls)
        images = glob(os.path.join(class_dir, "*"))

        for img_path in tqdm(images, desc=f"Loading {cls}"):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                img = img.astype(np.float32) / 255.0
                X.append(img)
                y.append(label)
            except:
                continue

    return np.array(X), np.array(y)


print("\n[INFO] Loading dataset...")
X, y = load_dataset(DATA_DIR)

print("[INFO] Total samples:", X.shape[0])


# ---------------------------
# TRAIN / VAL / TEST SPLIT
# ---------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=RANDOM_STATE
)

print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)


# ---------------------------
# DATA AUGMENTATION
# ---------------------------
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

def augment_data(X, y, copies=1):
    X_aug, y_aug = list(X), list(y)

    for i in tqdm(range(len(X)), desc="Augmenting"):
        for _ in range(copies):
            img = datagen.random_transform(X[i])
            X_aug.append(img)
            y_aug.append(y[i])

    return np.array(X_aug), np.array(y_aug)


print("\n[INFO] Augmenting training data...")
X_train_aug, y_train_aug = augment_data(X_train, y_train, copies=1)


# ---------------------------
# MOBILENET FEATURE EXTRACTOR
# ---------------------------
print("\n[INFO] Building MobileNetV2 feature extractor...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=x)

print("Feature vector size:", feature_extractor.output_shape)


# ---------------------------
# FEATURE EXTRACTION
# ---------------------------
def extract_features(X, model, batch_size=32):
    X_pp = preprocess_input(X * 255.0)
    return model.predict(X_pp, batch_size=batch_size, verbose=1)


print("\n[INFO] Extracting CNN features...")
X_train_feats = extract_features(X_train_aug, feature_extractor)
X_val_feats   = extract_features(X_val, feature_extractor)
X_test_feats  = extract_features(X_test, feature_extractor)


# ---------------------------
# SVM CLASSIFIER
# ---------------------------
print("\n[INFO] Training SVM...")
svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
svm.fit(X_train_feats, y_train_aug)

y_val_pred = svm.predict(X_val_feats)
print("\nSVM Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred, target_names=CLASS_NAMES))


# ---------------------------
# KNN CLASSIFIER
# ---------------------------
print("\n[INFO] Training KNN...")
knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn.fit(X_train_feats, y_train_aug)

y_val_pred_knn = knn.predict(X_val_feats)
print("\nKNN Validation Accuracy:", accuracy_score(y_val, y_val_pred_knn))
print(classification_report(y_val, y_val_pred_knn, target_names=CLASS_NAMES))


# ---------------------------
# TEST SET RESULTS
# ---------------------------
print("\n[TEST] SVM Accuracy:", accuracy_score(y_test, svm.predict(X_test_feats)))
print("[TEST] KNN Accuracy:", accuracy_score(y_test, knn.predict(X_test_feats)))


# ---------------------------
# SAVE MODELS
# ---------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(svm, "models/svm_model.pkl")
joblib.dump(knn, "models/knn_model.pkl")
feature_extractor.save("models/mobilenet_feature_extractor.h5")

print("\n[DONE] Models saved in /models/")
