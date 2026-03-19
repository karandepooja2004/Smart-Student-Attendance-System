from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import numpy as np
import joblib

# ---------------------------------
# Load Embeddings File
# ---------------------------------
data = np.load("embeddings/face_embeddings.npz")

print("Keys inside npz file:", data.files)

# Automatically read embeddings and labels
X = data[data.files[0]]
y = data[data.files[1]]

print("Embeddings Shape:", X.shape)
print("Labels Shape:", y.shape)

# ---------------------------------
# Train Test Split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ---------------------------------
# Train SVM Model
# ---------------------------------
model = SVC(
    kernel='linear',
    probability=True
)

model.fit(X_train, y_train)

print("Model training completed")

# ---------------------------------
# Prediction
# ---------------------------------
y_pred = model.predict(X_test)

# ---------------------------------
# Accuracy
# ---------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy: {:.2f}%".format(accuracy * 100))

# ---------------------------------
# Detailed Evaluation
# ---------------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------------
# Save Model
# ---------------------------------
joblib.dump(model, "models/face_recognition_model.pkl")

print("Model saved successfully in models/face_recognition_model.pkl")