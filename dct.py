import cv2
import numpy as np
import os
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Constants
IMG_SIZE = (128, 128)
THRESHOLD = 3.5  # Tweak based on tests

def preprocess_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)
    gray = cv2.equalizeHist(gray)
    return gray.flatten()

def extract_dct_features(image):
    # Perform 2D DCT
    dct = cv2.dct(np.float32(image))
    # Take the top-left 10x10 coefficients (you can adjust the size based on your need)
    dct_coeffs = dct[:10, :10]
    return dct_coeffs.flatten()

# Paths
target_face_path = r"C:\Users\nitya\OneDrive\Desktop\FaceRecog\target_face.jpg"
folder_path = r"C:\Users\nitya\OneDrive\Desktop\FaceRecog\output_faces_YOLOv8m_face"

# Preprocess images
target_vector = preprocess_image(target_face_path)
X = []
image_names = []

for filename in os.listdir(folder_path):
    path = os.path.join(folder_path, filename)
    try:
        img = preprocess_image(path)
        dct_features = extract_dct_features(img)
        X.append(dct_features)
        image_names.append(filename)
    except:
        continue

X = np.array(X)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
target_scaled = scaler.transform([extract_dct_features(target_vector)])

# Compare with all images
print("\n📊 Distance Report:")
min_distance = float('inf')
most_similar_face = None

for idx, face_vec in enumerate(X_scaled):
    dist = euclidean(face_vec, target_scaled[0])
    is_match = dist < THRESHOLD
    print(f"{image_names[idx]} → Distance: {dist:.4f} → Match: {is_match}")
    
    if dist < min_distance:
        min_distance = dist
        most_similar_face = (image_names[idx], dist, is_match)

# Output most similar
print("\n🏆 Most Similar Face:")
print(f"{most_similar_face[0]} → Distance: {most_similar_face[1]:.4f} → Match: {most_similar_face[2]}")


ground_truth_labels = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
] 
predicted_labels = []  

# Your DCT matching code
for idx, face_vec in enumerate(X_scaled):
    dist = euclidean(face_vec, target_scaled[0])
    is_match = dist < THRESHOLD  
    predicted_labels.append(1 if is_match else 0)

# Now, calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(ground_truth_labels, predicted_labels)
precision = precision_score(ground_truth_labels, predicted_labels, zero_division=0)
recall = recall_score(ground_truth_labels, predicted_labels, zero_division=0)
f1 = f1_score(ground_truth_labels, predicted_labels, zero_division=0)

# Print the evaluation metrics
print(f"\n📊 Evaluation Metrics:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
