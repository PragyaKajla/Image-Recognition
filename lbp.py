import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Constants
IMG_SIZE = (128, 128)
RADIUS = 1  # radius for LBP
N_POINTS = 8 * RADIUS  # number of points in LBP
THRESHOLD = 0.029 # Threshold for determining a match 

def preprocess_image(path):
    """Preprocess an image: convert to grayscale, resize, and apply histogram equalization."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    img = cv2.resize(img, IMG_SIZE)  # Resize image to standard size
    img = cv2.equalizeHist(img)  # Histogram equalization for better contrast
    return img

def extract_lbp_features(image):
    """Extract Local Binary Pattern (LBP) features from an image."""
    lbp = local_binary_pattern(image, N_POINTS, RADIUS, method='uniform')
    
    # Compute the histogram of the LBP image
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS+3), range=(0, N_POINTS+2))
    
    # Normalize the histogram
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= (lbp_hist.sum() + 1e-6) 
    
    return lbp_hist

# Paths
target_face_path = r"C:\Users\nitya\OneDrive\Desktop\FaceRecog\target_face.jpg"
folder_path = r"C:\Users\nitya\OneDrive\Desktop\FaceRecog\output_faces_YOLOv8m_face"

# Preprocess the target image
target_img = preprocess_image(target_face_path)
target_lbp_features = extract_lbp_features(target_img)

# Process images in the folder
X = []
image_names = []

for filename in os.listdir(folder_path):
    path = os.path.join(folder_path, filename)
    try:
        img = preprocess_image(path)
        lbp_features = extract_lbp_features(img)
        X.append(lbp_features)
        image_names.append(filename)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

X = np.array(X)

# Compare the target image with each image in the folder
print("\n📊 Distance Report:")
min_distance = float('inf')
most_similar_face = None

# Ground truth labels for the dataset (1 = match, 0 = non-match)
ground_truth_labels = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]

predicted_labels = []  # Storing predicted results here

for idx, face_features in enumerate(X):
    dist = euclidean(face_features, target_lbp_features)
    is_match = dist < THRESHOLD  
    predicted_labels.append(1 if is_match else 0)
    
    # Print the match result for each image
    print(f"{image_names[idx]} → Distance: {dist:.4f} → Match: {is_match}")
    
    if dist < min_distance:
        min_distance = dist
        most_similar_face = (image_names[idx], dist, is_match)

# Output most similar face
print("\n🏆 Most Similar Face:")
print(f"{most_similar_face[0]} → Distance: {most_similar_face[1]:.4f} → Match: {most_similar_face[2]}")

# calculate the evaluation metrics
accuracy = accuracy_score(ground_truth_labels, predicted_labels)
precision = precision_score(ground_truth_labels, predicted_labels)
recall = recall_score(ground_truth_labels, predicted_labels)
f1 = f1_score(ground_truth_labels, predicted_labels)

# Print the evaluation metrics
print(f"\n📊 Evaluation Metrics:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
