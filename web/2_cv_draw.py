import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the image
image = cv2.imread('static/3.jpg')
if image is None:
    print("Error loading image")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Adaptive thresholding instead of fixed
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Load the pretrained model
try:
    model = load_model('model/mnist_model_v3.h5')
except Exception as e:
    print("Failed to load model:", e)
    exit()

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Ensure there are contours to analyze
if not contours:
    print("No contours found")
    exit()

# Calculate the average area of contours
areas = [cv2.contourArea(c) for c in contours]
avg_area = np.mean(areas) if areas else 0

# Loop over the contours
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = area / float(hull_area) if hull_area > 0 else 0

    if area > avg_area * 0.1 and 0.4 < aspect_ratio < 1.6 and solidity > 0.4:
        digit_img = thresh[y:y+h, x:x+w]
        resized_digit = cv2.resize(digit_img, (28, 28))
        norm_digit = resized_digit / 255.0
        norm_digit = norm_digit.reshape(1, 28, 28, 1)

        prediction = model.predict(norm_digit)
        digit = np.argmax(prediction)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Display the image with detected digits
cv2.imshow('Digits Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()