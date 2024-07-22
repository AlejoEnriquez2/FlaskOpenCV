import cv2
from keras.models import load_model, model_from_json
import numpy as np
import json

def crop_image(image):
    ret_original, thresh_original = cv2.threshold(image, 127, 255, 0)

    ### CUT UP
    flag = False
    # shape[1] is the x-axis - ROWS
    for i in range(thresh_original.shape[1]):
        # shape[0] is the y-axis - COLUMNS
        for j in range(thresh_original.shape[0]):            
            if thresh_original[i][j] == 0 and flag == False:
                flag = True
                coordinateUp = (j, i)

    ### CUT DOWN
    flag = False
    # shape[1] is the x-axis - ROWS
    for i in reversed(range(thresh_original.shape[1])):
        # shape[0] is the y-axis - COLUMNS
        for j in reversed(range(thresh_original.shape[0])):
            if thresh_original[i][j] == 0 and flag == False:
                flag = True
                coordinateDown = (j, i)


    ### CUT LEFT
    flag = False
    # shape[0] is the y-axis - COLUMNS
    for i in range(thresh_original.shape[1]):            
        # shape[1] is the x-axis - ROWS
        for j in range(thresh_original.shape[0]):
            if thresh_original[j][i] == 0 and flag == False:
                flag = True
                coordinateLeft = (i, j)

    ### CUT RIGHT
    flag = False
    # shape[0] is the y-axis - COLUMNS
    for i in reversed(range(thresh_original.shape[1])):
        # shape[1] is the x-axis - ROWS
        for j in reversed(range(thresh_original.shape[0])):
            if thresh_original[j][i] == 0 and flag == False:
                flag = True
                coordinateRight = (i, j)

    print('The coordinate of the black pixel UP is:', coordinateUp)
    print('The coordinate of the black pixel LEFT is:', coordinateLeft)
    print('The coordinate of the black pixel DOWN is:', coordinateDown)
    print('The coordinate of the black pixel RIGHT is:', coordinateRight)

    # COPY of the image to show the CROP coordinates
    img_with_coordinates = thresh_original.copy()

    # Show where to CROP
    cv2.circle(img_with_coordinates, coordinateUp, radius=10, color=(0, 255, 0), thickness=-1)
    cv2.circle(img_with_coordinates, coordinateLeft, radius=10, color=(0, 255, 0), thickness=-1)
    cv2.circle(img_with_coordinates, coordinateDown, radius=10, color=(0, 255, 0), thickness=-1)
    cv2.circle(img_with_coordinates, coordinateRight, radius=10, color=(0, 255, 0), thickness=-1)
    cv2.imshow('Where to crop', img_with_coordinates)

    cropped_img = thresh_original[coordinateUp[1]-5:coordinateDown[1]+5, coordinateLeft[0]-5:coordinateRight[0]+5]
    return cropped_img

def preprocess_image(image):
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    blurred = cv2.GaussianBlur(thresh, (15, 15), 0)
    edges = cv2.Canny(blurred, 50, 150)
    img_dilation = cv2.dilate(edges, np.ones((8, 8), np.uint8), iterations=1)
    return img_dilation


def getRoi(image, model):
    # Ensure the image is in grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get the binary image
    # _, thresh_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    
    # Process each contour
    for contour in contours:
        if cv2.contourArea(contour) > 28:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
            resized = cv2.resize(roi, (28, 28))
            normalized = resized.astype('float32') / 255.0
            reshaped = normalized.reshape(1, 28, 28, 1)
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            results.append(label)
            
            # Draw rectangle and label on the main image
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Display the main image with drawings
    cv2.imshow("Detected Numbers", image) 
    return results

def main():
    model = load_model('model/mnist_model_V3.h5')    
    
    img = cv2.imread('static/1s.jpg',0)
    if img is None:
        print("Image not loaded properly")

    # img = crop_image(img)
    # cv2.imshow('Original Cropped', img)

    processed_img = preprocess_image(img)
    cv2.imshow('Processed', processed_img)

    digits = getRoi(processed_img, model)
    print("Detected digits:", digits)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()