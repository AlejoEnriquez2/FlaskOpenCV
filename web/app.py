from cv_comparison import crop_image, dilateImage, match_line_segments, preprocess_image
from flask import Flask, request, jsonify # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
import base64

app = Flask(__name__)

@app.route("/draw_recognizer", methods=["POST"])
def recognize_cube():
    if not request.json or not 'image' in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    img_base64 = request.json['image']
    formId = request.json['formId']
    print("FORM ID ==============> " + str(formId))
    try:
        # Decode the Base64 string
        img_bytes = base64.b64decode(img_base64)
        # Convert bytes to a numpy array
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        # Decode the numpy array into an image
        img = cv2.imdecode(img_array, 0)
        cv2.imwrite('static/image.jpg', img)
        response = areSimilarCube(img, formId)

        # Assuming you do some recognition or processing, let's just return a success message
        return jsonify({"message": response[0], "isCube": response[1]})
    except Exception as e:
        print(e)
        return jsonify({'error': 'Failed to process image'}), 500
    
@app.route("/executive_recognizer", methods=["POST"])
def recognize_executive():
    if not request.json or not 'image' in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    img_base64 = request.json['image']
    formId = request.json['formId']

    try:
        # Decode the Base64 string
        img_bytes = base64.b64decode(img_base64)
        # Convert bytes to a numpy array
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        # Decode the numpy array into an image
        img = cv2.imdecode(img_array, 0)
        cv2.imwrite('static/output_exec.jpg', img)
        response = areSimilarExecutive(img, formId)

        # Assuming you do some recognition or processing, let's just return a success message
        return jsonify({"message": response[0], "isSimilar": response[1]})
    except Exception as e:
        print(e)
        return jsonify({'error': 'Failed to process image'}), 500

def areSimilarCube(img_draw, formId):
    size = 256
    if formId == 1:
        img_original = cv2.imread('static/cube.jpg', 0)
    elif formId == 4:
        img_original = cv2.imread('static/draw_form_4.jpg', 0)

    cropped_original = crop_image(img_original)
    cropped_draw = crop_image(img_draw)
    processed_original = preprocess_image(cropped_original)
    processed_draw = preprocess_image(cropped_draw)
    dilation_original = dilateImage(processed_original, size)
    dilation_draw = dilateImage(processed_draw, size)

    similarity_match_line_segments = match_line_segments(dilation_original, dilation_draw)
    print(f'Similarity Match Line Segments percentage: {similarity_match_line_segments} %')
    
    if similarity_match_line_segments > 70:        
        return ['FOR SURE IS THE SAME:' + ' ' + str(similarity_match_line_segments) + '%', 'yes']
    elif similarity_match_line_segments > 50:
        return ['MAYBE IS THE SAME:' + ' ' + str(similarity_match_line_segments) + '%', 'maybe']
    else:
        return ['IS NOT THE SAME:' + ' ' + str(similarity_match_line_segments) + '%', 'no']


def areSimilarExecutive(img_draw, formId):
    size = 256
    if formId == 1:
        img_original = cv2.imread('static/executive_original.jpg', 0)
    elif formId == 4:
        img_original = cv2.imread('static/exec_form_4.jpg', 0)

    cropped_original = crop_image(img_original)
    cropped_draw = crop_image(img_draw)
    processed_original = preprocess_image(cropped_original)
    processed_draw = preprocess_image(cropped_draw)
    dilation_original = dilateImage(processed_original, size)
    dilation_draw = dilateImage(processed_draw, size)

    similarity_match_line_segments = match_line_segments(dilation_original, dilation_draw)
    print(f'Similarity Match Line Segments percentage: {similarity_match_line_segments} %')
    
    if similarity_match_line_segments > 70:        
        return ['FOR SURE IS THE SAME:' + ' ' + str(similarity_match_line_segments) + '%', 'yes']
    elif similarity_match_line_segments > 50:
        return ['MAYBE IS THE SAME:' + ' ' + str(similarity_match_line_segments) + '%', 'maybe']
    else:
        return ['IS NOT THE SAME:' + ' ' + str(similarity_match_line_segments) + '%', 'no']



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

