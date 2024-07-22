import numpy as np
import cv2
import random

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
    cv2.imshow('Circle coordinate UP - DOWN - LEFT - RIGHT image', img_with_coordinates)

    cropped_img = thresh_original[coordinateUp[1]-10:coordinateDown[1]+10, coordinateLeft[0]-10:coordinateRight[0]+10]
    return cropped_img

def preprocess_image(image):    
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges
                               

def dilateImage(image, size):
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    kernel = np.ones((15, 15), np.uint8)
    img_dilation = cv2.dilate(image, kernel, iterations=1)
    # Convert all non-zero pixels to white
    _, img_binary = cv2.threshold(img_dilation, 0, 255, cv2.THRESH_BINARY)

    return img_binary    


def compute_hog(image, size):    
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    # Initialize HOG Descriptor
    hog = cv2.HOGDescriptor(_winSize=(size, size),
                            _blockSize=(32, 32),
                            _blockStride=(16, 16),
                            _cellSize=(8, 8),
                            _nbins=18)
    
    # Compute HOG
    h = hog.compute(image)
    return h.flatten()

def euclidean_distance(a, b):
    diff = a - b
    return np.sqrt(np.sum(diff**2))


def templateMatching(image_original, image_draw, size):
    result = cv2.matchTemplate(image_original, image_draw, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    similarity_percentage = max_val * 100
    return similarity_percentage

def absdiff(image_original, image_draw):
    diff = cv2.absdiff(image_original, image_draw)
    non_zero_count = np.count_nonzero(diff)
    total_pixels = diff.size

    similarity_percentage = ((total_pixels - non_zero_count) / total_pixels) * 100
    return similarity_percentage

def contour_compare(img1, img2):
    # Find contours in both images
    contours1, _ = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours2, _ = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # Assuming each image contains only one main contour, the largest one
    # This part can be improved by adding more sophisticated logic to select relevant contours
    contour_original = max(contours1, key=cv2.contourArea)
    contour_draw = max(contours2, key=cv2.contourArea)
    
    contour_image_original = np.zeros_like(img1)
    contour_image_draw = np.zeros_like(img2)

    contour_image_original = cv2.cvtColor(contour_image_original, cv2.COLOR_GRAY2BGR)
    contour_image_draw = cv2.cvtColor(contour_image_draw, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image_original, contour_original, -1, (0, 255, 0), 0)
    cv2.drawContours(contour_image_draw, contour_draw, -1, (0, 255, 0), 0)

    cv2.imshow('Contour Original', contour_image_original)
    cv2.imshow('Contour Draw', contour_image_draw)

    # Compare contours
    match_score = cv2.matchShapes(contour_original, contour_draw, 1, 0.0)
    
    # Convert match score to a similarity percentage
    # You might need to adjust the scale factor to better fit the match_score distribution
    similarity_percentage = max(0, (1 - match_score) * 100)  # Scale factor can be adjusted
    
    return similarity_percentage

def contour_compare_v2(image_original, image_draw):
    # Find contours in both images
    contours1, hierarchy = cv2.findContours(image_original, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours2, hierarchy = cv2.findContours(image_draw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    drawing_original = np.zeros((image_original.shape[0], image_original.shape[1], 3), dtype=np.uint8)
    drawing_draw = np.zeros((image_draw.shape[0], image_draw.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours1)):        
        cv2.drawContours(drawing_original, contours1, i, (0,255,0), 2, cv2.LINE_8, hierarchy, 0)
        cv2.drawContours(drawing_draw, contours2, i, (0,255,0), 2, cv2.LINE_8, hierarchy, 0)

    cv2.imshow('Contours Original', drawing_original)
    cv2.imshow('Contours Draw', drawing_draw)


def contour_compare_hu_moments(image_original, image_draw):
    """
    Compare two images based on the Hu Moments of their contours to compute similarity as a percentage.
    
    Args:
    image_original (np.array): Grayscale image where original contours are to be found.
    image_draw (np.array): Grayscale image where drawn contours are to be found.
    
    Returns:
    float: Similarity percentage, where 100% is a perfect match.
    """
    # Find contours in both images
    contours1, hierarchy1 = cv2.findContours(image_original, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv2.findContours(image_draw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour by area which likely represents the main object
    contour1 = max(contours1, key=cv2.contourArea) if contours1 else None
    contour2 = max(contours2, key=cv2.contourArea) if contours2 else None
    
    drawing_original = np.zeros((image_original.shape[0], image_original.shape[1], 3), dtype=np.uint8)
    drawing_draw = np.zeros((image_draw.shape[0], image_draw.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours1)):        
        cv2.drawContours(drawing_original, contours1, i, (0,255,0), 2, cv2.LINE_8, hierarchy1, 0)
        cv2.drawContours(drawing_draw, contours2, i, (0,255,0), 2, cv2.LINE_8, hierarchy2, 0)

    cv2.imshow('Contours Original', drawing_original)
    cv2.imshow('Contours Draw', drawing_draw)

    # Calculate Moments
    moments1 = cv2.moments(contour1)
    moments2 = cv2.moments(contour2)
    
    # Calculate Hu Moments
    huMoments1 = cv2.HuMoments(moments1).flatten()
    huMoments2 = cv2.HuMoments(moments2).flatten()
    
    # Log scale transform Hu Moments to enhance scale invariance
    huMoments1_log = -np.sign(huMoments1) * np.log10(np.abs(huMoments1))
    huMoments2_log = -np.sign(huMoments2) * np.log10(np.abs(huMoments2))
    
    # Calculate the similarity score based on the scale-adjusted Hu Moments
    distance = np.sqrt(np.sum((huMoments1_log - huMoments2_log) ** 2))
    similarity_score = np.exp(-distance)  # Convert to a similarity score
    
    # Convert to percentage
    similarity_percentage = similarity_score * 100
    return similarity_percentage

def draw_lines(image, lines, color):
    """Draw lines on an image given a set of lines."""
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), color, 2)
    return image

def match_line_segments(image1, image2):
    # Parameters for Canny and HoughLinesP
    canny_threshold1, canny_threshold2 = 50, 150
    hough_thresh = 50
    min_line_length = 50
    max_line_gap = 10

    # Image preprocessing
    edges1 = cv2.Canny(image1, canny_threshold1, canny_threshold2)
    edges2 = cv2.Canny(image2, canny_threshold1, canny_threshold2)

    # Detect lines using Probabilistic Hough Transform
    lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, threshold=hough_thresh, minLineLength=min_line_length, maxLineGap=max_line_gap)
    lines2 = cv2.HoughLinesP(edges2, 1, np.pi/180, threshold=hough_thresh, minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Draw lines on copies of the original images for visualization
    image1_with_lines = draw_lines(image1.copy(), lines1, (255, 0, 0))  # Red lines on image 1
    image2_with_lines = draw_lines(image2.copy(), lines2, (0, 255, 0))  # Green lines on image 2

    matches = []
    # Check if lines1 and lines2 are not None before proceeding
    if lines1 is not None and lines2 is not None:
        for line1 in lines1:
            for line2 in lines2:
                # Calculate length and orientation of each line
                len1 = np.linalg.norm(line1[0][:2] - line1[0][2:])
                len2 = np.linalg.norm(line2[0][:2] - line2[0][2:])
                angle1 = np.arctan2(line1[0][3] - line1[0][1], line1[0][2] - line1[0][0])
                angle2 = np.arctan2(line2[0][3] - line2[0][1], line2[0][2] - line2[0][0])

                # Check if lines are similar based on length and angle
                if abs(len1 - len2) < 10 and abs(angle1 - angle2) < 0.1:
                    matches.append((line1, line2))
                    # Visualize matches
                    cv2.line(image1_with_lines, (line1[0][0], line1[0][1]), (line2[0][0], line2[0][1]), (0, 255, 0), 4)

    # Display images with lines and matches
    # cv2.imshow('Image 1 with Lines and Matches', image1_with_lines)
    # cv2.imshow('Image 2 with Lines', image2_with_lines)
    
    # Calculate match score
    total_lines = (len(lines1) if lines1 is not None else 0) + (len(lines2) if lines2 is not None else 0)
    
    match_score = (2 * len(matches)) / total_lines  # Factor of 2 to account for matches between both images    
    return match_score * 100  # Return as percentage


def main():    
    # Image Size    
    size = 256

    # Load a color image in grayscale
    img_original = cv2.imread('static/draw_form_4.jpg', 0)
    # img_original = cv2.imread('static/executive_original.jpg', 0)

    # img_draw = cv2.imread('static/square.jpg', 0) # RESULT: 40%           (X)
    # img_draw = cv2.imread('static/cube.jpg', 0) # RESULT: 100%            (O)
    # img_draw = cv2.imread('static/cube_draw2.jpg', 0) # RESULT: 60%       (-)
    # img_draw = cv2.imread('static/cube_draw4.jpg', 0) # RESULT: 64%       (-)
    # img_draw = cv2.imread('static/cube_draw5.jpg', 0) # RESULT: 74%       (O)
    # img_draw = cv2.imread('static/pyramid.jpg', 0) # RESULT: 0%           (X)
    # img_draw = cv2.imread('static/circle_cube.jpg', 0) # RESULT: 15%      (X)
    # img_draw = cv2.imread('static/none.jpg', 0) # RESULT: 13%             (X)
    # img_draw = cv2.imread('static/transparent_cube.jpg', 0) # RESULT: 42% (O)
    
    # img_draw = cv2.imread('static/image.jpg', 0) # RESULT: 33%    (O)
    img_draw = cv2.imread('static/exec_form_4_draw.jpg', 0) # RESULT: 33%    (O)

    
    cropped_original = crop_image(img_original)
    cropped_draw = crop_image(img_draw)
    cv2.imshow('Cropped Original', cropped_original)
    cv2.imshow('Cropped Draw', cropped_draw)

    processed_original = preprocess_image(cropped_original)
    processed_draw = preprocess_image(cropped_draw)
    cv2.imshow('Processed Original', processed_original)
    cv2.imshow('Processed Draw', processed_draw)
    
    dilation_original = dilateImage(processed_original, size)
    dilation_draw = dilateImage(processed_draw, size)
    cv2.imshow('Dilated Original', dilation_original)
    cv2.imshow('Dilated Draw', dilation_draw)

    # hog_original = compute_hog(dilation_original, size)
    # hog_draw = compute_hog(dilation_draw, size)
    # similarity = euclidean_distance(hog_original, hog_draw)
    # print(f'Euclidean distance: {similarity}')

    # similarity_percentage_tp = templateMatching(dilation_original, dilation_draw, size)
    # print(f'Similarity Template Matching percentage: {similarity_percentage_tp} %')
    
    # match_shapes_original = contour_compare_hu_moments(dilation_original, dilation_draw)
    # print(f'Similarity Match Shapes percentage: {match_shapes_original} %')

    # similarity_absdiff = absdiff(dilation_original, dilation_draw)
    # print(f'Similarity Absdiff percentage: {similarity_absdiff} %')

    similarity_match_line_segments = match_line_segments(dilation_original, dilation_draw)
    print(f'Similarity Match Line Segments percentage: {similarity_match_line_segments} %')
    
    if similarity_match_line_segments > 70:
        print('FOR SURE IS A CUBE')
    elif similarity_match_line_segments > 50:
        print('MAYBE IS A CUBE')
    else:
        print('IS NOT A CUBE')
        


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()