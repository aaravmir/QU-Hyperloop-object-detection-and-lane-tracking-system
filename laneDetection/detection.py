import cv2
import numpy as np
#import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.001, 0
    y1 = image.shape[0]
    #Change this depending on how far you want the lane to go, further -> lower
    y2 = int(y1 * (12/20))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2),(y1, y2), 1)
        slope = parameters[0]
        intercept = parameters [1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def canny(lane_image):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = np.copy(gray)
    x = 0
    #Change this if hte picture is too noise / there are too many edges
    for x in range(3):
        blur = cv2.GaussianBlur(blur, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines: 
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape [0]
    polygons = np.array([
    #Triangle. (left point), (Right point), (Top point). Height stands for bottom y value, Y axis is inversed
    [(150, height), (655, 275), (300, 165)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 0.8, 1)
    cv2.imshow("result ", combo_image)
    if cv2.waitKey(1) == ord('t'):
        break
cap.release()
cv2.destroyAllWindows()
    


# #####################################################################################################
# #This is code for image lane detection which is why it is commented out
# #Read an image
# image = cv2.imread('test_image2.jpg')
# #copy the image
# lane_image = np.copy(image)
# #call canny function
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# #this detects lines of cropped_image using hough transformation
# #(input image, 
# #(the next two are the resolution of the hough accumulator array where the boxes are made to identify points of intersection) distance (rho) resolution,
# #angle resolution in radians (the larger the number the lower the precision, but very low could cause innacuracies and increases runtime)
# #threshold (min number of intersection in one box needed to detect a line)
# #placeholder array (just make it an emtpy array)
# #length of line in pixels to accept into the output (any line length lower than this is rejected)
# #max line gap b/w segmented lines that can be connected automatically)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)

# #This performs a weighted sum of two images the final argument is gamma argument which adds a value to the sum. Here put one because it dont matter
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)


# #open the image in a window, waitKey is so that it keeps the image open until the user presses a key
# cv2.imshow("result ", combo_image)
# cv2.waitKey(0)
# #plt.imshow(canny_image)
# #plt.show()
# #####################################################################################################