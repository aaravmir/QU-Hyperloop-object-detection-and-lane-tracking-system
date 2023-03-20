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
    y2 = int(y1 * (1/2))
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
    #Change this if the picture is too noisy / there are too many edges
    for x in range(2):
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
    [(0, 325), (130, height), (370, 192), (500, height), (640, 325), (370, 145)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#This following code for stock video
#Change the text inside quotation to name of video
cap = cv2.VideoCapture("test4.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    edge_image = canny(frame)
    masked_image = region_of_interest(edge_image)
    lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 150, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    full_image = cv2.addWeighted(frame, 0.8, line_image, 0.8, 1)
    cv2.imshow("result ", full_image)
    #The number is to match the framerate of output video with frame rate of input video, but i could never figure it out lol
    if cv2.waitKey(17) == ord('t'):
        break
cap.release()
cv2.destroyAllWindows()

#The following code is for real-time webcam footage
""" cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    edge_image = canny(frame)
    masked_image = region_of_interest(edge_image)
    lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 150, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    full_image = cv2.addWeighted(frame, 0.8, line_image, 0.8, 1)
    cv2.imshow("result ", full_image)
    #The number is to match the framerate of output video with frame rate of input video, but i could never figure it out lol
    if cv2.waitKey(1) == ord('t'):
        break
cap.release()
cv2.destroyAllWindows() """

#The following code is for an image 
#Change the text inside quotation to name of image
""" image = cv2.imread('test_image2.jpg')
lane_image = np.copy(image)
edge_image = canny(lane_image)
masked_image = region_of_interest(edge_image)
lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow("result ", combo_image)
cv2.waitKey(0) """
