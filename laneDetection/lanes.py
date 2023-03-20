import cv2
import numpy as np
#import matplotlib.pyplot as plt


#make coordinates from slope and intercept given the image
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (9/20))
    #y = mx + b, x = (y - b)/m
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    #declare empty lists (like arrays)
    #will contain the coordinates of the averaged lines on the left and right lane
    left_fit = []
    right_fit = []

    for line in lines:
        #reshape all lines into a one dimentional array with 4 elements, which will be x1, y1, x2, y2
        x1, y1, x2, y2 = line.reshape(4)
        #this fits a polynomial to the x and y points and return a vector of coefficents which decribe slope and intercept
        #last argument is degree of polynomial which is just 1 becuase we are dealing with linear lines
        parameters = np.polyfit((x1, x2),(y1, y2), 1)
        #get the slope and intercept from parameters
        slope = parameters[0]
        intercept = parameters [1]
        #in this scenario, the y axis is flipped, i.e, 0 is at the top. Hence all right light segments have a negetive slope and
        #all left line segments have a poitive slope
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
    #apply canny edge detection technique: first make it black and white, then blur it, and finally apply canny function
    #Keep in mind that the guassian step was completely option as canny automatically applies it
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    #first mention what image to blur, then size of kernal, and then deviation (idk)
    #blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #input image, lower threshhold and higher threshold of gradient. If a gradient is higher than high threshold then it is accepted, 
    # opposite for lower
    # If its in the middle it is accepted only if it is connected to a strong edge (higher than high)
    # Usually use a ratio of 1:2 or 1:3
    canny = cv2.Canny(gray, 50, 150)
    return canny


#outputs a display image based on source image and lines from hough transformation
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:#if lines are not empty
        for x1, y1, x2, y2 in lines: 
            #draws line segment connecting two points (2nd and 3rd argument )over an image (1st argument),
            #then gives it a colour and thickeness (last two arguments) colour - (BGR)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


#this function makes a region of interest, i.e a black mask over the main image that hides unimportant stuff
def region_of_interest(image):
    #define a height constant that is at the y value of 0 in the image
    height = image.shape [0]
    #the fillPoly function only accpets polygons as arguments, so we make a traingle polygon
    polygons = np.array([
    #these are the coordinates of the trianle taken by printing the image using matlibplot
    [(200, height), (1100, height), (550, 250)]
    ])
    #makes an array of zeros with the same shape (dimentions) as 'image', colour is black
    mask = np.zeros_like(image)
    #superimpose the polygon on the mask, filling the area of the polygon with pixels of 255 intensity (white)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


#This following code for stock video
#Change the text inside quotation to name of video
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result ", combo_image)
    if cv2.waitKey(1) == ord('t'):
        break
cap.release()
cv2.destroyAllWindows()
    
#The following code is for real-time webcam footage
""" # Create a VideoCapture object to get the webcam video
cap = cv2.VideoCapture(0)

# Create a VideoCapture object to get the webcam video
cap = cv2.VideoCapture(0)

# Loop over the frames of the video
while True:
    # Read the current frame from the webcam video
    ret, frame = cap.read()
        
    # Display the current frame
    edge_image = canny(frame)
    masked_image = region_of_interest(edge_image)
    lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 150, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    full_image = cv2.addWeighted(frame, 0.8, line_image, 0.8, 1)
    cv2.imshow("result ", full_image)
    
    # Exit the loop if 't' is pressed
    if cv2.waitKey(1) == ord('t'):
        break
# Release the VideoCapture object and destroy all windows
cap.release()
cv2.destroyAllWindows() """ 

#The following code is for an image 
#Change the text inside quotation to name of image
""" #####################################################################################################
#This is code for image lane detection which is why it is commented out
#Read an image
image = cv2.imread('test_image.jpg')
#copy the image
lane_image = np.copy(image)
#call canny function
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
#this detects lines of cropped_image using hough transformation
#(input image, 
#(the next two are the resolution of the hough accumulator array where the boxes are made to identify points of intersection) distance (rho) resolution,
#angle resolution in radians (the larger the number the lower the precision, but very low could cause innacuracies and increases runtime)
#threshold (min number of intersection in one box needed to detect a line)
#placeholder array (just make it an emtpy array)
#length of line in pixels to accept into the output (any line length lower than this is rejected)
#max line gap b/w segmented lines that can be connected automatically)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)

#This performs a weighted sum of two images the final argument is gamma argument which adds a value to the sum. Here put one because it dont matter
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#open the image in a window, waitKey is so that it keeps the image open until the user presses a key
cv2.imshow("result ", image)
cv2.waitKey(0)
#plt.imshow(combo_image)
#plt.show()
##################################################################################################### """