import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(lane_image):
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = np.copy(gray)
    x = 0
    for x in range(1):
        blur = cv2.GaussianBlur(blur, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


image = cv2.imread('test_image2.jpg')
lane_image = np.copy(image)
canny_image = canny(lane_image)

plt.imshow(canny_image)
plt.show()