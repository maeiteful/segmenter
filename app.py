import cv2
import numpy as np
import matplotlib.pyplot as plt

Threshhold = 110 #Edit this number to increase/decrease sensitivity
MinSize = 40 #if you get alot of unuseful small rectangles you can increase this number to remove them

# Read the scanned page image
img = cv2.imread('4.jpeg', cv2.IMREAD_GRAYSCALE)

# Threshold the image to binarize it
ret, thresh = cv2.threshold(img, Threshhold, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through the contours and draw bounding boxes around them
for cnt in contours:
    if cv2.contourArea(cnt) > MinSize:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the image with bounding boxes
plt.imshow(img, cmap='gray')
plt.show()
