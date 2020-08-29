import numpy as np
import cv2
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,
                help="Path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)[1]
thresh = cv2.GaussianBlur(thresh,(11,11),3)
thresh = cv2.threshold(thresh,100,255,cv2.THRESH_BINARY_INV)[1]
thresh = cv2.threshold(thresh,100,255,cv2.THRESH_BINARY_INV)[1]

cv2.imshow("threshold", thresh)
edged = cv2.Canny(cv2.GaussianBlur(gray,(11,11),0.5), 30,200)
edged = cv2.threshold(edged,100,255,cv2.THRESH_BINARY_INV)[1]
cv2.imshow("edges", edged)

output = cv2.bitwise_and(thresh, edged)
output = cv2.GaussianBlur(output,(11,11),1)
cv2.imshow("Output", output)

# find contours (i.e., outlines) of the foreground objects in the
# thresholded image
cnts = cv2.findContours(output.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
result = image.copy()
# loop over the contours
#
for c in cnts:
	# draw each contour on the output image with a 3px thick purple
	# outline, then display the output contours one at a time
	cv2.drawContours(result, [c], -1, (240, 0, 159), 3)
	cv2.imshow("Contours", result)
	cv2.waitKey(0)

# draw the total number of contours found in purple
text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,(240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)

# we apply erosions to reduce the size of foreground objects
#mask = thresh.copy()
#mask = cv2.erode(mask, None, iterations=10)
#cv2.imshow("Eroded", mask)
#cv2.waitKey(0)

