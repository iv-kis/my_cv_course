import numpy as np
import cv2
import imutils

image = cv2.imread("Files/im1.jpg")
(h,w,d) = image.shape
print("width={}, height={}, depth={}".format(w,h,d))

(B,G,R) = image[100,100]
print(R,G,B)

# resize the image to 200x200px, ignoring aspect ratio
resized = cv2.resize(image, (200, 200))
#cv2.imshow("Fixed Resizing", resized)

# fixed resizing and distort aspect ratio so let's resize the width
# to be 300px but compute the new height based on the aspect ratio
r = 300.0 / w
dim = (300, int(h * r))
resized = cv2.resize(image, dim)
#cv2.imshow("Aspect Ratio Resize", resized)

cv2.imshow("image",image)

# manually computing the aspect ratio can be a pain so let's use the
# imutils library instead
resized = imutils.resize(image, width=300)
#cv2.imshow("Imutils Resize", resized)

# extract a 100x100 pixel square ROI (Region of Interest) from the
# input image starting at x=320,y=60 at ending at x=420,y=160
#roi = image[60:160, 320:420]
#cv2.imshow("ROI", roi)
#cv2.waitKey(0)

# let's rotate an image 45 degrees clockwise using OpenCV by first
# computing the image center, then constructing the rotation matrix,
# and then finally applying the affine warp
center = (w // 2,h // 2)
Matr = cv2.getRotationMatrix2D(center,-45,1)
rotated = cv2.warpAffine(image,Matr,(w,h))
#cv2.imshow("rotated",rotated)

# apply a Gaussian blur with a 11x11 kernel to the image to smooth it,
# useful when reducing high frequency noise
blurred = cv2.GaussianBlur(image,(11,11),3)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)