import numpy as np
import cv2
import argparse

def order_points(pts):
    rect = np.zeros((4,2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl,tr,br,bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates

    widthA = np.sqrt(((br[0] - bl[0]) ** 2)+((br[1]-bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA),int(widthB))

    heightA = np.sqrt(((br[0] - tr[0]) ** 2)+((br[1]-tr[1]) ** 2))
    heightB = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    maxHeight = max(int(heightA),int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-c", "--coords",
	help = "comma seperated list of source points")
args = vars(ap.parse_args())

original = cv2.imread(args["image"])
image = original.copy()
#do not use eval() in real code dealing with command line arguments!
pts = np.array(eval(args["coords"]),dtype = "float32")
for i in pts:
    image = cv2.circle(image, tuple(i), 10, (0, 255, 127), -1)

warped = four_point_transform(original, pts)
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)