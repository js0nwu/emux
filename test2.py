import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 4

img1 = cv2.imread('1.jpg', 0)  # queryImage
img2 = cv2.imread('2.jpg', 0)  # trainImage
# Initiate SIFT detector
# sift = cv2.SIFT()
# sift = cv2.xfeatures2d.SIFT_create()
# sift = cv2.ORB()
sift = cv2.ORB_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)[:4]
src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
# make 4th parameter of findHomography smaller to make it harder to match but narrow down
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,1.0)
matchesMask = mask.ravel().tolist()

h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)
print(pts)
print(dst)
img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   flags=0)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
times = [d[0][0] for d in dst]
times.sort()
times = np.clip(times, 0, img2.shape[1])
print(sorted(times))
timestart = times[0]
timelength = times[-1] - times[0]
correspond = (timestart, timelength)
print(correspond)
plt.imshow(img3), plt.show()
