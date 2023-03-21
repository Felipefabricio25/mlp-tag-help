#!/usr/bin/env python
 
import cv2
import numpy as np
import os
import glob
import mlp_jsons

import numpy as np
from datetime import datetime
import math
import cv2.aruco as aruco

import json
import time
cap = cv2.VideoCapture(0)

now = datetime.now()

final_array = ([[0,0,0]])
 
# Defining the dimensions of checkerboard
CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 
 
 
# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
 
# Extracting path of individual image stored in a given directory
images = glob.glob('/home/admsistemas/Pictures/Kamera/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
     
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
         
        imgpoints.append(corners2)
 
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
     
    cv2.imshow('img',img)
    cv2.waitKey(0)
 
cv2.destroyAllWindows()
 
h,w = img.shape[:2]
 
"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
 
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
arucoParameters = aruco.DetectorParameters_create()
corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParameters)
frame = aruco.drawDetectedMarkers(frame, corners)

while(True):
    potatoes = []

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    arucoParameters = aruco.DetectorParameters_create()
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParameters)
    frame = aruco.drawDetectedMarkers(frame, corners)



    cv2.line(frame, (270,190), (270,290), (0, 255, 255), 3)
    cv2.line(frame, (270,290), (370,290), (0, 255, 255), 3)
    cv2.line(frame, (370,290), (370,190), (0, 255, 255), 3)
    cv2.line(frame, (370,190), (270,190), (0, 255, 255), 3)

    if len(corners) > 0:
		# flatten the ArUco IDs list
        ids = ids.flatten()
		# loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned
			# in top-left, top-right, bottom-right, and bottom-left
			# order)

            corners = markerCorner.reshape((4, 2))
            
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # compute and draw the center (x, y)-coordinates of the ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            dX = (topLeft[0] - bottomLeft[0])
            mX = round((102 - (0.846 * dX) + (0.00109 * dX * dX))/100,2)
            if cX > 270 and cX < 370 and cY > 190 and cY < 290:
                cv2.circle(frame, (cX, cY), 4, (0, 255, 0), -1)
                print(markerID)


                # draw the bounding box of the ArUCo detection
                cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

            else:
                cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)            
            # draw the ArUco marker ID on the frame
            #cv2.putText(frame, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,	0.5, (0, 255, 0), 2)
            cv2.putText(frame, str(mX) + " m", (bottomRight[0], bottomRight[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,	0.5, (0, 255, 0), 2)

        markerSizeInCM = 15.9
        corners = np.array(corners).reshape((1, 4, 2))
        
        rvec , tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerSizeInCM, mtx, dist)

        
        for allitems in tvec:
            for things in allitems:
                print("---")
                print(markerID)
                # potatoes = things.tolist()

                # potatoes.append(int(markerID))      
                
                potatoes.append(topLeft[0])
                potatoes.append(topLeft[1])
                potatoes.append(mX)

                print(potatoes)

                mlp_jsons.determine_value(potatoes)  
        




    cv2.imshow('Display', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):

        with open("Dataset.json", 'w') as file:
            print("Final!")
            file.write(jsn)
            
        break



cap.release()
cv2.destroyAllWindows()