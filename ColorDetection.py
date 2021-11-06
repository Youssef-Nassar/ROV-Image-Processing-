#Youssef Nassar 20201133

import numpy as np 
import cv2 as cv 

lvideo = cv.VideoCapture(0)

while True:
    ret, frame = lvideo.read()
    width = int(lvideo.get(3))
    height = int(lvideo.get(4))

    #Layers
    hsv =cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    # lower_red = np.array([0,50,50])
    # upper_red = np.array([10,255,255])
    #blue range : lower[90, 50, 50] -- upper[130, 255 ,255]
    lower_red = np.array([90, 50, 50])
    upper_red = np.array([130, 255 ,255])

    mask = cv.inRange(hsv, lower_red, upper_red)
    af_mask = cv.bitwise_and(frame, frame, mask=mask)
    
    ####
    final = frame.copy()
    _,thres = cv.threshold(mask, 0, 255, cv.THRESH_OTSU)
    cont,_= cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    final = cv.drawContours(final, cont, -1, (0,255,0), 2)




    #create 4 screens
    collector = np.zeros(frame.shape, np.uint8)
    smal_frame = cv.resize(frame, (0,0), fx=0.5, fy=0.5)
    smal_hsv = cv.resize(hsv, (0,0), fx=0.5, fy=0.5)
    smal_afmsk = cv.resize(af_mask, (0,0), fx=0.5, fy=0.5)
    smal_msk = cv.resize(mask, (0,0), fx=0.5, fy=0.5)
    collector[:height//2, :width//2] = smal_frame
    collector[:height//2, width//2:] = smal_hsv
    collector[height//2:, width//2:] = smal_afmsk

    cv.imshow('LIVE', collector)
    cv.imshow('LIVE2',smal_msk)
    cv.imshow('LIVE3',final)

    #Exit the Live video
    if cv.waitKey(1) == ord('x'):
        break

lvideo.release()
cv.destroyAllWindows()