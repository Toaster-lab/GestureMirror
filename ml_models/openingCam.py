import numpy as np
import cv2 as cv
import time 
import sys

cap = cv.VideoCapture(0) #creates an object you can read from -> source
# Checking if everything is working
if cap == None:
    sys.exit("No source defined")
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# FPS timer setup:
prev_time = time.perf_counter() #returns clock value

while True:
    # Capture frame-by-frame
    ret, frame = cap.read() #updating frame
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #arguments(src, code)

    # calculating frame:
    now = time.perf_counter()
    fps = 1 / (now - prev_time)
    prev_time = now

    #showing fps
    cv.putText(gray, f"FPS: {fps:.1f}", (10, 30),
           cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release() #frees camera
cv.destroyAllWindows() #closes OpenCV window
print(frame.shape())