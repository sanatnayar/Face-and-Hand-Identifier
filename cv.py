import cv2
import numpy as np
import sys
import easygui

from sklearn.metrics import pairwise


background = None


weight = 0.5

roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600


def calc_avg(frame, weight):
    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, weight)



def segment(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    image, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        hand_segment = max(contours, key=cv2.contourArea)

        return (thresholded, hand_segment)



def count_fingers(thresholded, hand_segment):
    
    conv_hull = cv2.convexHull(hand_segment)
    
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    
    max_distance = distance.max()
    
    # Create a circle with 90% radius of the max euclidean distance
    radius = int(0.8 * max_distance)
    circumference = (2 * np.pi * radius)

    # Not grab an ROI of only that circle
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    
    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
    
    # This then returns the cut out obtained using the mask on the thresholded hand image.
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # Grab contours in circle ROI
    image, contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Finger count starts at 0
    count = 0

    # loop through the contours to see if we count any more fingers.
    for cnt in contours:
        
        # Bounding box of countour
        (x, y, w, h) = cv2.boundingRect(cnt)

        # Increment count of fingers based on two conditions:
        
        # 1. Contour region is not the very bottom of hand area (the wrist)
        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
        
        # 2. Number of points along the contour does not exceed 25% of the circumference of the circular ROI (otherwise we're counting points off the hand)
        limit_points = ((circumference * 0.25) > cnt.shape[0])
        
        
        if  out_of_wrist and limit_points:
            count += 1

    return count

myvar=''

while myvar!='f' and myvar!='h':
   myvar = easygui.enterbox("Type 'f' for face detection or 'h' to detect number of fingers held up") 


if myvar=='h':
    cam = cv2.VideoCapture(0)

    # Intialize a frame count
    frames = 0


    while True:
        ret, frame = cam.read()

        frame = cv2.flip(frame, 1)


        frame_copy = frame.copy()


        roi = frame[roi_top:roi_bottom, roi_right:roi_left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

     
        if frames < 60:
            calc_avg(gray, weight)
            if frames <= 59:
                cv2.putText(frame_copy, "PLEASE WAIT...SYNCHRONIZING BACKGROUND", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Finger Count",frame_copy)
                
        else:
            
            hand = segment(gray)

            if hand is not None:
                
          
                thresholded, hand_segment = hand

                cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0),1)

                fingers = count_fingers(thresholded, hand_segment)

                cv2.putText(frame_copy, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                cv2.imshow("Thesholded", thresholded)

      
        cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)

        frames += 1

        cv2.imshow("Finger Count", frame_copy)
 
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()

if myvar=='f':
    
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

        
    

