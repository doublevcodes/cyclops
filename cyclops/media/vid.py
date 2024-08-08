import numpy as np
import cv2 as cv
import face_recognition

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
 
    if not ret:
        break
    
    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)

    for (top, right, bottom, left) in face_locations:
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Display the resulting image
    cv.imshow('Video', frame)
    
    if cv.waitKey(1) == ord('q'):
        break
 
cap.release()
cv.destroyAllWindows()