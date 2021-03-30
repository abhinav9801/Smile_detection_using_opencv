import cv2
from random import randrange


# Face and smile classifier
face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector=cv2.CascadeClassifier('haarcascade_smile.xml')

# Grab webcam feed
webcam=cv2.VideoCapture(0)


# Show the current frame
while True:

    # Read the current frame from webcam
    sucessful_read,frame=webcam.read()
    
    # If any error is there abort
    if not sucessful_read:
        break

    # Change face to gray scale
    grayscale_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces=face_detector.detectMultiScale(grayscale_frame)

    # Detect smiles
    smiles=smile_detector.detectMultiScale(grayscale_frame,scaleFactor=1.7,minNeighbors=20)

    # Run face detection within each of the faces
    for (x,y,w,h) in faces:

        # Draw Rectangle around faces
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(255),randrange(255),randrange(255)),3)
        
        # Get the subframe (using N- Dimensional Sub array Slicing)
        the_face=frame[y:y+h,x:x+w]
        

        # change to gray scale
        grayscale_face=cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)

        smiles=smile_detector.detectMultiScale(grayscale_face,scaleFactor=1.7,minNeighbors=20)

        # Find all smiles in the faces
       # for (x_,y_,w_,h_)in smiles:

            # Draw Rectangle around the smile
             #cv2.rectangle(the_face,(x_,y_),(x_+w_,y_+h_),(50,50,200),5)

        
        # Label the face smiling
        if len(smiles)>0:
            cv2.putText(frame,'Smiling',(x,y+h+40),fontScale=3,
            fontFace=cv2.FONT_HERSHEY_PLAIN,color=(randrange(255),randrange(255),randrange(255)))


    # show the current frame
    cv2.imshow('Smile Detector',frame)
    
    key= cv2.waitKey(1)
    if key==81 or key==113:
        break

# clearup
webcam.release()
cv2.destroyAllWindows()

print("!!!!CODE COMPLETED SUCCESSFULLY!!!")