import numpy as np
import cv2

from keras.models import load_model

emotion_model_path ='models_mini_XCEPTION.78-0.64.hdf5'
emotion_classifier = load_model(emotion_model_path)

etiquetas=['Anger/Enfado','Disgust/disgust','Fear/Miedo','Happy/Feliz','Sad/Tristeza','Surprise/Sorpresa','Neutral']

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
#        scaleFactor=1.5,
        minNeighbors=10,
#        minSize=(30, 30),
#        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        
        crop_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(crop_img, (48,48), interpolation = cv2.INTER_AREA)/255
        b=emotion_classifier.predict(resized.reshape(1,48,48,1))
        b=np.argmax(b)
        cv2.putText(frame, etiquetas[b], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
#    x=200
#    y=200
#    w=200
#    h=200
#    cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()