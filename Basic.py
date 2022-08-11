import cv2
import numpy as np
import face_recognition

imgFrancis = face_recognition.load_image_file('imagesBasic/francis test.jpg')
imgFrancis = cv2.cvtColor(imgFrancis,cv2.COLOR_BGR2RGB)

imgtest = face_recognition.load_image_file('imagesBasic/francis.jpg')
imgtest= cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgFrancis)[0]
encodeFrancis = face_recognition.face_encodings(imgFrancis)[0]
cv2.rectangle(imgFrancis,(faceloc[3],faceloc[0]),(faceloc[1], faceloc[2]),(255,0,255),2)

facelocTest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(facelocTest[3],facelocTest[0]),(facelocTest[1], facelocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeFrancis],encodetest)
faceDis = face_recognition.face_distance([encodeFrancis],encodetest)
print(results,faceDis)
cv2.putText(imgtest,f'{results}{round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Francis Ojo', imgFrancis)
cv2.imshow('Francis test', imgtest)
cv2.waitKey(0) 