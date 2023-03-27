import cv2
import face_recognition
import numpy as np 


imgJ = face_recognition.load_image_file("jethiya.jpg")
imgJ = cv2.cvtColor(imgJ, cv2.COLOR_BGR2RGB)

imgT = face_recognition.load_image_file("bhide.jpg")
imgT = cv2.cvtColor(imgT, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgJ)[0]
encodeJ = face_recognition.face_encodings(imgJ)[0]
print(faceloc)
cv2.rectangle(imgJ,(faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]),(255,0,255),2 )

facelocT = face_recognition.face_locations(imgT)[0]
encodeT = face_recognition.face_encodings(imgT)[0]
print(facelocT)
cv2.rectangle(imgT,(facelocT[3], facelocT[0]), (facelocT[1], facelocT[2]),(255,0,255),2 )

results = face_recognition.compare_faces([encodeJ],encodeT)
facedis= face_recognition.face_distance([encodeJ], encodeT)
print(results, facedis)
cv2.putText(imgT, f'{results} , {round(facedis[0],2)}',(25,25), cv2.FONT_HERSHEY_COMPLEX,0.9,(0,0,255),1)

cv2.imshow("Jethala", imgJ)
cv2.imshow("Test Image", imgT)

cv2.waitKey(0)