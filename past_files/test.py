import cv2

cv2.namedWindow("Video Feed")
cap = cv2.VideoCapture(0)
ret,frame = cap.read()

if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False

while ret:

    ret, frame = cap.read()
    print(type(frame))
    cv2.imshow("Video Feed",frame)
    if cv2.waitKey(1)==27:
        break
cv2.destroyWindow("Video Feed")
cap.release()
