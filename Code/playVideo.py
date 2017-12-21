import cv2

cap = cv2.VideoCapture('../../../IDIAP_DATASET/replayattack/train/attack/fixed/attack_highdef_client001_session01_highdef_photo_adverse.mov')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()