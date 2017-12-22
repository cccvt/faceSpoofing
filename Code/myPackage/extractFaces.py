import cv2
from os.path import altsep, basename, isfile, join, splitext
from os import errno
import sys



def makeDet(video_in, out_path, face_path, pad):
    '''
    This function detects and draws the eyes and faces that appears in the video following the
    default scheme face_cascade and eye_cascade
    :param video_in: full or relative path for the video file
    :param video_out: path where the video resultant will be stored
    :param face_path: path where is the XML schema for faces
    :return: none
    '''

    face_cascade = cv2.CascadeClassifier(face_path)
    img_ext = '.png'
    video_name = splitext(basename(video_in))[0]

    # video = cv2.VideoCapture(video_in, 500)
    video = cv2.VideoCapture(video_in, 1900)
    # video = cv2.VideoCapture(video_in, 6)

    num = 0
    while (video.isOpened()):
        ret, frame = video.read()

        if ret:
            faces = face_cascade.detectMultiScale(frame,
                                                  scaleFactor=1.3,
                                                  minNeighbors=5,
                                                  minSize=(30, 30),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in faces:
                # Add padding if it exists to rectangle and ROI
                roi_color = frame[-int(pad/2)+y : y + h + int(pad/2), -int(pad/2) + x : x + w + int(pad/2)]
                # cv2.rectangle(frame, (x, y), (-int(pad/2) + x + w + int(pad/2), -int(pad/2) + y + h + int(pad/2)), (255, 255, 0), 1)  # BGR
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)  # BGR
                cv2.imwrite(altsep.join((out_path, video_name+'_'+str(num)+'_'+str(int(pad/2))+img_ext)), roi_color)
            # cv2.imshow(video_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        num += 1
    # num_ext = len([name for name in listdir(out_path) if isfile(join(out_path, name))])
    print("Extracted {} images from '{}'".format(num, video_name))

    video.release()
    cv2.destroyAllWindows()

    return num
