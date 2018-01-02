import cv2
from os.path import altsep, basename, isfile, join, splitext
from os import errno
import sys



def processPaths(training_real, training_attack_f, training_attack_h, test_real, test_attack_f, test_attack_h, args, pad):
    num_faces = 0
    # Make face extraction
    print("Extracting faces from paths: \n"
          "'{}' ({} videos)\n"
          "'{}' ({} videos)\n"
          "'{}' ({} videos)\n"
          "'{}' ({} videos)\n"
          "'{}' ({} videos)\n"
          "'{}' ({} videos)\n".format(args['videoRealtr'], len(training_real),
                                      args['videoAttackFtr'], len(training_attack_f),
                                      args['videoAttackHtr'], len(training_attack_h),
                                      args['videoRealte'], len(training_real),
                                      args['videoAttackFte'], len(training_attack_f),
                                      args['videoAttackHte'], len(training_attack_h)
                                      ))

    n = 1
    for video in training_real:
        print("Video processing {}/{}".format(n, len(training_real)))
        num_faces += extract(video, args['outputReal'], args['face'], pad)
        n += 1
    real_faces_tr = num_faces

    n = 1
    num_faces = 0
    for video in training_attack_f:
        print("Video processing {}/{}".format(n, len(training_attack_f)))
        num_faces += extract(video, args['outputAttackF'], args['face'], pad)
        n += 1
    attack_f_faces_tr = num_faces

    n = 1
    num_faces = 0
    for video in training_attack_h:
        print("Video processing {}/{}".format(n, len(training_attack_h)))
        num_faces += extract(video, args['outputAttackH'], args['face'], pad)
        n += 1
    attack_h_faces_tr = num_faces

    n = 1
    for video in test_real:
        print("Video processing {}/{}".format(n, len(training_real)))
        num_faces += extract(video, args['outputReal'], args['face'], pad)
        n += 1
    real_faces_te = num_faces

    n = 1
    num_faces = 0
    for video in test_attack_f:
        print("Video processing {}/{}".format(n, len(training_attack_f)))
        num_faces += extract(video, args['outputAttackF'], args['face'], pad)
        n += 1
    attack_f_faces_te = num_faces

    n = 1
    num_faces = 0
    for video in test_attack_h:
        print("Video processing {}/{}".format(n, len(training_attack_h)))
        num_faces += extract(video, args['outputAttackH'], args['face'], pad)
        n += 1
    attack_h_faces_te = num_faces

    print("\n\nExtracted {} real faces for training\n"
          "Extracted {} attack faces for training (fixed)\n"
          "Extracted {} attack faces for training (hand)\n"
          "Extracted {} real faces for test\n"
          "Extracted {} attack faces for test (fixed)\n"
          "Extracted {} attack faces for test (hand)\n\n"
          .format(real_faces_tr, attack_f_faces_tr, attack_h_faces_tr, real_faces_te, attack_f_faces_te, attack_h_faces_te))


def extract(video_in, out_path, face_path, pad):
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

    '''
    cv2.CAP_ANY
    cv2.CAP_VFW
    cv2.CAP_QT
    cv2.CAP_DSHOW
    cv2.CAP_MSMF 
    cv2.CAP_WINRT
    cv2.CAP_FFMPEG
    cv2.CAP_PROP_FOURCC
    '''
    video = cv2.VideoCapture(video_in, cv2.CAP_FFMPEG)
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
