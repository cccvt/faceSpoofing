import argparse
from Code.myPackage import tools as tl
from Code.myPackage import extractFaces as extFace


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-vR", "--videoReal", required = True,
                    help = "-vR Path with real faces in video files")
    ap.add_argument("-vAf", "--videoAttackF", required=True,
                    help="-vAf Path with attacks in video files (fixed)")
    ap.add_argument("-vAh", "--videoAttackH", required=True,
                    help="-vAh Path with attacks in video files (hand)")
    ap.add_argument("-oR", "--outputReal", required = True,
                    help = "-oR Output path to store the real faces")
    ap.add_argument("-oAf", "--outputAttackF", required=True,
                    help="-oR Output path to store the attacks (fixed)")
    ap.add_argument("-oAh", "--outputAttackH", required=True,
                    help="-oAh Output path to store the attacks (hand)")
    ap.add_argument("-f", "--face", required = True,
                    help = "-f Face Cascade Path")
    ap.add_argument("-pad", "--padding", required=True,
                    help="-pad Padding to add at the image cropped (more information about the context)")
    args = vars(ap.parse_args())

    num_faces = 0
    pad = int(args["padding"])
    # Create output paths
    print("\nCreating destination paths: \n{}\n{}\n{}\n".format(args['outputReal'], args['outputAttackF'], args['outputAttackH']))
    tl.makeDir(args['outputReal'])
    tl.makeDir(args['outputAttackF'])
    tl.makeDir(args['outputAttackH'])

    # Make face extraction
    print("Extracting faces from paths: \n{}\n{}\n{}\n".format(args['videoReal'], args['videoAttackF'],
                                                              args['videoAttackH']))

    training_real = tl.natSort(tl.getSamples(args["videoReal"]))
    training_attack_f = tl.natSort(tl.getSamples(args["videoAttackF"]))
    training_attack_h = tl.natSort(tl.getSamples(args["videoAttackH"]))
    print("Number of videos: {}, {}, {}".format(len(training_real), len(training_attack_f), len(training_attack_h)))

    n = 1
    for video in training_real:
        print("Video processing {}/{}".format(n, len(training_real)))
        num_faces += extFace.makeDet(video, args['outputReal'], args['face'], pad)
        n += 1
    real_faces = num_faces

    n = 1
    num_faces = 0
    for video in training_attack_f:
        print("Video processing {}/{}".format(n, len(training_attack_f)))
        num_faces += extFace.makeDet(video, args['outputAttackF'], args['face'], pad)
        n += 1
    attack_f_faces = num_faces

    n = 1
    num_faces = 0
    for video in training_attack_h:
        print("Video processing {}/{}".format(n, len(training_attack_h)))
        num_faces += extFace.makeDet(video, args['outputAttackH'], args['face'], pad)
        n += 1
    attack_h_faces = num_faces
    print("Extracted {} real faces\n"
          "Extracted {} attack faces (fixed)\n"
          "Extracted {} attack faces (hand)\n\n".format(real_faces, attack_f_faces, attack_h_faces))