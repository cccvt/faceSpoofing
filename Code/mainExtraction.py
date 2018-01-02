import argparse
from Code.myPackage import tools as tl
from Code.myPackage import extractFaces as extFace


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-vRtr", "--videoRealtr", required = True,
                    help = "-vRtr Path with real faces in video files for training")
    ap.add_argument("-vAftr", "--videoAttackFtr", required=True,
                    help="-vAftr Path with attacks in video files for training (fixed)")
    ap.add_argument("-vAhtr", "--videoAttackHtr", required=True,
                    help="-vAhtr Path with attacks in video files for training (hand)")
    ap.add_argument("-oRtr", "--outputRealtr", required = True,
                    help = "-oRtr Output path to store the real faces for training")
    ap.add_argument("-oAftr", "--outputAttackFtr", required=True,
                    help="-oAftr Output path to store the attacks for training (fixed)")
    ap.add_argument("-oAhtr", "--outputAttackHtr", required=True,
                    help="-oAhtr Output path to store the attacks for training (hand)")
    ap.add_argument("-vRte", "--videoRealte", required=True,
                    help="-vRte Path with real faces in video files for test")
    ap.add_argument("-vAfte", "--videoAttackFte", required=True,
                    help="-vAfte Path with attacks in video files for test (fixed)")
    ap.add_argument("-vAhte", "--videoAttackHte", required=True,
                    help="-vAhte Path with attacks in video files for test (hand)")
    ap.add_argument("-oRte", "--outputRealte", required=True,
                    help="-oRte Output path to store the real faces for test")
    ap.add_argument("-oAfte", "--outputAttackFte", required=True,
                    help="-oAfte Output path to store the attacks for test (fixed)")
    ap.add_argument("-oAhte", "--outputAttackHte", required=True,
                    help="-oAhte Output path to store the attacks for test (hand)")
    ap.add_argument("-f", "--face", required = True,
                    help = "-f Face Cascade Path")
    ap.add_argument("-pad", "--padding", required=True,
                    help="-pad Padding to add at the image cropped (more information about the context)")
    args = vars(ap.parse_args())


    real_ext = '.mov'
    attack_ext = '.avi'
    pad = int(args["padding"])

    # Create output paths
    tl.makeAllDirs(args)

    # Get all video names from given path
    training_real, training_attack_f, training_attack_h, test_real, test_attack_f, test_attack_h = \
        tl.getVideoNames(args, real_ext, attack_ext)

    # Extract all faces from videos using Haralike Cascade Face detector
    extFace.processPaths(training_real, training_attack_f, training_attack_h, test_real, test_attack_f, test_attack_h, args, pad)

    # Count all images in each path
    tl.countFiles(args)
