from os import listdir, makedirs, errno, system
from os.path import isfile, join, altsep, exists
from natsort import natsorted, ns
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import cv2
import itertools


def plot_confusion_matrix(plot_path, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt_name = altsep.join((plot_path,"".join((title,".png"))))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label', labelpad=0)

    plt.savefig(plt_name)
    plt.show()


def prepareData(paths, descriptor, plot_path, ratio= 1.0):
    labels = []
    data = []
    numData = int(countFiles(paths)*ratio)
    first_r = True
    first_a = True
    numLabels = {}

    if ratio != 1.0:
        print("{} examples will be processed for TESTING with ratio = {}\n".format(numData, ratio))
        for path in paths:
            images = getSamples(path)
            # Randomize data for testing and get the first numData files
            images = np.random.permutation(images)[:numData]

            for image in images:
                img = cv2.imread(image)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hist, bins, lbp_img = descriptor.describe(gray)

                # extract the label from the image path, then update the
                # label and data lists
                lbl = image.split("/")[-2]
                labels.append(lbl)
                data.append(hist)

                if first_r and lbl == "real" or first_a and lbl == "attack":
                    plt_name = altsep.join((plot_path, "".join((lbl, "_example_{}_{}.png".format(descriptor.numPoints, descriptor.radius)))))
                    fig = plt.figure()
                    plt.subplot(131), plt.imshow(gray, cmap='gray')
                    plt.axis('off'), plt.title("Image original (gray)")

                    plt.subplot(132), plt.imshow(lbp_img, cmap='gray')
                    plt.axis('off'), plt.title("LBP")

                    plt.subplot(133), plt.hist(lbp_img.ravel(), int(lbp_img.max()+2), normed= 1, color='red')
                    # plt.title("LBP histogram")

                    fig.suptitle(
                        "{} image using LBP ({},{})".format(lbl, descriptor.numPoints, descriptor.radius),
                        fontsize=14)

                    plt.tight_layout()
                    plt.savefig(plt_name)
                    plt.show()

                    if first_r and lbl == "real":
                        first_r = False
                    elif first_a and lbl == "attack":
                        first_a = False

        numLabels = Counter(labels)
        # print(numLabels.keys())
        print("Data array contains {} items ({} real and {} attack)\n"
              "Labels array contains {} items\n".format(len(data), numLabels['real'], numLabels['attack'], len(labels)))
        # input("Press Enter to continue...")
    else:
        print("{} examples will be processed for TRAINING with ratio = {}\n".format(numData, ratio))
        for path in paths:
            images = getSamples(path)

            for image in images:
                img = cv2.imread(image)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hist, bins, lbp_img = descriptor.describe(gray)

                # extract the label from the image path, then update the
                # label and data lists
                lbl = image.split("/")[-2]
                labels.append(lbl)
                data.append(hist)

                if first_r and lbl == "real" or first_a and lbl == "attack":
                    plt_name = altsep.join((plot_path, "".join(
                        (lbl, "_example_{}_{}.png".format(descriptor.numPoints, descriptor.radius)))))
                    fig = plt.figure()
                    plt.subplot(131), plt.imshow(gray, cmap='gray')
                    plt.axis('off'), plt.title("Image original (gray)")

                    plt.subplot(132), plt.imshow(lbp_img, cmap='gray')
                    plt.axis('off'), plt.title("LBP")

                    plt.subplot(133), plt.hist(lbp_img.ravel(), int(lbp_img.max()+2), normed= 1, color='red')
                    # plt.title("LBP histogram")

                    fig.suptitle(
                        "{} image using LBP ({},{})".format(lbl, descriptor.numPoints, descriptor.radius),
                        fontsize=14)

                    plt.tight_layout()
                    plt.savefig(plt_name)
                    plt.show()

                    if first_r and lbl == "real":
                        first_r = False
                    elif first_a and lbl == "attack":
                        first_a = False
        numLabels = Counter(labels)
        # print(numLabels.keys())
        print("Data array contains {} items ({} real and {} attack)\n"
              "Labels array contains {} items\n".format(len(data), numLabels['real'], numLabels['attack'], len(labels)))
        # input("Press Enter to continue...")
    return data, labels



def countFiles(paths):
    total = 0
    for path in paths:
        total += len([name for name in listdir(path) if isfile(join(path,name))])
    return total


def getVideoNames(args, real_ext, attack_ext):
    training_real = natSort(getSamples(args["videoRealtr"], real_ext))
    training_attack_f = natSort(getSamples(args["videoAttackFtr"], attack_ext))
    training_attack_h = natSort(getSamples(args["videoAttackHtr"], attack_ext))
    test_real = natSort(getSamples(args["videoRealte"], real_ext))
    test_attack_f = natSort(getSamples(args["videoAttackFte"], attack_ext))
    test_attack_h = natSort(getSamples(args["videoAttackHte"], attack_ext))

    return training_real, training_attack_f, training_attack_h, test_real, test_attack_f, test_attack_h
    
def makeAllDirs(args):
    print("\nCreating destination paths: \n'{}'\n'{}'\n'{}'\n{}\n{}\n{}\n"
          .format(args['outputRealtr'], args['outputAttackFtr'], args['outputAttackHtr'],
                  args['outputRealte'], args['outputAttackFte'], args['outputAttackHte']))
    makeDir(args['outputRealtr'])
    makeDir(args['outputAttackFtr'])
    makeDir(args['outputAttackHtr'])
    makeDir(args['outputRealte'])
    makeDir(args['outputAttackFte'])
    makeDir(args['outputAttackHte'])
    
    
def getSamples(path, ext=''):
    samples = [altsep.join((path, f)) for f in listdir(path)
              if isfile(join(path, f)) and f.endswith(ext)]
    return samples


def makeDir(path):
    '''
    To create output path if doesn't exist
    see: https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
    :param path: path to be created
    :return: none
    '''
    try:
        if not exists(path):
            makedirs(path)
            print("\nCreated '{}' folder\n".format(path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def natSort(list):
    '''
    Sort frames with human method
    see: https://pypi.python.org/pypi/natsort
    :param list: list that will be sorted
    :return: sorted list
    '''
    return natsorted(list, alg=ns.IGNORECASE)


def plotImages(titles, images, title, row, col):
    fig = plt.figure()
    for i in range(len(images)):
        plt.subplot(row, col, i + 1), plt.imshow(images[i])     #, 'gray'
        if len(titles) != 0:
            plt.title(titles[i])
        plt.gray()
        plt.axis('off')
    fig.suptitle(title, fontsize=14)
    plt.show()


# def getVideoDetails(filepath):
#     tmpf = tempfile.NamedTemporaryFile()
#     system("ffmpeg -i \"%s\" 2> %s" % (filepath, tmpf.name))
#     lines = tmpf.readlines()
#     tmpf.close()
#     metadata = {}
#     for l in lines:
#         l = l.strip()
#         if l.startswith('Duration'):
#             metadata['duration'] = re.search('Duration: (.*?),', l).group(0).split(':',1)[1].strip(' ,')
#             metadata['bitrate'] = re.search("bitrate: (\d+ kb/s)", l).group(0).split(':')[1].strip()
#         if l.startswith('Stream #0:0'):
#             metadata['video'] = {}
#             metadata['video']['codec'], metadata['video']['profile'] = \
#                 [e.strip(' ,()') for e in re.search('Video: (.*? \(.*?\)),? ', l).group(0).split(':')[1].split('(')]
#             metadata['video']['resolution'] = re.search('([1-9]\d+x\d+)', l).group(1)
#             metadata['video']['bitrate'] = re.search('(\d+ kb/s)', l).group(1)
#             metadata['video']['fps'] = re.search('(\d+ fps)', l).group(1)
#         if l.startswith('Stream #0:1'):
#             metadata['audio'] = {}
#             metadata['audio']['codec'] = re.search('Audio: (.*?) ', l).group(1)
#             metadata['audio']['frequency'] = re.search(', (.*? Hz),', l).group(1)
#             metadata['audio']['bitrate'] = re.search(', (\d+ kb/s)', l).group(1)
#     return metadata