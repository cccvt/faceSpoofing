from os import listdir, makedirs, errno
from os.path import isfile, join, altsep
from natsort import natsorted, ns
from matplotlib import pyplot as plt


def getSamples(path):
    samples = [altsep.join((path, f)) for f in listdir(path)
              if isfile(join(path, f)) and f.endswith('.mov')]
    return samples

def makeDir(path):
    '''
    To create output path if doesn't exist
    see: https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
    :param path: path to be created
    :return: none
    '''
    try:
        makedirs(path)
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