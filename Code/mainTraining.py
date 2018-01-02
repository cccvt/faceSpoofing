import argparse
import time
from Code.myPackage.SVM import trainTest
from Code.myPackage.lbp import lbp_Class
from Code.myPackage import tools as tl

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-tr_r", "--training_r", required = True,
                    help = "-tr_r Path with real faces for training")
    ap.add_argument("-tr_att", "--training_att", required=True,
                    help="-tr_att Path with attacks for training (fixed)")
    ap.add_argument("-te_r", "--test_r", required=True,
                    help="-tr_r Path with real faces for test")
    ap.add_argument("-te_att", "--test_att", required=True,
                    help="-tr_att Path with attacks for test (fixed)")
    ap.add_argument("-out", "--out_path", required=True,
                    help="-out Path where the best model will be stored")
    ap.add_argument("-te_rt", "--test_ratio", required=False, default= 0.2,
                    help="-tr_rt Ratio for select testing data")

    args = vars(ap.parse_args())

    tl.makeDir(args["out_path"])

    descriptor = lbp_Class(24, 8)

    tuned_parameters = [{'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]
    scores = ['precision', 'recall', 'accuracy', 'roc_auc']

    training_path = [args["training_r"], args["training_att"]]
    test_path = [args["test_r"], args["test_att"]]

    print("\nPreparing TRAINING data...\n")
    start = time.time()
    X_train, Y_train = tl.prepareData(training_path, descriptor)
    print("\nThe task took {} seconds".format(time.time()-start))
    print("DONE!!\n")

    print("Preparing TESTING data...")
    start = time.time()
    X_test, Y_test = tl.prepareData(test_path, descriptor, float(args["test_ratio"]))
    print("\nThe task took {} seconds".format(time.time() - start))
    print("DONE!!\n")

    print("\nTraining and testing the model...\n"
          "It may took several time. Coffee break!!\n")
    trainTest(tuned_parameters, scores, X_train, Y_train, X_test, Y_test)
    print("\nDONE!!\n")