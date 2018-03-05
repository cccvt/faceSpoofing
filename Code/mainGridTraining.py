import sys
import argparse
import time
import itertools
import winsound
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
    ap.add_argument("-model", "--model_path", required=True,
                    help="-model Path where the best model will be stored")
    ap.add_argument("-log", "--log_path", required=True,
                    help="-log Path where the log of the execution will be stored")
    ap.add_argument("-plt", "--plt_path", required=True,
                    help="-plt Path where the plots of the execution will be stored")
    ap.add_argument("-te_rt", "--test_ratio", required=False, default= 0.2,
                    help="-tr_rt Ratio for select testing data")

    args = vars(ap.parse_args())

    duration = 1000  # millisecond
    freq = 640  # Hz

    model_path = tl.altsep.join((args["model_path"],time.strftime("%d-%m-%Y-%H")))
    log_path = tl.altsep.join((args["log_path"], time.strftime("%d-%m-%Y-%H")))
    plot_path = tl.altsep.join((args["plt_path"], time.strftime("%d-%m-%Y-%H")))

    tl.makeDir(model_path)
    tl.makeDir(log_path)
    tl.makeDir(plot_path)

    # num_points = list(range(8, 17, 1))
    # radio = list(range(1, 3, 1))
    # combinations = list(itertools.product(num_points, radio))
    combinations = [(8, 1), (8, 2), (16, 2)]
    c_param = tl.np.arange(0.06, 0.16, 0.02)

    # print("--- INFORMATION ---\n"
    #       "Number of points {}\n"
    #       "Radios: {}\n"
    #       "Number of combinations: {}\n"
    #       "C: {}\n".format(num_points, radio, len(combinations), c_param))

    print("--- INFORMATION ---\n"
          "Combinations: {}\n"
          "C: {}\n".format(len(combinations), c_param))
    # print(c_param)

    for comb in combinations:
        # allPrints = tl.altsep.join(
        #     (log_path, "".join(("stdout_", time.strftime("%d-%m-%Y"), "_", str(comb[0]), "_", str(comb[1]), ".txt"))))
        # sys.stdout = open(allPrints, 'w')
        descriptor = lbp_Class(comb[0], comb[1])

        tuned_parameters = [{'kernel': ['linear'], 'C': c_param}]
        # tuned_parameters = [{'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100], 'max_iter': [-1, 1e3, 1e4, 1e5, 1e6]}]
        # scores = ['precision', 'recall', 'accuracy', 'roc_auc']
        scores = ['precision', 'recall']


        training_path = [args["training_r"], args["training_att"]]
        test_path = [args["test_r"], args["test_att"]]
        # print(test_path)

        print("\nPreparing TRAINING data...\n")
        start = time.time()
        X_train, Y_train = tl.prepareData(training_path, descriptor, plot_path)
        print("\nThe task took {:.3f} seconds".format(time.time()-start))
        print("DONE!!\n")

        print("Preparing TESTING data...")
        start = time.time()
        X_test, Y_test = tl.prepareData(test_path, descriptor, plot_path, float(args["test_ratio"]))
        print("\nThe task took {:.3f} seconds".format(time.time() - start))
        print("DONE!!\n")

        print("\nTraining and testing the model...\n"
              "It may took several time. Coffee break!!\n")
        print("Start at {}".format(time.strftime("%c")))
        trainTest(tuned_parameters, scores, X_train, Y_train, X_test, Y_test, model_path, log_path, plot_path, comb)
        print("\nDONE!!\n"
              "Finished at {}".format(time.strftime("%c")))

    print("Execution finished!!")

    for i in range(6):
        winsound.Beep(freq, duration)
    exit()