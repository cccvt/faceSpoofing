from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pickle
import time


from Code.myPackage import tools as tl






def trainTest(tuned_parameters, scores, X_train, Y_train, X_test, Y_test, model_path, log_path, plot_path, comb):
    logName = tl.altsep.join((log_path,"".join(("LogFile_",time.strftime("%d-%m-%Y"),"_",str(comb[0]), "_", str(comb[1]),".txt"))))
    file = open(logName, 'w')
    class_names = ['real', 'attack']

    for score in scores:
        print("# Tuning hyper-parameters for '{}' and LBP({}, {})".format(score, comb[0], comb[1]))
        print()
        file.write("# Tuning hyper-parameters for '{}' and LBP({}, {})\n\n".format(score, comb[0], comb[1]))

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5, n_jobs= -1,
                           scoring='%s_macro' % score, verbose= 0, error_score= 0)
        clf.fit(X_train, Y_train)

        print("\nResults for development set:")
        file.write("\nResults for development set:\n")
        # print(clf.cv_results_)
        # for x in clf.cv_results_:
        #     print(x)
        #     for y in clf.cv_results_[x]:
        #         print(y, ':', clf.cv_results_[x][y])
        for key, val in clf.cv_results_.items():
            print(key, ": ", val)
            file.write("{}: {}\n".format(key, val))

        print("\nBest parameters set found on development set:")
        file.write("\nBest parameters set found on development set:\n")
        print(clf.best_params_)
        file.write("{}\n".format(clf.best_params_))

        print("\nGrid scores on development set:")
        file.write("\nGrid scores on development set:\n")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
            file.write("{:.3f} (+/-{:.03}) for {}\n".format(mean, std*2, params))

        # Save best model
        modelName = tl.altsep.join((model_path,"".join(("SVM_",str(clf.best_params_['kernel']),"_",str(score),"_",str(clf.best_params_['C']),"_", str(comb[0]), "_", str(comb[1]),".model"))))
        print("Ful path for the model: '{}'".format(modelName))
        file.write("Ful path for the model: '{}'\n".format(modelName))
        pickle.dump(clf.best_estimator_, open(modelName, 'wb'))

        print("\nDetailed classification report:")
        file.write("\nDetailed classification report:\n")

        print("\nThe model is trained on the full development set.")
        file.write("\nThe model is trained on the full development set.\n")

        print("The scores are computed on the full evaluation set.\n")
        file.write("\nThe scores are computed on the full evaluation set.\n")
        Y_true, Y_pred = Y_test, clf.predict(X_test)
        print(classification_report(Y_true, Y_pred))
        file.write("{}\n\n\n".format(classification_report(Y_true, Y_pred)))


        # Compute confusion matrix
        cnf_matrix = confusion_matrix(Y_test, Y_pred)
        tl.np.set_printoptions(precision= 3)

        # Plot non-normalized confusion matrix
        # plt.figure()
        # plot_confusion_matrix(cnf_matrix, classes= class_names,
        #                       title='Confusion matrix, without normalization ' + str(score))

        # Plot normalized confusion matrix
        tl.plot_confusion_matrix(plot_path, cnf_matrix, classes= class_names, normalize=True,
                              title='Normalized confusion matrix for {} with LBP({},{})'.format(score, comb[0], comb[1]))

    file.close()