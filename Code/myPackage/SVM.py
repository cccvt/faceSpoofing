from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from os.path import join
from sklearn.svm import SVC
import pickle

from Code.myPackage import tools as tl




def trainTest(tuned_parameters, scores, X_train, Y_train, X_test, Y_test):
    for score in scores:
        print("# Tuning hyper-parameters for '%s'" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s_macro' % score, verbose= 1, n_jobs= 5)
        clf.fit(X_train, Y_train)

        print("Results for development set:\n".format(clf.cv_results_))
        print("Best parameters set found on development set:\n".format(clf.best_params_))
        print("\nGrid scores on development set:\n")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

        # Save best model
        modelName = join("SVM_", clf.best_params_['kernel'],"_",clf.best_params_['C'])
        pickle.dump(clf.best_estimator_, open(modelName, 'wb'))

        print("\nDetailed classification report:")
        print("\nThe model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.\n")
        Y_true, Y_pred = Y_test, clf.predict(X_test)
        print(classification_report(Y_true, Y_pred))