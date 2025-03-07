from trainML import *
import numpy as np

def classifiers(args,X_train, Y_train, X_test, Y_test):
    x_train = X_train.flatten()[:, np.newaxis]
    y_train = Y_train.flatten()
    x_test = X_test.flatten()[:, np.newaxis]
    y_test = Y_test.flatten()
    for classifier in args.classifiers:
        if classifier == 'knn':
            y_score = train_KNN(X_train, Y_train, X_test, Y_test)
            return y_score


        if classifier == 'AdaBoost':
            y_score = train_AdaBoost(x_train, y_train, x_test, y_test)
            y_pred = [int(item > args.threshold) for item in y_score.flatten()]
            return y_score


        if classifier == 'xgboost':
            y_score = train_xgboost(x_train, y_train, x_test, y_test)
            y_pred = [int(item > args.threshold) for item in y_score.flatten()]
            return y_score



        if classifier == 'SVM':
            y_score = train_SVM(x_train, y_train, x_test, y_test)
            y_pred = [int(item > args.threshold) for item in y_score.flatten()]
            return y_score



        if classifier == 'LR':
            y_score = train_LR(x_train, y_train, x_test, y_test)
            y_pred = [int(item > args.threshold) for item in y_score.flatten()]
            return y_score

