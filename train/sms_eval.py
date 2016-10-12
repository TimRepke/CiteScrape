import sklearn
import warnings
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import multiprocessing as mp
from pprint import pprint
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.metrics import auc, precision_recall_curve
from sms_pandas import label_dict


class Evaluator:
    def __init__(self, feature_frame, mask, feature_weights):
        self.__feature_frame = feature_frame
        self.__mask = mask

        self.__predictframe = None

        self.__feature_weights = feature_weights
        self.__features = sorted([key for key, value in feature_weights.items() if value])
        print('num features in frame', len(list(feature_frame.columns)))
        print('num feature weights', len(feature_weights))
        print('num selected features', len(self.__features))

        new_features = list(set(list(feature_frame.columns)) - set(list(feature_weights.keys())))
        if len(new_features) > 0:
            print('Warning: there are new features! Consider updating the map!')
            pprint(new_features)

    def get_keep_features(self):
        return self.__features

    def print_updated_featuremap(self):
        print('new features:')
        pprint(list(set(list(self.__feature_frame.columns)) - set(list(self.__feature_weights.keys()))))

        print('updated features')
        for k in sorted(list(self.__feature_frame.columns)):
            print('    "' + k + '": ' + (
                str(self.__feature_weights[k]) if k in list(self.__feature_weights.keys()) else 'XX') + ',')

    def get_traintest(self):
        reduced_frame = self.__feature_frame[self.__mask]
        el_target = reduced_frame['label']

        el_data = reduced_frame[keep_features].fillna(0)  # .clip(lower=0)
        X_train_p, X_test_p, y_train, y_test = train_test_split(el_data, el_target, test_size=0.25,
                                                                random_state=33, stratify=el_target)

        nb_classes = len(label_dict)

        Y_train = np_utils.to_categorical(np.array(list(y_train)), nb_classes)
        Y_test = np_utils.to_categorical(np.array(list(y_test)), nb_classes)

        X_train = X_train_p.as_matrix()
        X_test = X_test_p.as_matrix()
        return (X_train, Y_train, X_test, Y_test)

    def get_traintest_frame(self, label_remap={0: 0, 1: 1, 2: 2, 3: 3}, reduction=1.0, reduction_min=6,
                            testsetsplit=0.3, testsetfix=None):
        """
        :param label_remap: how labes should be renamed, set to np.nan to drop a label
        :param reduction: define how many unassigned samples to remove (1.0 = keep all, 0.0 remove all) per instance
        """

        reduced_frame = self.__feature_frame[self.__mask]
        nb_classes = len(set(label_remap.values()) - set([np.nan]))

        instancelist = list(set(reduced_frame['instance']))
        if testsetfix:
            testset = testsetfix
        else:
            testset = np.random.choice(np.array(instancelist), size=int(len(instancelist) * testsetsplit),
                                       replace=False)

        test = reduced_frame[reduced_frame['instance'].isin(list(testset))]
        train = reduced_frame[reduced_frame['instance'].isin(list(set(instancelist) - set(testset)))]

        if list(label_remap.values()) != list(label_remap.keys()):
            test.replace(to_replace={'label': list(label_remap.keys())},
                         value={'label': list(label_remap.values())}, inplace=True)
            test.dropna(subset=['label'], inplace=True)
            train.replace(to_replace={'label': list(label_remap.keys())},
                          value={'label': list(label_remap.values())}, inplace=True)
            train.dropna(subset=['label'], inplace=True)

        train = pd.concat([
                              labgrp.sample(
                                      n=reduction_min if len(
                                          labgrp) * reduction <= reduction_min and lab == 3 else None,
                                      frac=reduction if len(
                                              labgrp) * reduction > reduction_min and lab == 3 else None if lab == 3 else 1.0,
                                      replace=((len(labgrp) < reduction_min) and (lab == 3)))
                              for _, inst in train.groupby('instance') for lab, labgrp in inst.groupby('label')])
        train = train.iloc[np.random.permutation(len(train))]

        class_weights = dict((key, value) for key, value in enumerate(
                sklearn.utils.compute_class_weight('balanced', np.arange(nb_classes), np.array(train['label']))))

        print('num instances', len(instancelist))
        print('num train instances', len(list(set(instancelist) - set(testset))), ', rows:', train.shape[0])
        print('num test instances', len(testset), ', rows:', test.shape[0])

        self.__train = train
        self.__test = test

        return (nb_classes, class_weights, train, test)

    def get_traintest_nn(self, nb_neigh=6, label_remap={0: 0, 1: 1, 2: 2, 3: 3}, reduction=1.0, reduction_min=6,
                         reverse=False, flat_y=False, testsetsplit=0.3, testsetfix=None):
        nb_classes, class_weights, train, test = self.get_traintest_frame(label_remap, reduction, reduction_min,
                                                                          testsetsplit, testsetfix)
        X_train = []
        y_train = []
        Y_train = []
        X_test = []
        y_test = []
        Y_test = []

        for _, inst in train.groupby('instance'):
            knn = NearestNeighbors(nb_neigh + 1)  # + 1 because the self element will be excluded
            knn.fit(inst[['center_x', 'center_y']].as_matrix())
            for el in knn.kneighbors(inst[['center_x', 'center_y']].as_matrix(), return_distance=False):
                if reverse:
                    el = list(reversed(el))
                X_train.append(inst.iloc[el][self.__features].fillna(0).as_matrix())
                tmpy = np.array(list(inst.iloc[el]['label']), dtype=int)
                if flat_y:
                    tmpy = tmpy[-1] if reverse else tmpy[0]
                    Y_train.append(np_utils.to_categorical([tmpy], nb_classes)[0])
                else:
                    Y_train.append(np_utils.to_categorical(tmpy, nb_classes))
                y_train.append(tmpy)

        for _, inst in test.groupby('instance'):
            knn = NearestNeighbors(nb_neigh + 1)  # + 1 because the self element will be excluded

            tmp_inst = inst
            if tmp_inst.shape[0] < (nb_neigh + 1):
                tmp_inst = tmp_inst.append(tmp_inst.sample(n=(nb_neigh + 1 - len(tmp_inst))))

            knn.fit(tmp_inst[['center_x', 'center_y']].as_matrix())
            for el in knn.kneighbors(inst[['center_x', 'center_y']].as_matrix(), return_distance=False):
                if reverse:
                    el = list(reversed(el))
                X_test.append(tmp_inst.iloc[el][self.__features].fillna(0).as_matrix())
                tmpy = np.array(list(tmp_inst.iloc[el]['label']), dtype=int)
                if flat_y:
                    tmpy = tmpy[-1] if reverse else tmpy[0]
                    Y_test.append(np_utils.to_categorical([tmpy], nb_classes)[0])
                else:
                    Y_test.append(np_utils.to_categorical(tmpy, nb_classes))
                y_test.append(tmpy)

        return (test, train, np.array(X_train), np.array(y_train), np.array(Y_train),
                np.array(X_test), np.array(y_test), np.array(Y_test), nb_classes, class_weights)

    def get_traintest_i(self, label_remap={0: 0, 1: 1, 2: 2, 3: 3}, reduction=1.0, reduction_min=6, testsetsplit=0.3,
                        testsetfix=None):
        nb_classes, class_weights, train, test = self.get_traintest_frame(label_remap, reduction,
                                                                          reduction_min=reduction_min,
                                                                          testsetsplit=testsetsplit,
                                                                          testsetfix=testsetfix)

        X_train = np.array([inst[self.__features].fillna(0).as_matrix() for _, inst in train.groupby('instance')])
        X_test = np.array([inst[self.__features].fillna(0).as_matrix() for _, inst in test.groupby('instance')])

        y_train = np.array([np.array(i, dtype=int) for _, i in train.groupby('instance')['label']])
        Y_train = np.array([np_utils.to_categorical(i, nb_classes) for i in y_train])

        y_test = np.array([np.array(i, dtype=int) for _, i in test.groupby('instance')['label']])
        Y_test = np.array([np_utils.to_categorical(i, nb_classes) for i in y_test])

        return (test, train, X_train, y_train, Y_train, X_test, y_test, Y_test, nb_classes, class_weights)

    def get_traintest_i_flat(self, label_remap={0: 0, 1: 1, 2: 2, 3: 3}, reduction=1.0, reduction_min=6,
                             testsetsplit=0.3, testsetfix=None):
        nb_classes, class_weights, train, test = self.get_traintest_frame(label_remap, reduction,
                                                                          reduction_min=reduction_min,
                                                                          testsetsplit=testsetsplit,
                                                                          testsetfix=testsetfix)

        X_train = train[self.__features].fillna(0).as_matrix()  # .clip(lower=0)
        X_test = test[self.__features].fillna(0).as_matrix()

        y_train = np.array(list(train['label']), dtype=int)
        Y_train = np_utils.to_categorical(y_train, nb_classes)

        y_test = np.array(list(test['label']), dtype=int)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        return (test, train, X_train, y_train, Y_train, X_test, y_test, Y_test, nb_classes, class_weights)

    def measure_performance(self, X, y, clf, show_accuracy=True, show_classification_report=True,
                            show_confusion_matrix=True, show_curves=False):
        if type(clf) is type(list()):
            y_pred = clf
        elif type(clf) is type(np.array([])):
            if len(clf.shape) > 1:
                y_pred = clf.argmax(axis=1)
                y_pred_proba = clf
            else:
                y_pred = clf
        elif show_curves:
            y_pred_proba = clf.predict_proba(X)
            y_pred = y_pred_proba.argmax(axis=1)
        else:
            y_pred = clf.predict_classes(X, 20, 1)

        if show_accuracy:
            print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)))
        if show_classification_report:
            # print("Classification report")
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                print(metrics.classification_report(y, y_pred, target_names=['title', 'author', 'date', 'unassigned']))
        if show_confusion_matrix:
            # print("Confusion matrix")
            print(metrics.confusion_matrix(y, y_pred))
        if show_curves:
            plt.clf()
            for i, f in enumerate(['title', 'author', 'date', 'unassigned']):
                precision, recall, thresholds = precision_recall_curve(y == i, y_pred_proba[:, i])
                auc_ = auc(recall, precision)
                plt.plot(recall, precision, label="{} AUC={:.4}".format(f, auc_))
            plt.ylabel('Precision')
            plt.xlabel('Recall')
            plt.ylim(0, 1.1)
            plt.xlim(0, 1.1)
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            plt.grid()
            plt.show()

    def draw_traingraph(self, history):
        plt.subplot(121)
        plt.plot(history.history['acc'], label='accuracy')
        if 'val_acc' in history.history:
            plt.plot(history.history['val_acc'], label='eval_accuracy')
        plt.legend(loc='lower right')

        plt.subplot(122)
        plt.plot(history.history['loss'], label='loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='eval_loss')
        plt.legend(loc='upper right')

        plt.tight_layout(rect=(0, 0, 1.5, 1))
        plt.show()

    def get_predictframe_nested(self, predictions, maxp=True, avg=True, first=True):
        test = self.__test

    def get_predictframe(self, predictions, global_idxmax=False,
                         plabel_idxmax=False,
                         heatmap_adjusted=False, heatmap_bins=[10, 20], testfrm=None):
        if testfrm is not None:
            test = testfrm
        else:
            test = self.__test

        predictframe = pd.DataFrame([[inst, lab, 3, 3, 3, np.array(list(pred)).argmax(), 0] + list(pred)
                                     for inst, lab, pred in
                                     zip(list(test['instance']), list(test['label']), predictions)],
                                    columns=['instance', 'label', 'ilabel', 'gilabel', 'hmlabel', 'plabel',
                                             'adjusted', 'title', 'author', 'issued', 'unassigned'],
                                    index=test.index)

        if global_idxmax:
            for i, row in predictframe.groupby('instance').idxmax().iterrows():
                predictframe.loc[row['title'], 'gilabel'] = 0
                predictframe.loc[row['author'], 'gilabel'] = 1
                predictframe.loc[row['issued'], 'gilabel'] = 2

        if plabel_idxmax:
            for i, row in predictframe[predictframe['plabel'] != 3].groupby(['instance', 'plabel']).idxmax().iterrows():
                for l in range(3):
                    if i[1] == l:
                        predictframe.loc[row[['title', 'author', 'issued'][l]], 'ilabel'] = l

        self.__predictframe = predictframe

        return predictframe
