import numpy as np
import pandas as pd
from collections import Counter


class my_evaluation:
    def __init__(self, predictions, actuals, pred_proba=None):
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None

    def confusion_og(self):
        y_test = self.actuals
        predicted_values = self.predictions
        unique_classes = np.unique(y_test) 

        clist = unique_classes.tolist()

        for idx, x in np.ndenumerate(predicted_values):
            predicted_values[idx] = clist.index(x)

        no_of_classes = len(np.unique(y_test)) 
        no_of_rows = y_test.shape[0]  

        confusion_matrix = np.zeros((no_of_classes, no_of_classes), dtype=np.int64)

        for classes in range(no_of_classes):
            for i in range(no_of_rows):

                if y_test[i] == unique_classes[classes] and y_test[i] == predicted_values[i]:
                    confusion_matrix[classes, classes] = confusion_matrix[classes, classes] + 1
                elif y_test[i] == unique_classes[classes] and y_test[i] != predicted_values[i]:
                    confusion_matrix[classes, predicted_values[i]] = confusion_matrix[classes, predicted_values[i]] + 1

        return confusion_matrix

    def confusion(self):
        correct = self.predictions == self.actuals
        self.acc = float(Counter(correct)[True]) / len(correct)
        self.confusion_matrix = {}

        confustion_matrix = self.confusion_og()
        matrix = np.zeros((confustion_matrix.shape[0], 4), dtype=np.int32)
    

        for i in range(4):
            for y in range(confustion_matrix.shape[0]):
                if i == 0: 
                    matrix[y, i] = confustion_matrix[y, y]
                elif i == 1:
                    matrix[y, i] = np.sum(confustion_matrix[y, :]) - confustion_matrix[y, y]
                elif i == 2:
                    matrix[y, i] = np.sum(confustion_matrix) - np.sum(confustion_matrix[y, :]) - np.sum(
                        confustion_matrix[:, y]) + confustion_matrix[y, y]
                elif i == 3:
                    matrix[y, i] = np.sum(confustion_matrix[:, y]) - confustion_matrix[y, y]

        for x, label in enumerate(self.classes_):
            tp = matrix[x, 0]
            fp = matrix[x, 3]
            tn = matrix[x, 2]
            fn = matrix[x, 1]
            self.confusion_matrix[label] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
        return

    def accuracy(self):
        if self.confusion_matrix == None:
            self.confusion()
        return self.acc

    def precision(self, target=None, average="macro"):
        #         print(self.confusion_matrix)
        # compute precision
        # target: target class (str). If not None, then return precision of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average precision
        # output: prec = float
        # note: be careful for divided by 0

        if self.confusion_matrix == None:
            self.confusion()
        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fp = self.confusion_matrix[target]["FP"]
            if tp + fp == 0:
                prec = 0
            else:
                prec = float(tp) / (tp + fp)
        else:
            if average == "micro":
                prec = self.accuracy()
            else:
                prec = 0
                n = len(self.actuals)
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fp = self.confusion_matrix[label]["FP"]
                    if tp + fp == 0:
                        prec_label = 0
                    else:
                        prec_label = float(tp) / (tp + fp)
                    #                         print("test precision "+str(prec_label))
                    if average == "macro":
                        ratio = 1 / len(self.classes_)
                    elif average == "weighted":
                        ratio = Counter(self.actuals)[label] / float(n)
                    else:
                        raise Exception("Unknown type of average.")
                    prec += prec_label * ratio
        return prec

    def recall(self, target=None, average="macro"):
        if self.confusion_matrix == None:
            self.confusion()
        if target in self.classes_:
            tp = self.confusion_matrix[target]["TP"]
            fp = self.confusion_matrix[target]["FN"]
            if tp + fp == 0:
                prec = 0
            else:
                prec = float(tp) / (tp + fp)
        else:
            if average == "micro":
                prec = self.accuracy()
            else:
                prec = 0
                n = len(self.actuals)
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fp = self.confusion_matrix[label]["FN"]
                    if tp + fp == 0:
                        prec_label = 0
                    else:
                        prec_label = float(tp) / (tp + fp)
                    if average == "macro":
                        ratio = 1 / len(self.classes_)
                    elif average == "weighted":
                        ratio = Counter(self.actuals)[label] / float(n)
                    else:
                        raise Exception("Unknown type of average.")
                    prec += prec_label * ratio
        return prec

    def f1(self, target=None, average="macro"):
        
        if target:
            prec = self.precision(target=target, average=average)
            rec = self.recall(target=target, average=average)
            if prec + rec == 0:
                f1_score = 0
            else:
                f1_score = 2.0 * prec * rec / (prec + rec)
        else:
            prec = self.precision(target=target, average=average)
            rec = self.recall(target=target, average=average)
            if prec + rec == 0:
                f1_score = 0
            else:
                f1_score = 2.0 * prec * rec / (prec + rec)

        return f1_score

    def auc(self, target):
        
        if type(self.pred_proba) == type(None):
            return None
        else:
            if target in self.classes_:
                order = np.argsort(self.pred_proba[target])[::-1]
                tp = 0
                fp = 0
                fn = Counter(self.actuals)[target]
                tn = len(self.actuals) - fn
                tpr = 0
                fpr = 0
                auc_target = 0
                for i in order:
                    if self.actuals[i] == target:
                        tp = tp + 1
                        fn = fn - 1
                        tpr = tp / (tp + fn)
                    else:
                        fp = fp + 1
                        tn = tn - 1
                        pre_fpr = fpr
                        fpr = fp / (fp + tn)
                        auc_target = auc_target + tpr * (fpr - pre_fpr)
            else:
                raise Exception("Unknown target class.")

            return auc_target