import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import sys
from sklearn.svm import SVC
from pdb import set_trace
import re
##################################
sys.path.insert(0,'../..')
from assignments.assignment8.my_evaluation import my_evaluation
from assignments.assignment9.my_GA import my_GA

class my_model():

    def obj_func(self, predictions, actuals, pred_proba=None):
        # One objectives: higher f1 score
        eval = my_evaluation(predictions, actuals, pred_proba)
        return [eval.f1()]

    

    def fit(self, X, y):

        def clean(text):
            text=text.lower()
            obj=re.compile(r"<.*?>")                     #removing html tags
            text=obj.sub(r" ",text)
            obj=re.compile(r"https://\S+|http://\S+")    #removing url
            text=obj.sub(r" ",text)
            obj=re.compile(r"[^\w\s]")                   #removing punctuations
            text=obj.sub(r" ",text)
            obj=re.compile(r"\d{1,}")                    #removing digits
            text=obj.sub(r" ",text)
            obj=re.compile(r"_+")                        #removing underscore
            text=obj.sub(r" ",text)
            obj=re.compile(r"\s\w\s")                    #removing single character
            text=obj.sub(r" ",text)
            obj=re.compile(r"\s{2,}")                    #removing multiple spaces
            text=obj.sub(r" ",text)

            
            return "".join(text)

        col_drop=['title', 'location','requirements', 'telecommuting', 'has_company_logo', 'has_questions']
        X.drop(col_drop,axis=1,inplace=True)

        X["description"]=X["description"].apply(clean)
        self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False)
        XX = self.preprocessor.fit_transform(X["description"])
        XX = pd.DataFrame(XX.toarray())
        ga = my_GA(SGDClassifier, XX, y, {"loss": ("hinge", "log_loss", "perceptron"), "penalty": ("l2", "l1"), "alpha": [0.0001, 0.01]}, self.obj_func, generation_size=50,
                   crossval_fold=5,
                    max_generation=10, max_life=2)
        best = ga.tune()[0]
        dec_dict = {key: best[i] for i, key in enumerate(["loss", "penalty", "alpha"])}
        self.clf =  SGDClassifier(alpha=0.001,
                    class_weight={1:0.5, 0:0.5},
                    eta0=10,
                    learning_rate='adaptive',
                    loss='perceptron', penalty='l2')
        self.clf.fit(XX,y)
        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions

        def clean(text):
            text=text.lower()
            obj=re.compile(r"<.*?>")                     #removing html tags
            text=obj.sub(r" ",text)
            obj=re.compile(r"https://\S+|http://\S+")    #removing url
            text=obj.sub(r" ",text)
            obj=re.compile(r"[^\w\s]")                   #removing punctuations
            text=obj.sub(r" ",text)
            obj=re.compile(r"\d{1,}")                    #removing digits
            text=obj.sub(r" ",text)
            obj=re.compile(r"_+")                        #removing underscore
            text=obj.sub(r" ",text)
            obj=re.compile(r"\s\w\s")                    #removing single character
            text=obj.sub(r" ",text)
            obj=re.compile(r"\s{2,}")                    #removing multiple spaces
            text=obj.sub(r" ",text)

            
            return "".join(text)

        X["description"]=X["description"].apply(clean)
        
        XX = self.preprocessor.transform(X["description"])
        predictions = self.clf.predict(XX)
        return predictions