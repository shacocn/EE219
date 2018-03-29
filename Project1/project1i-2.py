from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import sklearn.metrics
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import PorterStemmer

from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download('stopwords')
import re
import string

import project1g
import project1b
import project1d
import project1e
import project1f


def print_stats(actual, predicted):
    print ("Accuracy is ", sklearn.metrics.accuracy_score(actual, predicted) * 100)
    print ("Precision is ", sklearn.metrics.precision_score(actual, predicted, average='macro') * 100)

    print ("Recall is ", sklearn.metrics.recall_score(actual, predicted, average='macro') * 100)

    print ("Confusion Matrix is ", sklearn.metrics.confusion_matrix(actual, predicted))



def perform_classification(clf):
    global svd_train, svd_test, train, test

    clf.fit(svd_train, train_data.target)
    predicted = clf.predict(svd_test)
    print_stats(test_data.target, predicted)

if __name__ == "__main__":


    stop_words = text.ENGLISH_STOP_WORDS

    categories = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']

    train_data = fetch_20newsgroups(subset='train', categories=categories,shuffle=True, random_state=42)
    test_data = fetch_20newsgroups(subset='test', categories=categories,shuffle=True, random_state=42)

    svd_train, svd_test = project1d.fetch_LSI(train_data, test_data)


    clf_list = [OneVsOneClassifier(GaussianNB()), OneVsOneClassifier(svm.SVC(kernel='linear')), OneVsRestClassifier(GaussianNB()), OneVsRestClassifier(svm.SVC(kernel='linear'))]
    clf_name = ['OneVsOneClassifier Naive Bayes', 'OneVsOneClassifier SVM','OneVsRestClassifier Naive Bayes', 'OneVsRestClassifier SVM']




print ("One Vs One Classification using Naive Bayes")
perform_classification(OneVsOneClassifier(GaussianNB()))

print ("One Vs Rest Classifciation using Naive Bayes")
perform_classification(OneVsRestClassifier(GaussianNB()))

print ("One Vs One Classification using SVM")
perform_classification(OneVsOneClassifier(svm.SVC(kernel='linear')))

print ("One Vs Rest Classificaiton using SVM")
perform_classification(OneVsRestClassifier(svm.SVC(kernel='linear')))
