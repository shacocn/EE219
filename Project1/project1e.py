import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.decomposition import NMF
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import string

import project1d

def classify(data):
    data.target = list(map (lambda x: int(x<4), data.target))

def plot_roc(target, guess, method):
    fpr, tpr, thresholds = roc_curve(target, guess)
    plt.plot(fpr, tpr, label="ROC curve")
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.2])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    titl = 'ROC curves for' + method
    plt.title(titl)
    plt.legend(loc="best")
    plt.show()

def print_stats (target, guess, method):
    print ('stats for:', method)
    print ('accuracy is:', accuracy_score(target, guess))
    print ('precision is:',precision_score(target, guess, average='macro'))
    print ('recall score is:',recall_score(target, guess, average='macro'))
    print ('confusion matrix is:',confusion_matrix(target, guess))

if __name__ == "__main__":
    stop_words = text.ENGLISH_STOP_WORDS

    categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
    train_data = fetch_20newsgroups(subset='train', categories=categories,shuffle=True, random_state=42)
    test_data = fetch_20newsgroups(subset='test', categories=categories,shuffle=True, random_state=42)
    classify(train_data)
    classify(test_data)
    
    # for LSI
    svd_train, svd_test = project1d.fetch_LSI(train_data, test_data)
    clf_lsi = SVC(kernel='linear',C=1000, probability = True)
    clf_lsi.fit(svd_train, train_data.target)
    prediction_lsi = clf_lsi.predict(svd_test)
    prediction_prob_lsi = clf_lsi.predict_proba(svd_test)

    # for NMF
    nmf_train, nmf_test = project1d.fetch_NMF(train_data, test_data)
    clf_nmf = SVC(kernel='linear',C=1000, probability = True)
    clf_nmf.fit(nmf_train, train_data.target)
    prediction_nmf = clf_nmf.predict(nmf_test)
    prediction_prob_nmf = clf_nmf.predict_proba(nmf_test)

    #print stats for both LSI and NMF
    print ('stats for hard SVM (C=1000)')
    print_stats (test_data.target, prediction_lsi, 'LSI')
    plot_roc(test_data.target, prediction_prob_lsi[:,1], 'LSI')

    print_stats (test_data.target, prediction_nmf, 'NMF')
    plot_roc(test_data.target, prediction_prob_nmf[:,1], 'NMF')


    #also need to do soft SVM, C = 0.001
    
    # for LSI
    svd_train, svd_test = project1d.fetch_LSI(train_data, test_data)
    clf_lsi = SVC(kernel='linear',C=0.001, probability = True)
    clf_lsi.fit(svd_train, train_data.target)
    prediction_lsi = clf_lsi.predict(svd_test)
    prediction_prob_lsi = clf_lsi.predict_proba(svd_test)
    
    # for NMF
    nmf_train, nmf_test = project1d.fetch_NMF(train_data, test_data)
    clf_nmf = SVC(kernel='linear',C=0.001, probability = True)
    clf_nmf.fit(nmf_train, train_data.target)
    prediction_nmf = clf_nmf.predict(nmf_test)
    prediction_prob_nmf = clf_nmf.predict_proba(nmf_test)
    
    #print stats for both LSI and NMF
    print ('stats for soft SVM (C=0.001)')
    print_stats (test_data.target, prediction_lsi, 'LSI')
    plot_roc(test_data.target, prediction_prob_lsi[:,1], 'LSI')
    
    print_stats (test_data.target, prediction_nmf, 'NMF')
    plot_roc(test_data.target, prediction_prob_nmf[:,1], 'NMF')


