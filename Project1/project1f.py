import numpy as np
from sklearn.model_selection import train_test_split

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
from sklearn.model_selection import cross_val_score
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import string

import project1e
import project1b

def find_best_parameter(svd_train, train_data):
    expo = list(range(-3,4))
    scores = []
    for e in expo:
        clr = SVC(kernel='linear', C=10 ** (e))
        scores.append(np.mean(cross_val_score(clr, svd_train, train_data.target, cv=5)))
    return expo[scores.index(max(scores))]

if __name__ == "__main__":
    stop_words = text.ENGLISH_STOP_WORDS
    categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
    train_data = fetch_20newsgroups(subset='train', categories=categories,shuffle=True, random_state=42)
    test_data = fetch_20newsgroups(subset='test', categories=categories,shuffle=True, random_state=42)
    
    project1e.classify(train_data)
    project1e.classify(test_data)

    count_vect = CountVectorizer(tokenizer=project1b.tokenizer_class(), stop_words = stop_words, lowercase= True, min_df = 2, max_df = 0.99)
    tf_transformer = TfidfTransformer(use_idf=False)

    # for LSI
    print ('stats for LSI')
    svd_model = TruncatedSVD(n_components=50)
    pipeline = Pipeline([
                         ('vectorize',count_vect),
                         ('tf-idf',tf_transformer),
                         ('svd',svd_model)
                         ])

    svd_train = pipeline.fit_transform(train_data.data)
    svd_test = pipeline.transform(test_data.data)

    # find best parameter
    k = find_best_parameter(svd_train, train_data)
    print ('best score is obtained with k=: ',k)

    # perform SVM on the parameter
    clr = SVC(kernel='linear', C=10 ** (k), probability = True)
    cross_val_score(clr, svd_train, train_data.target, cv=5)
    clr.fit(svd_train, train_data.target)

    #print statistics
    prediction = clr.predict(svd_test)
    prediction_prob = clr.predict_proba(svd_test)
    
    project1e.print_stats (test_data.target, prediction, 'LSI')
    project1e.plot_roc(test_data.target, prediction_prob[:,1],'LSI')

    # for NMF
    print ('stats for NMF')
    nmf_model = NMF(n_components=50)
    pipeline = Pipeline([
                         ('vectorize',count_vect),
                         ('tf-idf',tf_transformer),
                         ('nmf',nmf_model)
                         ])
        
    nmf_train = pipeline.fit_transform(train_data.data)
    nmf_test = pipeline.transform(test_data.data)

    # find best parameter
    k = find_best_parameter(nmf_train, train_data)
    print ('best score is obtained with k=: ',k)

    # perform SVM on the parameter
    clr = SVC(kernel='linear', C=10 ** (k), probability = True)
    cross_val_score(clr, nmf_train, train_data.target, cv=5)
    clr.fit(svd_train, train_data.target)

    #print statistics
    prediction = clr.predict(nmf_test)
    prediction_prob = clr.predict_proba(nmf_test)
    
    project1e.print_stats (test_data.target, prediction, 'NMF')
    project1e.plot_roc(test_data.target, prediction_prob[:,1],'NMF')


