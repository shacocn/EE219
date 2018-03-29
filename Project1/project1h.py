
from sklearn.linear_model import LogisticRegression

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

if __name__ == "__main__":

    stop_words = text.ENGLISH_STOP_WORDS

    categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
    train_data = fetch_20newsgroups(subset='train', categories=categories,shuffle=True, random_state=42)
    test_data = fetch_20newsgroups(subset='test', categories=categories,shuffle=True, random_state=42)


    project1e.classify(train_data)
    project1e.classify(test_data)
    # fetch LSI representation

    count_vect = CountVectorizer(tokenizer=project1b.tokenizer_class(), stop_words = stop_words, lowercase= True, min_df = 2, max_df = 0.99)
    #need to try min_df = 2, and min_df = 5

    #svd_train = pipeline.fit_transform(train_data.data)
    #svd_test = pipeline.transform(test_data.data)

    classifier = LogisticRegression()
    svd_train, svd_test = project1d.fetch_LSI(train_data, test_data)
    nmf_train, nmf_test = project1d.fetch_NMF(train_data, test_data)


    print ("Training Logistic Regression classifier")
    classifier.fit(svd_train, train_data.target)

    print ("Predicting classifications of testing dataset")
   # LogisticRegressionpredicted_svd = classifier.predict(svd_test)
    #prediction_prob_lsi = classifier.predict_proba(svd_test)

    LogisticRegressionpredicted_nmf = classifier.predict(nmf_test)
    prediction_prob_nmf = classifier.predict_proba(nmf_test)

    print ("Statistics of Logistic Regression classifiers:")

    #lsi
  #  project1e.print_stats (test_data.target, LogisticRegressionpredicted_svd, 'LSI')
  #  project1e.plot_roc(test_data.target, prediction_prob_lsi[:,1], 'LogisticRegression - LSI')

    #nmf
    project1e.print_stats (test_data.target, LogisticRegressionpredicted_nmf, 'NMF')
    project1e.plot_roc(test_data.target, prediction_prob_nmf[:,1], 'LogisticRegression - NMF')
