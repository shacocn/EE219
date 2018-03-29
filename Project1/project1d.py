from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import PorterStemmer
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import string

import project1b


def fetch_LSI(train_data, test_data):
    stop_words = text.ENGLISH_STOP_WORDS
    count_vect = CountVectorizer(tokenizer=project1b.tokenizer_class(), stop_words = stop_words, lowercase= True, min_df = 2, max_df = 0.99)
    tf_transformer = TfidfTransformer(use_idf=False)
    svd_model = TruncatedSVD(n_components=50)
    pipeline = Pipeline([
                         ('vectorize',count_vect),
                         ('tf-idf',tf_transformer),
                         ('svd',svd_model)
                         ])
    svd_train = pipeline.fit_transform(train_data.data)
    svd_test = pipeline.transform(test_data.data)
    return svd_train, svd_test

def fetch_NMF(train_data, test_data):
    stop_words = text.ENGLISH_STOP_WORDS
    count_vect = CountVectorizer(tokenizer=project1b.tokenizer_class(), stop_words = stop_words, lowercase= True, min_df = 2, max_df = 0.99)
    tf_transformer = TfidfTransformer(use_idf=False)
    nmf_model = NMF(n_components=50)
    pipeline = Pipeline([
                         ('vectorize',count_vect),
                         ('tf-idf',tf_transformer),
                         ('nmf',nmf_model)
                         ])
    nmf_train = pipeline.fit_transform(train_data.data)
    nmf_test = pipeline.transform(test_data.data)
    return nmf_train, nmf_test

if __name__ == "__main__":
    stop_words = text.ENGLISH_STOP_WORDS
    categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

    train_data = fetch_20newsgroups(subset='train', categories=categories,shuffle=True, random_state=42)
    test_data = fetch_20newsgroups(subset='test', categories=categories,shuffle=True, random_state=42)
    

    svd_train, svd_test = fetch_LSI(train_data, test_data)
    print ('size of LSI train', svd_train.shape)
    print ('size of LSI test', svd_test.shape)

    nmf_train, nmf_test = fetch_NMF(train_data, test_data)
    print ('size of NMF train', nmf_train.shape)
    print ('size of NMF test', nmf_test.shape)




