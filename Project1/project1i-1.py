
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
import sklearn.metrics
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
    
    svd_train, svd_test = project1d.fetch_LSI(train_data, test_data)
    

    #range
    params = list(range(-3,4))

    #accuracy
    l1_accuracies=[]
    l2_accuracies=[]

    #coefficient
    l1_coefficient = []
    l2_coefficient = []

    for param in params:

        l1_classifier = LogisticRegression(penalty = 'l1', C = 10 ** param, solver = 'liblinear')
        l1_classifier.fit(svd_train, train_data.target)
        l1_predicted = l1_classifier.predict(svd_test)
        l1_accuracies.append(sklearn.metrics.accuracy_score(test_data.target, l1_predicted) * 100)
        l1_coefficient.append(np.mean(l1_classifier.coef_))


        l2_classifier = LogisticRegression(penalty = 'l2', C = 10 ** param, solver = 'liblinear')
        l2_classifier.fit(svd_train, train_data.target)
        l2_predicted = l2_classifier.predict(svd_test)
        l2_accuracies.append(sklearn.metrics.accuracy_score(test_data.target, l2_predicted) * 100)
        l2_coefficient.append(np.mean(l2_classifier.coef_))


    for count, param in enumerate(params):
        print ("Regularization parameter set to ", param)
        print ("Accuracy with L1 Regularization is ", l1_accuracies[count])
        print ("Mean of coefficients is ", l1_coefficient[count])

        print ("Accuracy with L2 Regularization is ", l2_accuracies[count])
        print ("Mean of coefficients is ", l2_coefficient[count])
 

	#plot
    plt.plot(l1_accuracies, label = 'l1')
    plt.plot(l2_accuracies, label = 'l2')
    plt.title("L1/L2 Regularized Logistic Regression vs Regularization Parameter")	
    plt.xlabel('Regularization Parameter')
    plt.ylabel('Accuracy')
    plt.xticks(range(6), [10 ** param for param in params])
    plt.show()
    plt.clf()

    



